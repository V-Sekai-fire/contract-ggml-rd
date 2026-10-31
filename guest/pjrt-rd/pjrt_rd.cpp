#include "pjrt_rd.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "program.h"
#include "stablehlo.h"

namespace {

ggml_backend_t g_backend = nullptr;
std::string g_last_error;

} // namespace

struct RdError : PJRT_Error {
	PJRT_Error_Code code;
	std::string message;
};

struct PJRT_Event {
	int unused;
};

struct UserData {
	const void *key;
	void *data;
	void (*dtor)(void *);
};

struct RdMemory : PJRT_Memory {
	int id;
	std::vector<UserData> user;
	~RdMemory() {
		for (const UserData &u : user) {
			if (u.dtor != nullptr) {
				u.dtor(u.data);
			}
		}
	}
};

struct PJRT_DeviceDescription {
	int id;
};

struct PJRT_Device {
	PJRT_DeviceDescription description;
	PJRT_Memory *memory;
};

struct PJRT_Client {
	PJRT_Device device;
	RdMemory memory;
	std::vector<PJRT_Device *> devices;
	std::vector<PJRT_Memory *> memories;
};

struct PJRT_Buffer {
	PJRT_Client *client;
	ggml_context *ctx;
	ggml_tensor *tensor;
	ggml_backend_buffer_t buffer;
	size_t bytes;
	PJRT_Buffer_Type type;
	std::vector<int64_t> dims;
	bool deleted;
};

struct PJRT_Executable {
	std::shared_ptr<pjrt_rd::Program> program;
	std::string name;
};

struct PJRT_LoadedExecutable {
	PJRT_Client *client;
	std::shared_ptr<pjrt_rd::Program> program;
	PJRT_Executable executable;
	bool deleted;
};

namespace {

void error_destroy(PJRT_Error *e) {
	delete static_cast<RdError *>(e);
}

void error_message(const PJRT_Error *e, const char **message, size_t *size) {
	const RdError *r = static_cast<const RdError *>(e);
	*message = r->message.c_str();
	*size = r->message.size();
}

PJRT_Error_Code error_code(const PJRT_Error *e) {
	return static_cast<const RdError *>(e)->code;
}

void error_payload(const PJRT_Error *, PJRT_Error_PayloadVisitor, void *) {}

const PJRT_Error_FunctionTable kErrorTable = { PJRT_Error_FunctionTable_STRUCT_SIZE, sizeof(RdError), nullptr, error_destroy,
	error_message, error_code, error_payload };

void *memory_get_user_data(PJRT_Memory *m, const void *key) {
	for (const UserData &u : static_cast<RdMemory *>(m)->user) {
		if (u.key == key) {
			return u.data;
		}
	}
	return nullptr;
}

void memory_set_user_data(PJRT_Memory *m, const void *key, void *data, void (*dtor)(void *)) {
	std::vector<UserData> &v = static_cast<RdMemory *>(m)->user;
	for (UserData &u : v) {
		if (u.key == key) {
			if (u.dtor != nullptr) {
				u.dtor(u.data);
			}
			u.data = data;
			u.dtor = dtor;
			return;
		}
	}
	v.push_back(UserData{ key, data, dtor });
}

const PJRT_Memory_FunctionTable kMemoryTable = { PJRT_Memory_FunctionTable_STRUCT_SIZE, nullptr, PJRT_Memory_STRUCT_SIZE,
	memory_get_user_data, memory_set_user_data };

PJRT_Error *make_error(PJRT_Error_Code code, const std::string &message) {
	g_last_error = message;
	RdError *e = new RdError;
	e->vtable = &kErrorTable;
	e->code = code;
	e->message = message;
	return e;
}

PJRT_Error *unimplemented(const char *name) {
	return make_error(PJRT_Error_Code_UNIMPLEMENTED, std::string(name) + " is not implemented by pjrt-rd");
}

#include "pjrt_rd_stubs.inc"

size_t element_bytes(PJRT_Buffer_Type t) {
	switch (t) {
		case PJRT_Buffer_Type_PRED:
		case PJRT_Buffer_Type_S8:
		case PJRT_Buffer_Type_U8:
			return 1;
		case PJRT_Buffer_Type_S16:
		case PJRT_Buffer_Type_U16:
		case PJRT_Buffer_Type_F16:
		case PJRT_Buffer_Type_BF16:
			return 2;
		case PJRT_Buffer_Type_S32:
		case PJRT_Buffer_Type_U32:
		case PJRT_Buffer_Type_F32:
			return 4;
		case PJRT_Buffer_Type_S64:
		case PJRT_Buffer_Type_U64:
		case PJRT_Buffer_Type_F64:
		case PJRT_Buffer_Type_C64:
			return 8;
		case PJRT_Buffer_Type_C128:
			return 16;
		default:
			return 0;
	}
}

PJRT_Event *ready_event() {
	return new PJRT_Event{ 0 };
}

// A buffer of `bytes` on the device. `like` gives the layout of a result (the
// buffer then holds exactly one tensor of that layout); null makes a byte array.
PJRT_Buffer *new_buffer(PJRT_Client *client, size_t bytes, const ggml_tensor *like, PJRT_Buffer_Type type,
		const std::vector<int64_t> &dims, std::string &err) {
	if (g_backend == nullptr) {
		err = "no ggml-rd backend is attached";
		return nullptr;
	}
	if (bytes == 0) {
		err = "a buffer of zero bytes";
		return nullptr;
	}
	ggml_init_params ip = { ggml_tensor_overhead() * 2 + 256, nullptr, true };
	ggml_context *ctx = ggml_init(ip);
	if (ctx == nullptr) {
		err = "ggml_init failed";
		return nullptr;
	}
	ggml_tensor *t = like != nullptr ? ggml_dup_tensor(ctx, like) : ggml_new_tensor_1d(ctx, GGML_TYPE_I8, int64_t(bytes));
	ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(g_backend, ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(g_backend), t));
	if (buf == nullptr) {
		ggml_free(ctx);
		err = "device allocation of " + std::to_string(bytes) + " bytes failed";
		return nullptr;
	}
	if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS) {
		ggml_backend_buffer_free(buf);
		ggml_free(ctx);
		err = "placing the buffer's tensor failed";
		return nullptr;
	}
	PJRT_Buffer *b = new PJRT_Buffer;
	b->client = client;
	b->ctx = ctx;
	b->tensor = t;
	b->buffer = buf;
	b->bytes = ggml_nbytes(t);
	b->type = type;
	b->dims = dims;
	b->deleted = false;
	return b;
}

void free_buffer(PJRT_Buffer *b) {
	if (b == nullptr) {
		return;
	}
	if (!b->deleted) {
		ggml_backend_buffer_free(b->buffer);
		ggml_free(b->ctx);
	}
	delete b;
}

// --- errors, events, plugin ---------------------------------------------------------

void rd_error_destroy(PJRT_Error_Destroy_Args *a) {
	if (a->error != nullptr) {
		a->error->vtable->destroy(a->error);
	}
}

PJRT_Error *rd_plugin_initialize(PJRT_Plugin_Initialize_Args *) {
	return nullptr;
}

PJRT_Error *rd_plugin_attributes(PJRT_Plugin_Attributes_Args *a) {
	a->attributes = nullptr;
	a->num_attributes = 0;
	return nullptr;
}

PJRT_Error *rd_event_destroy(PJRT_Event_Destroy_Args *a) {
	delete a->event;
	return nullptr;
}

PJRT_Error *rd_event_is_ready(PJRT_Event_IsReady_Args *a) {
	a->is_ready = true;
	return nullptr;
}

PJRT_Error *rd_event_error(PJRT_Event_Error_Args *) {
	return nullptr;
}

PJRT_Error *rd_event_await(PJRT_Event_Await_Args *) {
	return nullptr;
}

PJRT_Error *rd_event_on_ready(PJRT_Event_OnReady_Args *a) {
	a->callback(nullptr, a->user_arg);
	return nullptr;
}

// --- client, devices, memories -------------------------------------------------------

PJRT_Error *rd_client_create(PJRT_Client_Create_Args *a) {
	if (g_backend == nullptr) {
		return make_error(PJRT_Error_Code_FAILED_PRECONDITION, "no ggml-rd backend is attached; call pjrt_rd::attach first");
	}
	PJRT_Client *c = new PJRT_Client;
	c->memory.vtable = &kMemoryTable;
	c->memory.id = 0;
	c->device.description.id = 0;
	c->device.memory = &c->memory;
	c->devices.push_back(&c->device);
	c->memories.push_back(&c->memory);
	a->client = c;
	return nullptr;
}

PJRT_Error *rd_client_destroy(PJRT_Client_Destroy_Args *a) {
	delete a->client;
	return nullptr;
}

PJRT_Error *rd_client_platform_name(PJRT_Client_PlatformName_Args *a) {
	a->platform_name = "rd";
	a->platform_name_size = 2;
	return nullptr;
}

PJRT_Error *rd_client_process_index(PJRT_Client_ProcessIndex_Args *a) {
	a->process_index = 0;
	return nullptr;
}

PJRT_Error *rd_client_platform_version(PJRT_Client_PlatformVersion_Args *a) {
	a->platform_version = "ggml-rd over RenderingDevice";
	a->platform_version_size = std::strlen(a->platform_version);
	return nullptr;
}

PJRT_Error *rd_client_devices(PJRT_Client_Devices_Args *a) {
	a->devices = a->client->devices.data();
	a->num_devices = a->client->devices.size();
	return nullptr;
}

PJRT_Error *rd_client_addressable_devices(PJRT_Client_AddressableDevices_Args *a) {
	a->addressable_devices = a->client->devices.data();
	a->num_addressable_devices = a->client->devices.size();
	return nullptr;
}

PJRT_Error *rd_client_addressable_memories(PJRT_Client_AddressableMemories_Args *a) {
	a->addressable_memories = a->client->memories.data();
	a->num_addressable_memories = a->client->memories.size();
	return nullptr;
}

PJRT_Error *rd_desc_id(PJRT_DeviceDescription_Id_Args *a) {
	a->id = a->device_description->id;
	return nullptr;
}

PJRT_Error *rd_desc_process_index(PJRT_DeviceDescription_ProcessIndex_Args *a) {
	a->process_index = 0;
	return nullptr;
}

PJRT_Error *rd_desc_attributes(PJRT_DeviceDescription_Attributes_Args *a) {
	a->attributes = nullptr;
	a->num_attributes = 0;
	return nullptr;
}

PJRT_Error *rd_desc_kind(PJRT_DeviceDescription_Kind_Args *a) {
	a->device_kind = "rd";
	a->device_kind_size = 2;
	return nullptr;
}

PJRT_Error *rd_desc_debug_string(PJRT_DeviceDescription_DebugString_Args *a) {
	a->debug_string = "rd:0";
	a->debug_string_size = 4;
	return nullptr;
}

PJRT_Error *rd_desc_to_string(PJRT_DeviceDescription_ToString_Args *a) {
	a->to_string = "rd:0";
	a->to_string_size = 4;
	return nullptr;
}

PJRT_Error *rd_device_get_description(PJRT_Device_GetDescription_Args *a) {
	a->device_description = &a->device->description;
	return nullptr;
}

PJRT_Error *rd_device_is_addressable(PJRT_Device_IsAddressable_Args *a) {
	a->is_addressable = true;
	return nullptr;
}

PJRT_Error *rd_device_local_hardware_id(PJRT_Device_LocalHardwareId_Args *a) {
	a->local_hardware_id = 0;
	return nullptr;
}

PJRT_Error *rd_device_addressable_memories(PJRT_Device_AddressableMemories_Args *a) {
	a->memories = &a->device->memory;
	a->num_memories = 1;
	return nullptr;
}

PJRT_Error *rd_device_default_memory(PJRT_Device_DefaultMemory_Args *a) {
	a->memory = a->device->memory;
	return nullptr;
}

PJRT_Error *rd_memory_id(PJRT_Memory_Id_Args *a) {
	a->id = static_cast<RdMemory *>(a->memory)->id;
	return nullptr;
}

PJRT_Error *rd_memory_kind(PJRT_Memory_Kind_Args *a) {
	a->kind = "device";
	a->kind_size = 6;
	return nullptr;
}

PJRT_Error *rd_memory_debug_string(PJRT_Memory_DebugString_Args *a) {
	a->debug_string = "rd:0/device";
	a->debug_string_size = 11;
	return nullptr;
}

PJRT_Error *rd_memory_to_string(PJRT_Memory_ToString_Args *a) {
	a->to_string = "rd:0/device";
	a->to_string_size = 11;
	return nullptr;
}

PJRT_Error *rd_memory_addressable_by_devices(PJRT_Memory_AddressableByDevices_Args *a) {
	a->devices = nullptr;
	a->num_devices = 0;
	return nullptr;
}

// --- buffers ----------------------------------------------------------------------------

PJRT_Error *rd_client_buffer_from_host(PJRT_Client_BufferFromHostBuffer_Args *a) {
	const size_t eb = element_bytes(a->type);
	if (eb == 0) {
		return make_error(PJRT_Error_Code_UNIMPLEMENTED, "BufferFromHostBuffer: element type " + std::to_string(int(a->type)) + " is not supported");
	}
	if (a->data == nullptr) {
		return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "BufferFromHostBuffer: no host data");
	}
	size_t count = 1;
	for (size_t i = 0; i < a->num_dims; ++i) {
		if (a->dims[i] < 0) {
			return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "BufferFromHostBuffer: a negative dimension");
		}
		count *= size_t(a->dims[i]);
	}
	if (a->num_byte_strides != 0) {
		size_t want = eb;
		for (size_t i = a->num_dims; i-- > 0;) {
			if (size_t(a->byte_strides[i]) != want) {
				return make_error(PJRT_Error_Code_UNIMPLEMENTED, "BufferFromHostBuffer: only dense row-major host data is supported");
			}
			want *= size_t(a->dims[i]);
		}
	}
	std::string err;
	const std::vector<int64_t> dims(a->dims, a->dims + a->num_dims);
	PJRT_Buffer *b = new_buffer(a->client, count * eb, nullptr, a->type, dims, err);
	if (b == nullptr) {
		return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "BufferFromHostBuffer: " + err);
	}
	ggml_backend_tensor_set(b->tensor, a->data, 0, b->bytes);
	a->buffer = b;
	a->done_with_host_buffer = ready_event();
	return nullptr;
}

PJRT_Error *rd_buffer_destroy(PJRT_Buffer_Destroy_Args *a) {
	free_buffer(a->buffer);
	return nullptr;
}

PJRT_Error *rd_buffer_element_type(PJRT_Buffer_ElementType_Args *a) {
	a->type = a->buffer->type;
	return nullptr;
}

PJRT_Error *rd_buffer_dimensions(PJRT_Buffer_Dimensions_Args *a) {
	a->dims = a->buffer->dims.data();
	a->num_dims = a->buffer->dims.size();
	return nullptr;
}

PJRT_Error *rd_buffer_unpadded_dimensions(PJRT_Buffer_UnpaddedDimensions_Args *a) {
	a->unpadded_dims = a->buffer->dims.data();
	a->num_dims = a->buffer->dims.size();
	return nullptr;
}

PJRT_Error *rd_buffer_dynamic_dimension_indices(PJRT_Buffer_DynamicDimensionIndices_Args *a) {
	a->dynamic_dim_indices = nullptr;
	a->num_dynamic_dims = 0;
	return nullptr;
}

PJRT_Error *rd_buffer_on_device_size(PJRT_Buffer_OnDeviceSizeInBytes_Args *a) {
	a->on_device_size_in_bytes = a->buffer->bytes;
	return nullptr;
}

PJRT_Error *rd_buffer_device(PJRT_Buffer_Device_Args *a) {
	a->device = &a->buffer->client->device;
	return nullptr;
}

PJRT_Error *rd_buffer_memory(PJRT_Buffer_Memory_Args *a) {
	a->memory = &a->buffer->client->memory;
	return nullptr;
}

PJRT_Error *rd_buffer_delete(PJRT_Buffer_Delete_Args *a) {
	PJRT_Buffer *b = a->buffer;
	if (!b->deleted) {
		ggml_backend_buffer_free(b->buffer);
		ggml_free(b->ctx);
		b->deleted = true;
	}
	return nullptr;
}

PJRT_Error *rd_buffer_is_deleted(PJRT_Buffer_IsDeleted_Args *a) {
	a->is_deleted = a->buffer->deleted;
	return nullptr;
}

PJRT_Error *rd_buffer_is_on_cpu(PJRT_Buffer_IsOnCpu_Args *a) {
	a->is_on_cpu = false;
	return nullptr;
}

PJRT_Error *rd_buffer_ready_event(PJRT_Buffer_ReadyEvent_Args *a) {
	a->event = ready_event();
	return nullptr;
}

PJRT_Error *rd_buffer_to_host(PJRT_Buffer_ToHostBuffer_Args *a) {
	PJRT_Buffer *b = a->src;
	if (b->deleted) {
		return make_error(PJRT_Error_Code_FAILED_PRECONDITION, "ToHostBuffer: the buffer was deleted");
	}
	if (a->dst == nullptr) {
		a->dst_size = b->bytes;
		a->event = nullptr;
		return nullptr;
	}
	if (a->dst_size < b->bytes) {
		return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
				"ToHostBuffer: dst_size " + std::to_string(a->dst_size) + " is below the buffer's " + std::to_string(b->bytes));
	}
	ggml_backend_tensor_get(b->tensor, a->dst, 0, b->bytes);
	a->event = ready_event();
	return nullptr;
}

PJRT_Error *rd_buffer_copy_raw_to_host(PJRT_Buffer_CopyRawToHost_Args *a) {
	PJRT_Buffer *b = a->buffer;
	if (b->deleted) {
		return make_error(PJRT_Error_Code_FAILED_PRECONDITION, "CopyRawToHost: the buffer was deleted");
	}
	if (a->offset < 0 || a->transfer_size < 0 || size_t(a->offset) > b->bytes
			|| size_t(a->transfer_size) > b->bytes - size_t(a->offset)) {
		return make_error(PJRT_Error_Code_OUT_OF_RANGE,
				"CopyRawToHost: [" + std::to_string((long long)a->offset) + ", +" + std::to_string((long long)a->transfer_size)
						+ ") leaves the buffer of " + std::to_string(b->bytes) + " bytes");
	}
	if (a->transfer_size > 0) {
		ggml_backend_tensor_get(b->tensor, a->dst, size_t(a->offset), size_t(a->transfer_size));
	}
	a->event = ready_event();
	return nullptr;
}

// --- executables ---------------------------------------------------------------------------

PJRT_Error *rd_client_compile(PJRT_Client_Compile_Args *a) {
	const PJRT_Program *prog = a->program;
	if (prog == nullptr || prog->code == nullptr) {
		return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Compile: no program");
	}
	const std::string format(prog->format, prog->format_size);
	if (format != "mlir") {
		return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Compile: format '" + format + "' is not supported; pass \"mlir\"");
	}
	std::shared_ptr<pjrt_rd::Program> p = std::make_shared<pjrt_rd::Program>();
	std::string err;
	const std::string code(prog->code, prog->code_size);
	if (pjrt_rd::is_stablehlo(code)) {
		if (!pjrt_rd::lower_stablehlo(code, *p, err)) {
			return make_error(err.compare(0, 11, "unsupported") == 0 ? PJRT_Error_Code_UNIMPLEMENTED : PJRT_Error_Code_INVALID_ARGUMENT,
					"Compile: " + err);
		}
	} else if (!pjrt_rd::parse(code, *p, err)) {
		return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Compile: " + err);
	}
	PJRT_LoadedExecutable *le = new PJRT_LoadedExecutable;
	le->client = a->client;
	le->program = p;
	le->executable.program = p;
	le->executable.name = "ggml_graph";
	le->deleted = false;
	a->executable = le;
	return nullptr;
}

PJRT_Error *rd_executable_destroy(PJRT_Executable_Destroy_Args *a) {
	if (a->executable != nullptr) {
		delete a->executable;
	}
	return nullptr;
}

PJRT_Error *rd_executable_name(PJRT_Executable_Name_Args *a) {
	a->executable_name = a->executable->name.c_str();
	a->executable_name_size = a->executable->name.size();
	return nullptr;
}

PJRT_Error *rd_executable_num_replicas(PJRT_Executable_NumReplicas_Args *a) {
	a->num_replicas = 1;
	return nullptr;
}

PJRT_Error *rd_executable_num_partitions(PJRT_Executable_NumPartitions_Args *a) {
	a->num_partitions = 1;
	return nullptr;
}

PJRT_Error *rd_executable_num_outputs(PJRT_Executable_NumOutputs_Args *a) {
	a->num_outputs = a->executable->program->outs.size();
	return nullptr;
}

PJRT_Error *rd_executable_code_size(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *a) {
	a->size_in_bytes = 0;
	return nullptr;
}

PJRT_Error *rd_loaded_destroy(PJRT_LoadedExecutable_Destroy_Args *a) {
	delete a->executable;
	return nullptr;
}

PJRT_Error *rd_loaded_get_executable(PJRT_LoadedExecutable_GetExecutable_Args *a) {
	PJRT_Executable *e = new PJRT_Executable;
	e->program = a->loaded_executable->program;
	e->name = a->loaded_executable->executable.name;
	a->executable = e;
	return nullptr;
}

PJRT_Error *rd_loaded_addressable_devices(PJRT_LoadedExecutable_AddressableDevices_Args *a) {
	a->addressable_devices = a->executable->client->devices.data();
	a->num_addressable_devices = a->executable->client->devices.size();
	return nullptr;
}

PJRT_Error *rd_loaded_delete(PJRT_LoadedExecutable_Delete_Args *a) {
	a->executable->deleted = true;
	return nullptr;
}

PJRT_Error *rd_loaded_is_deleted(PJRT_LoadedExecutable_IsDeleted_Args *a) {
	a->is_deleted = a->executable->deleted;
	return nullptr;
}

PJRT_Error *rd_loaded_execute(PJRT_LoadedExecutable_Execute_Args *a) {
	PJRT_LoadedExecutable *le = a->executable;
	const pjrt_rd::Program &prog = *le->program;
	if (le->deleted) {
		return make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Execute: the executable was deleted");
	}
	if (a->num_devices != 1) {
		return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Execute: one device only");
	}
	if (a->num_args != prog.args.size()) {
		return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
				"Execute: " + std::to_string(a->num_args) + " arguments for a program of " + std::to_string(prog.args.size()));
	}
	PJRT_Buffer *const *argv = a->argument_lists[0];
	const size_t n = prog.t.size();
	ggml_init_params ip = { ggml_tensor_overhead() * (n + 8) + ggml_graph_overhead_custom(n + 8, false) + 4096, nullptr, true };
	ggml_context *ctx = ggml_init(ip);
	if (ctx == nullptr) {
		return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Execute: ggml_init failed");
	}
	std::vector<ggml_tensor *> t(n, nullptr);
	std::string fail;
	size_t next_arg = 0;
	std::vector<int> const_at(n, -1);
	std::vector<size_t> const_off(prog.consts.size(), 0);
	size_t const_total = 0;
	for (size_t k = 0; k < prog.consts.size(); ++k) {
		const_at[size_t(prog.consts[k].id)] = int(k);
		const_off[k] = const_total;
		const_total += (prog.consts[k].bytes.size() + 255) & ~size_t(255);
	}
	ggml_backend_buffer_t const_buf = nullptr;
	if (const_total != 0) {
		const_buf = ggml_backend_alloc_buffer(g_backend, const_total);
		if (const_buf == nullptr) {
			fail = "Execute: allocating " + std::to_string(const_total) + " bytes for the module's constants failed";
		}
	}
	for (size_t i = 0; i < n && fail.empty(); ++i) {
		const pjrt_rd::TensorRec &r = prog.t[i];
		ggml_tensor *x = ggml_new_tensor_4d(ctx, ggml_type(r.type), r.ne[0], r.ne[1], r.ne[2], r.ne[3]);
		for (int d = 0; d < GGML_MAX_DIMS; ++d) {
			x->nb[d] = size_t(r.nb[d]);
		}
		ggml_set_name(x, r.name.c_str());
		t[i] = x;
		if (r.is_arg) {
			PJRT_Buffer *b = argv[next_arg++];
			if (b == nullptr || b->deleted) {
				fail = "Execute: argument " + std::to_string(next_arg - 1) + " is null or deleted";
			} else if (ggml_nbytes(x) > b->bytes || ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(g_backend), x) > ggml_backend_buffer_get_size(b->buffer)) {
				fail = "Execute: argument " + std::to_string(next_arg - 1) + " holds " + std::to_string(b->bytes) + " bytes, '"
						+ r.name + "' needs " + std::to_string(ggml_nbytes(x));
			} else if (ggml_backend_tensor_alloc(b->buffer, x, ggml_backend_buffer_get_base(b->buffer)) != GGML_STATUS_SUCCESS) {
				fail = "Execute: placing argument '" + r.name + "' failed";
			}
			continue;
		}
		if (const_at[i] >= 0) {
			const pjrt_rd::Program::Const &c = prog.consts[size_t(const_at[i])];
			if (ggml_nbytes(x) != c.bytes.size() ||
					ggml_backend_tensor_alloc(const_buf, x, static_cast<char *>(ggml_backend_buffer_get_base(const_buf)) + const_off[size_t(const_at[i])]) != GGML_STATUS_SUCCESS) {
				fail = "Execute: placing constant '" + r.name + "' failed";
			} else {
				ggml_backend_tensor_set(x, c.bytes.data(), 0, c.bytes.size());
			}
			continue;
		}
		x->op = ggml_op(r.op);
		std::memcpy(x->op_params, r.op_params, sizeof x->op_params);
		x->flags = r.flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_PARAM);
		for (int s = 0; s < GGML_MAX_SRC; ++s) {
			x->src[s] = r.src[s] >= 0 ? t[size_t(r.src[s])] : nullptr;
		}
		if (r.view_src >= 0) {
			x->view_src = t[size_t(r.view_src)];
			x->view_offs = size_t(r.view_offs);
			if (x->view_src->buffer != nullptr && ggml_backend_view_init(x) != GGML_STATUS_SUCCESS) {
				fail = "Execute: view '" + r.name + "' could not be placed";
			}
		}
	}
	PJRT_Error *result = nullptr;
	ggml_gallocr_t galloc = nullptr;
	if (!fail.empty()) {
		result = make_error(PJRT_Error_Code_INVALID_ARGUMENT, fail);
	} else {
		ggml_cgraph *gf = ggml_new_graph_custom(ctx, n + 8, false);
		for (int o : prog.outs) {
			ggml_set_output(t[size_t(o)]);
			ggml_build_forward_expand(gf, t[size_t(o)]);
		}
		galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
		if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, gf)) {
			result = make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Execute: allocating the graph failed");
		} else if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
			result = make_error(PJRT_Error_Code_INTERNAL, "Execute: ggml-rd graph_compute failed");
		}
	}
	if (result == nullptr) {
		PJRT_Buffer **outv = a->output_lists[0];
		for (size_t k = 0; k < prog.outs.size() && result == nullptr; ++k) {
			const ggml_tensor *src = t[size_t(prog.outs[k])];
			std::string err;
			if (!ggml_is_contiguous(src)) {
				result = make_error(PJRT_Error_Code_UNIMPLEMENTED, "Execute: result " + std::to_string(k) + " is not contiguous");
				break;
			}
			PJRT_Buffer *ob = new_buffer(le->client, ggml_nbytes(src), src, PJRT_Buffer_Type_U8, { int64_t(ggml_nbytes(src)) }, err);
			if (ob == nullptr) {
				result = make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Execute: result " + std::to_string(k) + ": " + err);
				break;
			}
			ggml_backend_tensor_copy(src, ob->tensor);
			outv[k] = ob;
		}
	}
	if (galloc != nullptr) {
		ggml_gallocr_free(galloc);
	}
	if (const_buf != nullptr) {
		ggml_backend_buffer_free(const_buf);
	}
	ggml_free(ctx);
	if (result == nullptr && a->device_complete_events != nullptr) {
		a->device_complete_events[0] = ready_event();
	}
	return result;
}

PJRT_Api g_api;
bool g_api_ready = false;

void fill_api() {
	std::memset(&g_api, 0, sizeof g_api);
	g_api.struct_size = PJRT_Api_STRUCT_SIZE;
	g_api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
	g_api.pjrt_api_version.major_version = PJRT_API_MAJOR;
	g_api.pjrt_api_version.minor_version = PJRT_API_MINOR;
	fill_stubs(&g_api);
	g_api.PJRT_Error_Destroy = rd_error_destroy;
	g_api.PJRT_Error_Message = [](PJRT_Error_Message_Args *a) { a->error->vtable->message(a->error, &a->message, &a->message_size); };
	g_api.PJRT_Error_GetCode = [](PJRT_Error_GetCode_Args *a) -> PJRT_Error * {
		a->code = a->error->vtable->get_code(a->error);
		return nullptr;
	};
	g_api.PJRT_Plugin_Initialize = rd_plugin_initialize;
	g_api.PJRT_Plugin_Attributes = rd_plugin_attributes;
	g_api.PJRT_Event_Destroy = rd_event_destroy;
	g_api.PJRT_Event_IsReady = rd_event_is_ready;
	g_api.PJRT_Event_Error = rd_event_error;
	g_api.PJRT_Event_Await = rd_event_await;
	g_api.PJRT_Event_OnReady = rd_event_on_ready;
	g_api.PJRT_Client_Create = rd_client_create;
	g_api.PJRT_Client_Destroy = rd_client_destroy;
	g_api.PJRT_Client_PlatformName = rd_client_platform_name;
	g_api.PJRT_Client_ProcessIndex = rd_client_process_index;
	g_api.PJRT_Client_PlatformVersion = rd_client_platform_version;
	g_api.PJRT_Client_Devices = rd_client_devices;
	g_api.PJRT_Client_AddressableDevices = rd_client_addressable_devices;
	g_api.PJRT_Client_AddressableMemories = rd_client_addressable_memories;
	g_api.PJRT_Client_Compile = rd_client_compile;
	g_api.PJRT_Client_BufferFromHostBuffer = rd_client_buffer_from_host;
	g_api.PJRT_DeviceDescription_Id = rd_desc_id;
	g_api.PJRT_DeviceDescription_ProcessIndex = rd_desc_process_index;
	g_api.PJRT_DeviceDescription_Attributes = rd_desc_attributes;
	g_api.PJRT_DeviceDescription_Kind = rd_desc_kind;
	g_api.PJRT_DeviceDescription_DebugString = rd_desc_debug_string;
	g_api.PJRT_DeviceDescription_ToString = rd_desc_to_string;
	g_api.PJRT_Device_GetDescription = rd_device_get_description;
	g_api.PJRT_Device_IsAddressable = rd_device_is_addressable;
	g_api.PJRT_Device_LocalHardwareId = rd_device_local_hardware_id;
	g_api.PJRT_Device_AddressableMemories = rd_device_addressable_memories;
	g_api.PJRT_Device_DefaultMemory = rd_device_default_memory;
	g_api.PJRT_Memory_Id = rd_memory_id;
	g_api.PJRT_Memory_Kind = rd_memory_kind;
	g_api.PJRT_Memory_DebugString = rd_memory_debug_string;
	g_api.PJRT_Memory_ToString = rd_memory_to_string;
	g_api.PJRT_Memory_AddressableByDevices = rd_memory_addressable_by_devices;
	g_api.PJRT_Executable_Destroy = rd_executable_destroy;
	g_api.PJRT_Executable_Name = rd_executable_name;
	g_api.PJRT_Executable_NumReplicas = rd_executable_num_replicas;
	g_api.PJRT_Executable_NumPartitions = rd_executable_num_partitions;
	g_api.PJRT_Executable_NumOutputs = rd_executable_num_outputs;
	g_api.PJRT_Executable_SizeOfGeneratedCodeInBytes = rd_executable_code_size;
	g_api.PJRT_LoadedExecutable_Destroy = rd_loaded_destroy;
	g_api.PJRT_LoadedExecutable_GetExecutable = rd_loaded_get_executable;
	g_api.PJRT_LoadedExecutable_AddressableDevices = rd_loaded_addressable_devices;
	g_api.PJRT_LoadedExecutable_Delete = rd_loaded_delete;
	g_api.PJRT_LoadedExecutable_IsDeleted = rd_loaded_is_deleted;
	g_api.PJRT_LoadedExecutable_Execute = rd_loaded_execute;
	g_api.PJRT_Buffer_Destroy = rd_buffer_destroy;
	g_api.PJRT_Buffer_ElementType = rd_buffer_element_type;
	g_api.PJRT_Buffer_Dimensions = rd_buffer_dimensions;
	g_api.PJRT_Buffer_UnpaddedDimensions = rd_buffer_unpadded_dimensions;
	g_api.PJRT_Buffer_DynamicDimensionIndices = rd_buffer_dynamic_dimension_indices;
	g_api.PJRT_Buffer_OnDeviceSizeInBytes = rd_buffer_on_device_size;
	g_api.PJRT_Buffer_Device = rd_buffer_device;
	g_api.PJRT_Buffer_Memory = rd_buffer_memory;
	g_api.PJRT_Buffer_Delete = rd_buffer_delete;
	g_api.PJRT_Buffer_IsDeleted = rd_buffer_is_deleted;
	g_api.PJRT_Buffer_IsOnCpu = rd_buffer_is_on_cpu;
	g_api.PJRT_Buffer_ReadyEvent = rd_buffer_ready_event;
	g_api.PJRT_Buffer_ToHostBuffer = rd_buffer_to_host;
	g_api.PJRT_Buffer_CopyRawToHost = rd_buffer_copy_raw_to_host;
	g_api_ready = true;
}

} // namespace

extern "C" const PJRT_Api *GetPjrtApi(void) {
	if (!g_api_ready) {
		fill_api();
	}
	return &g_api;
}

namespace pjrt_rd {

void attach(ggml_backend_t backend) {
	g_backend = backend;
}

const char *last_error(void) {
	return g_last_error.c_str();
}

} // namespace pjrt_rd
