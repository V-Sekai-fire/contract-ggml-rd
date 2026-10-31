#include "ggml-pjrt.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "program.h"

#define PJRT_ARGS(T, name) \
	T name; \
	std::memset(&name, 0, sizeof name); \
	name.struct_size = T##_STRUCT_SIZE

namespace {

struct Entry {
	size_t bytes = 0;
	PJRT_Buffer *buf = nullptr; // the bytes on the device, when they are there
	std::vector<uint8_t> stage; // the bytes on the host, when they are newer
	bool staged = false;
	int fill = -1; // >= 0: the content is this byte repeated, until a write
};

struct Buf {
	uintptr_t base = 0;
	size_t size = 0;
	std::map<uintptr_t, Entry> entries;
};

struct Stats {
	int64_t uploads = 0, upload_bytes = 0, readbacks = 0, readback_bytes = 0, rmw = 0, temps = 0;
	int64_t compiles = 0, cache_hits = 0, executes = 0, buffers_live = 0;
};

struct Ctx {
	const PJRT_Api *api = nullptr;
	PJRT_Client *client = nullptr;
	PJRT_Device *device = nullptr;
	uint64_t next_index = 0;
	std::string last_error;
	Stats st;
	std::unordered_map<std::string, PJRT_LoadedExecutable *> exes;
};

Ctx &ctx() {
	static Ctx c;
	return c;
}

void set_error(const std::string &e) {
	ctx().last_error = e;
	GGML_LOG_ERROR("ggml-pjrt: %s\n", e.c_str());
}

std::string take(PJRT_Error *e) {
	if (e == nullptr) {
		return std::string();
	}
	const PJRT_Api *api = ctx().api;
	PJRT_ARGS(PJRT_Error_Message_Args, m);
	m.error = e;
	api->PJRT_Error_Message(&m);
	std::string s(m.message, m.message_size);
	PJRT_ARGS(PJRT_Error_Destroy_Args, d);
	d.error = e;
	api->PJRT_Error_Destroy(&d);
	return s.empty() ? std::string("(no text)") : s;
}

bool must(PJRT_Error *e, const char *what) {
	const std::string s = take(e);
	if (!s.empty()) {
		set_error(std::string(what) + ": " + s);
		return false;
	}
	return true;
}

bool await_destroy(PJRT_Event *ev, const char *what) {
	if (ev == nullptr) {
		return true;
	}
	const PJRT_Api *api = ctx().api;
	PJRT_ARGS(PJRT_Event_Await_Args, w);
	w.event = ev;
	const bool ok = must(api->PJRT_Event_Await(&w), what);
	PJRT_ARGS(PJRT_Event_Destroy_Args, d);
	d.event = ev;
	take(api->PJRT_Event_Destroy(&d));
	return ok;
}

void drop_buffer(PJRT_Buffer *b) {
	if (b == nullptr) {
		return;
	}
	PJRT_ARGS(PJRT_Buffer_Destroy_Args, d);
	d.buffer = b;
	take(ctx().api->PJRT_Buffer_Destroy(&d));
}

PJRT_Buffer *upload_bytes(const void *data, size_t bytes) {
	Ctx &c = ctx();
	PJRT_ARGS(PJRT_Client_BufferFromHostBuffer_Args, a);
	const int64_t dim = int64_t(bytes);
	a.client = c.client;
	a.data = data;
	a.type = PJRT_Buffer_Type_U8;
	a.dims = &dim;
	a.num_dims = 1;
	a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
	a.device = c.device;
	if (!must(c.api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer")) {
		return nullptr;
	}
	if (!await_destroy(a.done_with_host_buffer, "BufferFromHostBuffer completion")) {
		drop_buffer(a.buffer);
		return nullptr;
	}
	++c.st.uploads;
	c.st.upload_bytes += int64_t(bytes);
	return a.buffer;
}

bool read_range(PJRT_Buffer *b, void *dst, size_t off, size_t size) {
	Ctx &c = ctx();
	PJRT_ARGS(PJRT_Buffer_CopyRawToHost_Args, a);
	a.buffer = b;
	a.dst = dst;
	a.offset = int64_t(off);
	a.transfer_size = int64_t(size);
	if (!must(c.api->PJRT_Buffer_CopyRawToHost(&a), "CopyRawToHost")) {
		return false;
	}
	++c.st.readbacks;
	c.st.readback_bytes += int64_t(size);
	return await_destroy(a.event, "CopyRawToHost completion");
}

// --- entries -----------------------------------------------------------------------------------

void drop_entry(Entry &e) {
	drop_buffer(e.buf);
	e.buf = nullptr;
	e.stage.clear();
	e.stage.shrink_to_fit();
	e.staged = false;
}

// The staging copy becomes the truth: the device bytes are read back first unless a
// caller is about to overwrite them all.
bool stage_entry(Entry &e) {
	if (e.staged) {
		return true;
	}
	if (e.buf != nullptr) {
		e.stage.resize(e.bytes);
		if (!read_range(e.buf, e.stage.data(), 0, e.bytes)) {
			e.stage.clear();
			return false;
		}
		++ctx().st.rmw;
		drop_buffer(e.buf);
		e.buf = nullptr;
	} else {
		e.stage.assign(e.bytes, uint8_t(e.fill >= 0 ? e.fill : 0));
	}
	e.staged = true;
	e.fill = -1;
	return true;
}

bool entry_set(Entry &e, size_t off, const void *data, size_t size) {
	if (off == 0 && size == e.bytes) {
		drop_buffer(e.buf);
		e.buf = nullptr;
		e.stage.resize(e.bytes);
		std::memcpy(e.stage.data(), data, size);
		e.staged = true;
		e.fill = -1;
		return true;
	}
	if (!stage_entry(e)) {
		return false;
	}
	std::memcpy(e.stage.data() + off, data, size);
	return true;
}

bool entry_get(Entry &e, size_t off, void *dst, size_t size) {
	if (e.staged) {
		std::memcpy(dst, e.stage.data() + off, size);
		return true;
	}
	if (e.buf != nullptr) {
		return read_range(e.buf, dst, off, size);
	}
	std::memset(dst, e.fill >= 0 ? e.fill : 0, size);
	return true;
}

// The entry's bytes as a device buffer, uploading what the host holds.
PJRT_Buffer *entry_buffer(Entry &e) {
	if (e.staged) {
		PJRT_Buffer *b = upload_bytes(e.stage.data(), e.bytes);
		if (b == nullptr) {
			return nullptr;
		}
		e.buf = b;
		e.staged = false;
		e.stage.clear();
		e.stage.shrink_to_fit();
		e.fill = -1;
	} else if (e.buf == nullptr) {
		const std::vector<uint8_t> zero(e.bytes, uint8_t(e.fill >= 0 ? e.fill : 0));
		e.buf = upload_bytes(zero.data(), e.bytes);
		e.fill = -1;
	}
	return e.buf;
}

Entry *find_entry(Buf *b, uintptr_t addr, size_t size, uintptr_t *start) {
	std::map<uintptr_t, Entry>::iterator it = b->entries.upper_bound(addr);
	if (it == b->entries.begin()) {
		return nullptr;
	}
	--it;
	if (addr + size > it->first + it->second.bytes) {
		return nullptr;
	}
	if (start != nullptr) {
		*start = it->first;
	}
	return &it->second;
}

// A new tensor owns its range: every entry it overlaps is dead.
Entry &register_entry(Buf *b, uintptr_t start, size_t bytes) {
	std::map<uintptr_t, Entry>::iterator it = b->entries.lower_bound(start);
	if (it != b->entries.begin()) {
		std::map<uintptr_t, Entry>::iterator prev = it;
		--prev;
		if (prev->first + prev->second.bytes > start) {
			it = prev;
		}
	}
	while (it != b->entries.end() && it->first < start + bytes) {
		drop_entry(it->second);
		it = b->entries.erase(it);
	}
	Entry &e = b->entries[start];
	e.bytes = bytes;
	return e;
}

// --- buffer ---------------------------------------------------------------------------------------

ggml_backend_buffer_type_t buft();

Buf *buf_of(ggml_backend_buffer_t b) {
	return static_cast<Buf *>(b->context);
}

Buf *buf_of_tensor(const ggml_tensor *t) {
	ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
	if (b == nullptr || b->buft != buft()) {
		return nullptr;
	}
	return buf_of(b);
}

void buffer_free(ggml_backend_buffer_t buffer) {
	Buf *b = buf_of(buffer);
	for (std::pair<const uintptr_t, Entry> &p : b->entries) {
		drop_entry(p.second);
	}
	--ctx().st.buffers_live;
	delete b;
}

void *buffer_get_base(ggml_backend_buffer_t buffer) {
	return reinterpret_cast<void *>(buf_of(buffer)->base);
}

enum ggml_status buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
	if (tensor->view_src == nullptr && ggml_nbytes(tensor) > 0) {
		register_entry(buf_of(buffer), uintptr_t(tensor->data), ggml_nbytes(tensor));
	}
	return GGML_STATUS_SUCCESS;
}

Entry *locate(ggml_backend_buffer_t buffer, const ggml_tensor *t, size_t offset, size_t size, size_t *inner, const char *what) {
	Buf *b = buf_of(buffer);
	const uintptr_t addr = uintptr_t(t->data) + offset;
	uintptr_t start = 0;
	Entry *e = find_entry(b, addr, size, &start);
	if (e == nullptr) {
		set_error(std::string(what) + " " + t->name + ": [" + std::to_string(offset) + ", +" + std::to_string(size)
				+ ") is outside every tensor of the buffer");
		return nullptr;
	}
	*inner = size_t(addr - start);
	return e;
}

void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor, const void *data, size_t offset, size_t size) {
	size_t inner = 0;
	Entry *e = locate(buffer, tensor, offset, size, &inner, "set_tensor");
	if (e != nullptr && !entry_set(*e, inner, data, size)) {
		set_error(std::string("set_tensor ") + tensor->name + " failed");
	}
}

void buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor *tensor, void *data, size_t offset, size_t size) {
	size_t inner = 0;
	Entry *e = locate(buffer, tensor, offset, size, &inner, "get_tensor");
	if (e == nullptr || !entry_get(*e, inner, data, size)) {
		std::memset(data, 0, size);
	}
}

void buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor, uint8_t value, size_t offset, size_t size) {
	std::vector<uint8_t> v(size, value);
	buffer_set_tensor(buffer, tensor, v.data(), offset, size);
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
	for (std::pair<const uintptr_t, Entry> &p : buf_of(buffer)->entries) {
		drop_entry(p.second);
		p.second.fill = value;
	}
}

const ggml_backend_buffer_i &buffer_iface() {
	static const ggml_backend_buffer_i i = [] {
		ggml_backend_buffer_i x{};
		x.free_buffer = buffer_free;
		x.get_base = buffer_get_base;
		x.init_tensor = buffer_init_tensor;
		x.memset_tensor = buffer_memset_tensor;
		x.set_tensor = buffer_set_tensor;
		x.get_tensor = buffer_get_tensor;
		x.cpy_tensor = nullptr;
		x.clear = buffer_clear;
		x.reset = nullptr;
		return x;
	}();
	return i;
}

// --- buffer type -------------------------------------------------------------------------------------

const char *buft_get_name(ggml_backend_buffer_type_t) {
	return "PJRT";
}

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t bt, size_t size) {
	Ctx &c = ctx();
	if (c.client == nullptr) {
		set_error("alloc_buffer: no PJRT client is attached");
		return nullptr;
	}
	Buf *b = new Buf;
	b->size = size;
	b->base = uintptr_t(0x1000) + (uintptr_t(c.next_index++) << 40);
	++c.st.buffers_live;
	return ggml_backend_buffer_init(bt, buffer_iface(), b, size);
}

size_t buft_get_alignment(ggml_backend_buffer_type_t) {
	return 256;
}

size_t buft_get_max_size(ggml_backend_buffer_type_t) {
	return size_t(1) << 39;
}

size_t buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor *t) {
	return (ggml_nbytes(t) + 15) & ~size_t(15);
}

bool buft_is_host(ggml_backend_buffer_type_t) {
	return false;
}

// --- device ----------------------------------------------------------------------------------------------

ggml_backend_device &device_obj();
ggml_backend_reg &reg_obj();

ggml_backend_buffer_type_t buft() {
	static ggml_backend_buffer_type t = [] {
		ggml_backend_buffer_type x{};
		x.iface.get_name = buft_get_name;
		x.iface.alloc_buffer = buft_alloc_buffer;
		x.iface.get_alignment = buft_get_alignment;
		x.iface.get_max_size = buft_get_max_size;
		x.iface.get_alloc_size = buft_get_alloc_size;
		x.iface.is_host = buft_is_host;
		x.device = &device_obj();
		x.context = nullptr;
		return x;
	}();
	return &t;
}

const char *dev_get_name(ggml_backend_dev_t) {
	return "PJRT0";
}

const char *dev_get_description(ggml_backend_dev_t) {
	return "PJRT C API plugin";
}

void dev_get_memory(ggml_backend_dev_t, size_t *free, size_t *total) {
	*free = 0;
	*total = 0;
}

enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t) {
	return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void dev_get_props(ggml_backend_dev_t d, ggml_backend_dev_props *props) {
	std::memset(props, 0, sizeof(*props));
	props->name = dev_get_name(d);
	props->description = dev_get_description(d);
	props->type = dev_get_type(d);
	dev_get_memory(d, &props->memory_free, &props->memory_total);
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t d, const char *);

ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t) {
	return buft();
}

// A graph that writes into one of its own leaves is refused: the leaf's new bytes would
// have to come back from the device into an entry the caller already holds.
bool writes_a_leaf(const ggml_tensor *t) {
	switch (t->op) {
		case GGML_OP_NONE:
		case GGML_OP_VIEW:
		case GGML_OP_RESHAPE:
		case GGML_OP_PERMUTE:
		case GGML_OP_TRANSPOSE:
			return false;
		default:
			break;
	}
	if (t->view_src == nullptr) {
		return false;
	}
	const ggml_tensor *root = t->view_src;
	while (root->view_src != nullptr) {
		root = root->view_src;
	}
	return root->op == GGML_OP_NONE && root != t;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor *op) {
	return !writes_a_leaf(op);
}

bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t bt) {
	return bt == buft();
}

bool dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) {
	return false;
}

ggml_backend_device &device_obj() {
	static ggml_backend_device d = [] {
		ggml_backend_device x{};
		x.iface.get_name = dev_get_name;
		x.iface.get_description = dev_get_description;
		x.iface.get_memory = dev_get_memory;
		x.iface.get_type = dev_get_type;
		x.iface.get_props = dev_get_props;
		x.iface.init_backend = dev_init_backend;
		x.iface.get_buffer_type = dev_get_buffer_type;
		x.iface.supports_op = dev_supports_op;
		x.iface.supports_buft = dev_supports_buft;
		x.iface.offload_op = dev_offload_op;
		x.reg = &reg_obj();
		x.context = nullptr;
		return x;
	}();
	return d;
}

// --- backend --------------------------------------------------------------------------------------------------

ggml_guid *backend_guid() {
	static ggml_guid g = { 0x50, 0x4a, 0x52, 0x54, 0x2d, 0x67, 0x67, 0x6d, 0x6c, 0x2d, 0x76, 0x73, 0x6b, 0x61, 0x69, 0x01 };
	return &g;
}

const char *backend_get_name(ggml_backend_t) {
	return "PJRT0";
}

void backend_free(ggml_backend_t backend) {
	delete backend;
}

void backend_synchronize(ggml_backend_t) {}

PJRT_LoadedExecutable *compiled(const std::string &text) {
	Ctx &c = ctx();
	std::unordered_map<std::string, PJRT_LoadedExecutable *>::iterator it = c.exes.find(text);
	if (it != c.exes.end()) {
		++c.st.cache_hits;
		return it->second;
	}
	PJRT_Program prog;
	std::memset(&prog, 0, sizeof prog);
	prog.struct_size = PJRT_Program_STRUCT_SIZE;
	prog.code = const_cast<char *>(text.data());
	prog.code_size = text.size();
	prog.format = "mlir";
	prog.format_size = 4;
	PJRT_ARGS(PJRT_Client_Compile_Args, cc);
	cc.client = c.client;
	cc.program = &prog;
	if (!must(c.api->PJRT_Client_Compile(&cc), "Compile")) {
		return nullptr;
	}
	++c.st.compiles;
	if (c.exes.size() >= 64) {
		for (std::pair<const std::string, PJRT_LoadedExecutable *> &p : c.exes) {
			PJRT_ARGS(PJRT_LoadedExecutable_Destroy_Args, d);
			d.executable = p.second;
			take(c.api->PJRT_LoadedExecutable_Destroy(&d));
		}
		c.exes.clear();
	}
	c.exes[text] = cc.executable;
	return cc.executable;
}

enum ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph *cgraph) {
	Ctx &c = ctx();
	const int n_nodes = ggml_graph_n_nodes(cgraph);
	if (n_nodes == 0) {
		return GGML_STATUS_SUCCESS;
	}
	std::vector<ggml_tensor *> outs;
	for (int i = 0; i < n_nodes; ++i) {
		ggml_tensor *t = ggml_graph_node(cgraph, i);
		if (writes_a_leaf(t)) {
			set_error(std::string("graph_compute: ") + ggml_op_name(t->op) + " '" + t->name + "' writes into one of the graph's leaves");
			return GGML_STATUS_FAILED;
		}
		if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
			outs.push_back(t);
		}
	}
	if (outs.empty()) {
		outs.push_back(ggml_graph_node(cgraph, n_nodes - 1));
	}
	std::vector<ggml_tensor *> args;
	const std::string text = pjrt_rd::serialize(outs, args);
	if (text.empty()) {
		set_error("graph_compute: the graph could not be serialized");
		return GGML_STATUS_FAILED;
	}
	PJRT_LoadedExecutable *exe = compiled(text);
	if (exe == nullptr) {
		return GGML_STATUS_FAILED;
	}
	std::vector<PJRT_Buffer *> argv;
	std::vector<PJRT_Buffer *> temps;
	bool ok = true;
	for (ggml_tensor *a : args) {
		Buf *b = buf_of_tensor(a);
		const size_t nb = ggml_nbytes(a);
		uintptr_t start = 0;
		Entry *e = b != nullptr ? find_entry(b, uintptr_t(a->data), nb, &start) : nullptr;
		if (e == nullptr) {
			set_error(std::string("graph_compute: argument '") + a->name + "' is not in a pjrt buffer");
			ok = false;
			break;
		}
		PJRT_Buffer *pb = nullptr;
		if (start == uintptr_t(a->data)) {
			pb = entry_buffer(*e);
		} else {
			std::vector<uint8_t> part(nb);
			if (entry_get(*e, size_t(uintptr_t(a->data) - start), part.data(), nb)) {
				pb = upload_bytes(part.data(), nb);
				if (pb != nullptr) {
					temps.push_back(pb);
					++c.st.temps;
				}
			}
		}
		if (pb == nullptr) {
			ok = false;
			break;
		}
		argv.push_back(pb);
	}
	std::vector<PJRT_Buffer *> results(outs.size(), nullptr);
	if (ok) {
		PJRT_ARGS(PJRT_ExecuteOptions, opt);
		PJRT_Buffer *const *arg_list = argv.data();
		PJRT_Buffer **out_list = results.data();
		PJRT_Event *done[1] = { nullptr };
		PJRT_ARGS(PJRT_LoadedExecutable_Execute_Args, ex);
		ex.executable = exe;
		ex.options = &opt;
		ex.argument_lists = &arg_list;
		ex.num_devices = 1;
		ex.num_args = argv.size();
		ex.output_lists = &out_list;
		ex.device_complete_events = done;
		ok = must(c.api->PJRT_LoadedExecutable_Execute(&ex), "Execute");
		ok = await_destroy(done[0], "Execute completion") && ok;
		++c.st.executes;
	}
	for (PJRT_Buffer *t : temps) {
		drop_buffer(t);
	}
	for (size_t k = 0; k < outs.size(); ++k) {
		if (!ok || results[k] == nullptr) {
			drop_buffer(results[k]);
			continue;
		}
		Buf *b = buf_of_tensor(outs[k]);
		if (b == nullptr) {
			set_error(std::string("graph_compute: result '") + outs[k]->name + "' is not in a pjrt buffer");
			drop_buffer(results[k]);
			ok = false;
			continue;
		}
		Entry &e = register_entry(b, uintptr_t(outs[k]->data), ggml_nbytes(outs[k]));
		e.buf = results[k];
	}
	return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t d, const char *) {
	if (ctx().client == nullptr) {
		return nullptr;
	}
	ggml_backend_i i{};
	i.get_name = backend_get_name;
	i.free = backend_free;
	i.synchronize = backend_synchronize;
	i.graph_compute = backend_graph_compute;
	return new ggml_backend{ backend_guid(), i, d, nullptr };
}

// --- registry ------------------------------------------------------------------------------------------------------

const char *reg_get_name(ggml_backend_reg_t) {
	return "PJRT";
}

size_t reg_get_device_count(ggml_backend_reg_t) {
	return ctx().client != nullptr ? 1 : 0;
}

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t, size_t index) {
	GGML_ASSERT(index == 0 && ctx().client != nullptr);
	return &device_obj();
}

ggml_backend_reg &reg_obj() {
	static ggml_backend_reg r = [] {
		ggml_backend_reg x{};
		x.api_version = GGML_BACKEND_API_VERSION;
		x.iface.get_name = reg_get_name;
		x.iface.get_device_count = reg_get_device_count;
		x.iface.get_device = reg_get_device;
		x.iface.get_proc_address = nullptr;
		x.context = nullptr;
		return x;
	}();
	return r;
}

} // namespace

ggml_backend_reg_t ggml_backend_pjrt_reg(void) {
	return &reg_obj();
}

bool ggml_backend_pjrt_attach(const PJRT_Api *api) {
	Ctx &c = ctx();
	if (c.client != nullptr) {
		set_error("attach: a client is already attached");
		return false;
	}
	c.api = api;
	c.st = Stats();
	PJRT_ARGS(PJRT_Plugin_Initialize_Args, pi);
	if (!must(api->PJRT_Plugin_Initialize(&pi), "Plugin_Initialize")) {
		return false;
	}
	PJRT_ARGS(PJRT_Client_Create_Args, cc);
	if (!must(api->PJRT_Client_Create(&cc), "Client_Create")) {
		return false;
	}
	PJRT_ARGS(PJRT_Client_AddressableDevices_Args, dv);
	dv.client = cc.client;
	if (!must(api->PJRT_Client_AddressableDevices(&dv), "Client_AddressableDevices") || dv.num_addressable_devices == 0) {
		set_error("attach: the client has no addressable device");
		return false;
	}
	c.client = cc.client;
	c.device = dv.addressable_devices[0];
	return true;
}

void ggml_backend_pjrt_detach(void) {
	Ctx &c = ctx();
	if (c.api == nullptr) {
		return;
	}
	for (std::pair<const std::string, PJRT_LoadedExecutable *> &p : c.exes) {
		PJRT_ARGS(PJRT_LoadedExecutable_Destroy_Args, d);
		d.executable = p.second;
		take(c.api->PJRT_LoadedExecutable_Destroy(&d));
	}
	c.exes.clear();
	if (c.client != nullptr) {
		PJRT_ARGS(PJRT_Client_Destroy_Args, d);
		d.client = c.client;
		take(c.api->PJRT_Client_Destroy(&d));
	}
	c.client = nullptr;
	c.device = nullptr;
	c.api = nullptr;
}

bool ggml_backend_is_pjrt(ggml_backend_t backend) {
	return backend != nullptr && ggml_guid_matches(backend->guid, backend_guid());
}

std::string ggml_backend_pjrt_last_error(void) {
	return ctx().last_error;
}

std::string ggml_backend_pjrt_stats(void) {
	const Stats &s = ctx().st;
	char b[512];
	std::snprintf(b, sizeof b,
			"uploads=%lld upload_bytes=%lld readbacks=%lld readback_bytes=%lld rmw=%lld temps=%lld compiles=%lld "
			"cache_hits=%lld executes=%lld buffers_live=%lld",
			(long long)s.uploads, (long long)s.upload_bytes, (long long)s.readbacks, (long long)s.readback_bytes,
			(long long)s.rmw, (long long)s.temps, (long long)s.compiles, (long long)s.cache_hits, (long long)s.executes,
			(long long)s.buffers_live);
	return b;
}
