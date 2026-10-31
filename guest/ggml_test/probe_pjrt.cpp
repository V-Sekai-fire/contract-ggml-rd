// The pjrt-rd gates (probe "pjrt"), driven only through the PJRT C API:
//
//   pjrt buffers          a 117,440,512-byte buffer (Kimodo's text-layer FFN weight,
//                         4096 x 14336 bf16) uploaded and read back whole and by range,
//                         across the 16 MiB transfer chunk; out-of-range reads fail
//                         without writing; each check has a control that must fail.
//   pjrt <graph>          kimodo_denoiser or kimodo_text: the graph serialized to a
//                         module, compiled and executed by the plugin on the argument
//                         buffers, every result bit for bit against the same graph run
//                         directly on ggml-rd; controls: a changed op, a garbage module
//                         and a wrong argument count must each be refused or differ.
#include <api.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "graph_nets.h"
#include "pjrt_rd.h"
#include "program.h"
#include "rd_compute.h"

namespace probes {

namespace {

using namespace graph_nets;

#define PJRT_ARGS(T, name) \
	T name; \
	std::memset(&name, 0, sizeof name); \
	name.struct_size = T##_STRUCT_SIZE

const PJRT_Api *g_api = nullptr;

// Consumes the error. Returns its text, empty when there was none.
std::string take(PJRT_Error *e, PJRT_Error_Code *code = nullptr) {
	if (e == nullptr) {
		return std::string();
	}
	PJRT_ARGS(PJRT_Error_Message_Args, m);
	m.error = e;
	g_api->PJRT_Error_Message(&m);
	std::string s(m.message, m.message_size);
	if (code != nullptr) {
		PJRT_ARGS(PJRT_Error_GetCode_Args, c);
		c.error = e;
		g_api->PJRT_Error_GetCode(&c);
		*code = c.code;
	}
	PJRT_ARGS(PJRT_Error_Destroy_Args, d);
	d.error = e;
	g_api->PJRT_Error_Destroy(&d);
	return s.empty() ? std::string("(no text)") : s;
}

bool must(PJRT_Error *e, const char *what) {
	const std::string s = take(e);
	if (!s.empty()) {
		std::printf("PROBE pjrt: %s failed: %s\n", what, s.c_str());
		return false;
	}
	return true;
}

// Expects a refusal with `code`. False, and a line, if the call succeeded or used another code.
bool refused(PJRT_Error *e, PJRT_Error_Code want, const char *what) {
	PJRT_Error_Code code = PJRT_Error_Code_OK;
	const std::string s = take(e, &code);
	if (s.empty()) {
		std::printf("PROBE pjrt: %s was accepted; it must be refused\n", what);
		return false;
	}
	if (code != want) {
		std::printf("PROBE pjrt: %s refused with code %d, wanted %d (%s)\n", what, int(code), int(want), s.c_str());
		return false;
	}
	return true;
}

PJRT_Client *make_client() {
	PJRT_ARGS(PJRT_Plugin_Initialize_Args, pi);
	if (!must(g_api->PJRT_Plugin_Initialize(&pi), "Plugin_Initialize")) {
		return nullptr;
	}
	PJRT_ARGS(PJRT_Client_Create_Args, cc);
	if (!must(g_api->PJRT_Client_Create(&cc), "Client_Create")) {
		return nullptr;
	}
	PJRT_ARGS(PJRT_Client_Devices_Args, dv);
	dv.client = cc.client;
	if (!must(g_api->PJRT_Client_Devices(&dv), "Client_Devices") || dv.num_devices != 1) {
		std::printf("PROBE pjrt: the client has %zu devices, wanted 1\n", dv.num_devices);
		return nullptr;
	}
	return cc.client;
}

void destroy_client(PJRT_Client *c) {
	PJRT_ARGS(PJRT_Client_Destroy_Args, d);
	d.client = c;
	take(g_api->PJRT_Client_Destroy(&d));
}

PJRT_Buffer *upload(PJRT_Client *c, const void *data, size_t bytes) {
	PJRT_ARGS(PJRT_Client_BufferFromHostBuffer_Args, a);
	const int64_t dim = int64_t(bytes);
	a.client = c;
	a.data = data;
	a.type = PJRT_Buffer_Type_U8;
	a.dims = &dim;
	a.num_dims = 1;
	a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
	if (!must(g_api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer")) {
		return nullptr;
	}
	PJRT_ARGS(PJRT_Event_Destroy_Args, ed);
	ed.event = a.done_with_host_buffer;
	take(g_api->PJRT_Event_Destroy(&ed));
	return a.buffer;
}

void drop(PJRT_Buffer *b) {
	PJRT_ARGS(PJRT_Buffer_Destroy_Args, d);
	d.buffer = b;
	take(g_api->PJRT_Buffer_Destroy(&d));
}

PJRT_Error *copy_range(PJRT_Buffer *b, void *dst, int64_t off, int64_t size) {
	PJRT_ARGS(PJRT_Buffer_CopyRawToHost_Args, a);
	a.buffer = b;
	a.dst = dst;
	a.offset = off;
	a.transfer_size = size;
	PJRT_Error *e = g_api->PJRT_Buffer_CopyRawToHost(&a);
	if (e == nullptr && a.event != nullptr) {
		PJRT_ARGS(PJRT_Event_Destroy_Args, ed);
		ed.event = a.event;
		take(g_api->PJRT_Event_Destroy(&ed));
	}
	return e;
}

uint8_t pattern(size_t i) {
	return uint8_t(((uint64_t(i) * 2654435761ull) >> 11) ^ (i >> 7));
}

// Mismatches between bytes read at [off, off + size) and the pattern shifted by `skew`.
size_t mismatches(const std::vector<uint8_t> &got, size_t off, size_t skew) {
	size_t bad = 0;
	for (size_t i = 0; i < got.size(); ++i) {
		bad += got[i] != pattern(off + i + skew);
	}
	return bad;
}

bool probe_buffers() {
	constexpr size_t kBytes = size_t(4096) * 14336 * 2;
	constexpr size_t kChunk = size_t(16) << 20;
	PJRT_Client *client = make_client();
	if (client == nullptr) {
		return false;
	}
	bool ok = true;
	std::vector<uint8_t> host(kBytes);
	for (size_t i = 0; i < kBytes; ++i) {
		host[i] = pattern(i);
	}
	PJRT_Buffer *b = upload(client, host.data(), kBytes);
	host.clear();
	host.shrink_to_fit();
	if (b == nullptr) {
		destroy_client(client);
		return false;
	}
	PJRT_ARGS(PJRT_Buffer_OnDeviceSizeInBytes_Args, sz);
	sz.buffer = b;
	ok = must(g_api->PJRT_Buffer_OnDeviceSizeInBytes(&sz), "OnDeviceSizeInBytes") && ok;
	if (sz.on_device_size_in_bytes != kBytes) {
		std::printf("PROBE pjrt buffers: on-device size %zu, wanted %zu\n", sz.on_device_size_in_bytes, kBytes);
		ok = false;
	}
	struct Range {
		const char *what;
		size_t off, size;
	};
	const Range ranges[] = {
		{ "head", 0, 4096 },
		{ "across the first chunk edge", kChunk - 100, 300 },
		{ "one whole chunk", kChunk, kChunk },
		{ "across the sixth chunk edge", 6 * kChunk - 7, 4099 },
		{ "tail", kBytes - 5000, 5000 },
	};
	size_t total_bad = 0;
	for (const Range &r : ranges) {
		std::vector<uint8_t> got(r.size, 0xA5);
		ok = must(copy_range(b, got.data(), int64_t(r.off), int64_t(r.size)), r.what) && ok;
		const size_t bad = mismatches(got, r.off, 0);
		total_bad += bad;
		std::printf("PROBE pjrt buffers range %-30s off=%zu size=%zu mismatches=%zu\n", r.what, r.off, r.size, bad);
	}
	std::vector<uint8_t> all(kBytes, 0xA5);
	PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args, th);
	th.src = b;
	th.dst = all.data();
	th.dst_size = kBytes;
	ok = must(g_api->PJRT_Buffer_ToHostBuffer(&th), "ToHostBuffer") && ok;
	const size_t whole_bad = mismatches(all, 0, 0);
	total_bad += whole_bad;
	std::printf("PROBE pjrt buffers whole buffer mismatches=%zu of %zu\n", whole_bad, kBytes);
	all.clear();
	all.shrink_to_fit();

	// Each control must show what its check exists to catch.
	std::vector<uint8_t> probe(4096, 0xA5);
	ok = must(copy_range(b, probe.data(), int64_t(kChunk - 100), 300), "control read") && ok;
	probe.resize(300);
	const size_t skewed = mismatches(probe, kChunk - 100, 1);
	std::printf("PROBE pjrt buffers control skewed expectation mismatches=%zu (must be > 0)\n", skewed);
	std::vector<uint8_t> shortd(300, 0xA5);
	ok = must(copy_range(b, shortd.data(), int64_t(kChunk - 100), 299), "short read") && ok;
	const size_t shorts = mismatches(shortd, kChunk - 100, 0);
	std::printf("PROBE pjrt buffers control copy one byte short mismatches=%zu (must be > 0)\n", shorts);
	if (skewed == 0 || shorts == 0) {
		ok = false;
	}

	std::vector<uint8_t> poison(64, 0xA5);
	bool oob = true;
	oob = refused(copy_range(b, poison.data(), int64_t(kBytes - 10), 11), PJRT_Error_Code_OUT_OF_RANGE, "a read one byte past the end") && oob;
	oob = refused(copy_range(b, poison.data(), -1, 8), PJRT_Error_Code_OUT_OF_RANGE, "a negative offset") && oob;
	oob = refused(copy_range(b, poison.data(), int64_t(kBytes) + 1, 0), PJRT_Error_Code_OUT_OF_RANGE, "an offset past the end") && oob;
	bool untouched = true;
	for (uint8_t v : poison) {
		untouched = untouched && v == 0xA5;
	}
	std::vector<uint8_t> small(16, 0xA5);
	PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args, ts);
	ts.src = b;
	ts.dst = small.data();
	ts.dst_size = small.size();
	oob = refused(g_api->PJRT_Buffer_ToHostBuffer(&ts), PJRT_Error_Code_INVALID_ARGUMENT, "ToHostBuffer into a destination that is too small") && oob;
	for (uint8_t v : small) {
		untouched = untouched && v == 0xA5;
	}
	std::printf("PROBE pjrt buffers refused out-of-range reads=%d destinations untouched=%d\n", oob ? 1 : 0, untouched ? 1 : 0);

	PJRT_ARGS(PJRT_Buffer_Delete_Args, del);
	del.buffer = b;
	ok = must(g_api->PJRT_Buffer_Delete(&del), "Buffer_Delete") && ok;
	const bool after_delete = refused(copy_range(b, poison.data(), 0, 8), PJRT_Error_Code_FAILED_PRECONDITION, "a read of a deleted buffer");
	drop(b);
	destroy_client(client);
	ok = ok && total_bad == 0 && oob && untouched && after_delete;
	std::printf("PROBE pjrt buffers SUMMARY bytes=%zu mismatches=%zu refused=%d untouched=%d deleted_refused=%d\n", kBytes,
			total_bad, oob ? 1 : 0, untouched ? 1 : 0, after_delete ? 1 : 0);
	return ok;
}

struct Run {
	bool ok = false;
	std::vector<std::vector<uint8_t>> out;
	std::string error;
};

Run execute(PJRT_Client *client, const std::string &text, const std::vector<PJRT_Buffer *> &args, size_t n_outs) {
	Run r;
	PJRT_Program prog;
	std::memset(&prog, 0, sizeof prog);
	prog.struct_size = PJRT_Program_STRUCT_SIZE;
	prog.code = const_cast<char *>(text.data());
	prog.code_size = text.size();
	prog.format = "mlir";
	prog.format_size = 4;
	PJRT_ARGS(PJRT_Client_Compile_Args, cc);
	cc.client = client;
	cc.program = &prog;
	r.error = take(g_api->PJRT_Client_Compile(&cc));
	if (!r.error.empty()) {
		return r;
	}
	PJRT_ARGS(PJRT_ExecuteOptions, opt);
	PJRT_Buffer *const *arg_list = args.data();
	std::vector<PJRT_Buffer *> outs(n_outs, nullptr);
	PJRT_Buffer **out_list = outs.data();
	PJRT_ARGS(PJRT_LoadedExecutable_Execute_Args, ex);
	ex.executable = cc.executable;
	ex.options = &opt;
	ex.argument_lists = &arg_list;
	ex.num_devices = 1;
	ex.num_args = args.size();
	ex.output_lists = &out_list;
	r.error = take(g_api->PJRT_LoadedExecutable_Execute(&ex));
	if (r.error.empty()) {
		r.ok = true;
		for (PJRT_Buffer *o : outs) {
			PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args, q);
			q.src = o;
			take(g_api->PJRT_Buffer_ToHostBuffer(&q));
			std::vector<uint8_t> v(q.dst_size);
			PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args, th);
			th.src = o;
			th.dst = v.data();
			th.dst_size = v.size();
			if (!must(g_api->PJRT_Buffer_ToHostBuffer(&th), "ToHostBuffer(result)")) {
				r.ok = false;
			}
			r.out.push_back(std::move(v));
			drop(o);
		}
	}
	PJRT_ARGS(PJRT_LoadedExecutable_Destroy_Args, ld);
	ld.executable = cc.executable;
	take(g_api->PJRT_LoadedExecutable_Destroy(&ld));
	return r;
}

bool probe_graph_through_pjrt(const std::string &which) {
	BuildFn build;
	std::string native, resid;
	if (!graph_builder(which, 0, build, native, resid)) {
		std::printf("PROBE pjrt: unknown graph '%s'\n", which.c_str());
		return false;
	}
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		std::printf("PROBE pjrt %s: no RD device\n", which.c_str());
		return false;
	}
	const int64_t t0 = rdc::host_usec();
	Net n;
	n.be = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
	n.rd = true;
	build(n, false);
	if (!allocate(n)) {
		return false;
	}
	ggml_backend_rd_ensure_idle();
	if (ggml_backend_graph_compute(n.be, n.gf) != GGML_STATUS_SUCCESS) {
		std::printf("PROBE pjrt %s: the direct run failed\n", which.c_str());
		return false;
	}
	const Outs direct = read_outs(n);
	const int64_t t_direct = rdc::host_usec();

	std::vector<ggml_tensor *> outs;
	for (const std::pair<std::string, ggml_tensor *> &o : n.outs) {
		outs.push_back(o.second);
	}
	std::vector<ggml_tensor *> args;
	const std::string text = pjrt_rd::serialize(outs, args);

	pjrt_rd::attach(n.be);
	g_api = GetPjrtApi();
	PJRT_Client *client = make_client();
	if (client == nullptr) {
		return false;
	}
	std::vector<PJRT_Buffer *> bufs;
	size_t arg_bytes = 0;
	for (ggml_tensor *a : args) {
		std::vector<uint8_t> h(ggml_nbytes(a));
		ggml_backend_tensor_get(a, h.data(), 0, h.size());
		PJRT_Buffer *b = upload(client, h.data(), h.size());
		if (b == nullptr) {
			return false;
		}
		bufs.push_back(b);
		arg_bytes += h.size();
	}
	const int64_t t_up = rdc::host_usec();

	const Run r = execute(client, text, bufs, outs.size());
	const int64_t t_run = rdc::host_usec();
	size_t same = 0;
	if (r.ok && r.out.size() == outs.size()) {
		for (size_t k = 0; k < outs.size(); ++k) {
			const std::vector<float> &want = direct[k].second;
			same += r.out[k].size() == want.size() * 4 && std::memcmp(r.out[k].data(), want.data(), r.out[k].size()) == 0;
		}
	}
	std::printf("PROBE pjrt %s result: outputs=%zu identical=%zu args=%zu arg_bytes=%zu module_bytes=%zu error=%s\n", which.c_str(),
			outs.size(), same, args.size(), arg_bytes, text.size(), r.error.empty() ? "none" : r.error.c_str());

	// Controls: each must be refused or must change a result.
	bool controls = true;
	size_t tried = 0, moved = 0;
	const char *kinds[2][2] = { { "ggml.ADD\"", "ggml.SUB\"" }, { "ggml.MUL\"", "ggml.ADD\"" } };
	for (const char *const *kind : kinds) {
		const std::string needle = std::string("call_target_name = \"") + kind[0];
		for (size_t at = text.find(needle); at != std::string::npos && tried < 48; at = text.find(needle, at + 1)) {
			std::string changed = text;
			changed.replace(at + std::strlen("call_target_name = \""), std::strlen(kind[0]), kind[1]);
			const Run c = execute(client, changed, bufs, outs.size());
			bool moved_one = !c.ok || c.out.size() != r.out.size();
			for (size_t k = 0; !moved_one && k < c.out.size(); ++k) {
				moved_one = c.out[k] != r.out[k];
			}
			++tried;
			moved += moved_one ? 1 : 0;
		}
	}
	std::printf("PROBE pjrt %s control: %zu of %zu changed ops (ADD to SUB, MUL to ADD) changed a result (must be > 0)\n", which.c_str(),
			moved, tried);
	controls = controls && moved > 0;
	{
		PJRT_Program prog;
		std::memset(&prog, 0, sizeof prog);
		prog.struct_size = PJRT_Program_STRUCT_SIZE;
		const std::string junk = "module @other { }";
		prog.code = const_cast<char *>(junk.data());
		prog.code_size = junk.size();
		prog.format = "mlir";
		prog.format_size = 4;
		PJRT_ARGS(PJRT_Client_Compile_Args, cc);
		cc.client = client;
		cc.program = &prog;
		controls = refused(g_api->PJRT_Client_Compile(&cc), PJRT_Error_Code_INVALID_ARGUMENT, "Compile of a module that is not a ggml graph") && controls;
	}
	{
		std::vector<PJRT_Buffer *> fewer(bufs.begin(), bufs.end() - 1);
		const Run c = execute(client, text, fewer, outs.size());
		const bool refused_count = !c.ok && !c.error.empty();
		std::printf("PROBE pjrt %s control one argument short: %s\n", which.c_str(), refused_count ? "refused" : "ACCEPTED");
		controls = controls && refused_count;
	}
	for (PJRT_Buffer *b : bufs) {
		drop(b);
	}
	destroy_client(client);
	pjrt_rd::attach(nullptr);
	ggml_backend_free(n.be);
	n.be = nullptr;
	const bool ok = r.ok && same == outs.size() && controls;
	std::printf("PROBE pjrt %s SUMMARY identical=%zu/%zu controls=%s direct_ms=%.1f upload_ms=%.1f compile_execute_ms=%.1f\n",
			which.c_str(), same, outs.size(), controls ? "held" : "FAILED", (t_direct - t0) / 1000.0, (t_up - t_direct) / 1000.0,
			(t_run - t_up) / 1000.0);
	return ok;
}

} // namespace

bool pjrt_probe(const std::string &arg) {
	if (arg == "buffers") {
		ggml_backend_reg_t reg = ggml_backend_rd_reg();
		if (ggml_backend_reg_dev_count(reg) == 0) {
			std::printf("PROBE pjrt buffers: no RD device\n");
			return false;
		}
		ggml_backend_t be = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
		pjrt_rd::attach(be);
		g_api = GetPjrtApi();
		const bool ok = probe_buffers();
		pjrt_rd::attach(nullptr);
		ggml_backend_free(be);
		return ok;
	}
	return probe_graph_through_pjrt(arg);
}

} // namespace probes
