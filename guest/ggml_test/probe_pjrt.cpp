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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-pjrt.h"
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


// A reference that is all zeros compares nothing: two zero outputs are bit-identical for any
// plugin. Fail loudly instead.
bool output_is_dead(const Outs &o, const std::string &who) {
	for (const std::pair<std::string, std::vector<float>> &x : o) {
		if (x.first == "__residual_in") {
			continue;
		}
		size_t nz = 0;
		for (float f : x.second) {
			nz += f != 0.0f;
		}
		std::printf("PROBE pjrt %s reference output %s: n=%zu nonzero=%zu\n", who.c_str(), x.first.c_str(), x.second.size(), nz);
		if (nz == 0) {
			std::printf("PROBE pjrt %s FAIL: the reference output %s is all zeros, so a match proves nothing "
						"(try the f32 arm: %s:f32)\n", who.c_str(), x.first.c_str(), who.c_str());
			return true;
		}
	}
	return false;
}

// "name:f32" runs the f32-weight arm of the graph, "name" the bf16 one.
std::string split_arm(const std::string &which, bool *f32) {
	const std::string suffix = ":f32";
	*f32 = which.size() > suffix.size() && which.compare(which.size() - suffix.size(), suffix.size(), suffix) == 0;
	return *f32 ? which.substr(0, which.size() - suffix.size()) : which;
}

bool probe_graph_through_pjrt(const std::string &arg_which) {
	bool f32 = false;
	const std::string which = split_arm(arg_which, &f32);
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
	build(n, f32);
	if (!allocate(n)) {
		return false;
	}
	ggml_backend_rd_ensure_idle();
	if (ggml_backend_graph_compute(n.be, n.gf) != GGML_STATUS_SUCCESS) {
		std::printf("PROBE pjrt %s: the direct run failed\n", which.c_str());
		return false;
	}
	const Outs direct = read_outs(n);
	if (output_is_dead(direct, "graph " + arg_which)) {
		return false;
	}
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


// A Net on the ggml-pjrt backend (its device is the plugin, whose device is ggml-rd): the same
// leaves and graph as the direct run, every output bit for bit. Controls: a partial write to a
// weight must change the result and writing the bytes back must restore it, and a second
// compute must upload nothing.



// Runs the graph on ggml-rd with every node kept, and prints each node's nonzero count and mean
// magnitude in order, so the first node that is all zeros stands out.
bool probe_bisect(const std::string &arg_which) {
	bool f32 = false;
	const std::string which = split_arm(arg_which, &f32);
	BuildFn build;
	std::string native, resid;
	if (!graph_builder(which, 0, build, native, resid)) {
		std::printf("PROBE pjrt bisect: unknown graph '%s'\n", which.c_str());
		return false;
	}
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	Net n;
	n.be = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
	n.rd = true;
	build(n, f32);
	const int count = ggml_graph_n_nodes(n.gf);
	for (int i = 0; i < count; ++i) {
		ggml_set_output(ggml_graph_node(n.gf, i));
	}
	if (!allocate(n)) {
		return false;
	}
	ggml_backend_rd_ensure_idle();
	if (ggml_backend_graph_compute(n.be, n.gf) != GGML_STATUS_SUCCESS) {
		std::printf("PROBE pjrt bisect %s: the run failed\n", arg_which.c_str());
		return false;
	}
	int first_zero = -1;
	for (int i = 0; i < count; ++i) {
		ggml_tensor *t = ggml_graph_node(n.gf, i);
		const int64_t ne = ggml_nelements(t);
		if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16)) {
			std::printf("BISECT %3d %-12s %-6s ne=[%lld,%lld,%lld,%lld] skipped (%s)\n", i, ggml_op_name(t->op), ggml_type_name(t->type),
					(long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
					ggml_is_contiguous(t) ? "type" : "not contiguous");
			continue;
		}
		std::vector<uint8_t> raw(ggml_nbytes(t));
		ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
		size_t nz = 0, nonfinite = 0;
		double sum = 0.0;
		for (int64_t k = 0; k < ne; ++k) {
			float f = 0.0f;
			if (t->type == GGML_TYPE_F32) {
				std::memcpy(&f, raw.data() + size_t(k) * 4, 4);
			} else {
				ggml_bf16_t b;
				std::memcpy(&b, raw.data() + size_t(k) * 2, 2);
				f = ggml_bf16_to_fp32(b);
			}
			nz += f != 0.0f;
			nonfinite += !std::isfinite(f);
			sum += std::fabs(double(f));
		}
		if (nz == 0 && first_zero < 0) {
			first_zero = i;
		}
		const ggml_tensor *s0 = t->src[0];
		const ggml_tensor *s1 = t->src[1];
		std::printf("BISECT %3d %-12s %-6s ne=[%lld,%lld,%lld,%lld] nonzero=%zu/%lld nonfinite=%zu mean_abs=%.4e srcs=%s/%s\n", i,
				ggml_op_name(t->op), ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
				(long long)t->ne[3], nz, (long long)ne, nonfinite, sum / double(ne > 0 ? ne : 1), s0 ? ggml_op_name(s0->op) : "-",
				s1 ? ggml_op_name(s1->op) : "-");
	}
	std::printf("PROBE pjrt bisect %s nodes=%d first_all_zero_node=%d\n", arg_which.c_str(), count, first_zero);
	ggml_backend_free(n.be);
	return first_zero < 0;
}


// f32 -> bf16 casts on ggml-rd against the host's own rounding, on a bare leaf and in the
// shapes the Kimodo text layer uses.
bool probe_cast(const std::string &variant) {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	Net n;
	n.be = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
	n.rd = true;
	n.lctx = new_ctx(8);
	ggml_tensor *x = leaf(n, "x", GGML_TYPE_F32, { 4096, 16 }, -1.0f, 1.0f);
	n.gctx = new_ctx(64);
	n.gf = ggml_new_graph_custom(n.gctx, 64, false);
	std::vector<ggml_tensor *> nodes;
	ggml_tensor *b = ggml_cast(n.gctx, x, GGML_TYPE_BF16);
	nodes.push_back(b);
	ggml_tensor *last = b;
	if (variant == "roundtrip" || variant == "roundtrip_all") {
		last = ggml_cast(n.gctx, b, GGML_TYPE_F32);
		nodes.push_back(last);
	}
	if (variant == "roundtrip_all") {
		for (ggml_tensor *t : nodes) {
			ggml_set_output(t);
		}
	} else {
		ggml_set_output(last);
	}
	ggml_build_forward_expand(n.gf, last);
	if (!allocate(n)) {
		return false;
	}
	ggml_backend_rd_ensure_idle();
	if (ggml_backend_graph_compute(n.be, n.gf) != GGML_STATUS_SUCCESS) {
		std::printf("PROBE pjrt cast %s: the run failed\n", variant.c_str());
		return false;
	}
	std::vector<float> xs(size_t(ggml_nelements(x)));
	ggml_backend_tensor_get(x, xs.data(), 0, xs.size() * 4);
	std::vector<uint8_t> raw(ggml_nbytes(last));
	ggml_backend_tensor_get(last, raw.data(), 0, raw.size());
	size_t nz = 0, wrong = 0;
	for (size_t k = 0; k < xs.size(); ++k) {
		float got = 0.0f;
		if (last->type == GGML_TYPE_F32) {
			std::memcpy(&got, raw.data() + k * 4, 4);
		} else {
			ggml_bf16_t g;
			std::memcpy(&g, raw.data() + k * 2, 2);
			got = ggml_bf16_to_fp32(g);
		}
		const float want = ggml_bf16_to_fp32(ggml_fp32_to_bf16(xs[k]));
		nz += got != 0.0f;
		wrong += std::memcmp(&got, &want, 4) != 0;
	}
	{
		size_t ev_ok = 0, od_ok = 0;
		for (size_t k = 0; k < xs.size(); ++k) {
			float got = 0.0f;
			if (last->type == GGML_TYPE_F32) {
				std::memcpy(&got, raw.data() + k * 4, 4);
			} else {
				ggml_bf16_t g;
				std::memcpy(&g, raw.data() + k * 2, 2);
				got = ggml_bf16_to_fp32(g);
			}
			const float want = ggml_bf16_to_fp32(ggml_fp32_to_bf16(xs[k]));
			const bool same = std::memcmp(&got, &want, 4) == 0;
			(k % 2 == 0 ? ev_ok : od_ok) += same ? 1 : 0;
		}
		std::printf("PROBE pjrt cast %s: correct even-index elements %zu of %zu, odd-index %zu of %zu\n", variant.c_str(), ev_ok, xs.size() / 2, od_ok, xs.size() / 2);
		if (nodes.size() > 1) {
			std::vector<uint8_t> mid(ggml_nbytes(nodes[0]));
			ggml_backend_tensor_get(nodes[0], mid.data(), 0, mid.size());
			size_t bad = 0;
			for (size_t k = 0; k < xs.size(); ++k) {
				ggml_bf16_t g;
				std::memcpy(&g, mid.data() + k * 2, 2);
				const float want = ggml_bf16_to_fp32(ggml_fp32_to_bf16(xs[k]));
				const float have = ggml_bf16_to_fp32(g);
				bad += std::memcmp(&have, &want, 4) != 0;
			}
			std::printf("PROBE pjrt cast %s: the bf16 intermediate differs from host rounding in %zu elements\n", variant.c_str(), bad);
		}
	}
	std::printf("PROBE pjrt cast %s: type=%s n=%zu nonzero=%zu differs_from_host_rounding=%zu\n", variant.c_str(), ggml_type_name(last->type),
			xs.size(), nz, wrong);
	ggml_backend_free(n.be);
	return wrong == 0 && nz > 0;
}

bool probe_backend(const std::string &arg_which) {
	bool f32 = false;
	const std::string which = split_arm(arg_which, &f32);
	BuildFn build;
	std::string native, resid;
	if (!graph_builder(which, 0, build, native, resid)) {
		std::printf("PROBE pjrt backend: unknown graph '%s'\n", which.c_str());
		return false;
	}
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		std::printf("PROBE pjrt backend %s: no RD device\n", which.c_str());
		return false;
	}
	const int64_t t0 = rdc::host_usec();
	Net direct;
	direct.be = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
	direct.rd = true;
	build(direct, f32);
	if (!allocate(direct)) {
		return false;
	}
	ggml_backend_rd_ensure_idle();
	if (ggml_backend_graph_compute(direct.be, direct.gf) != GGML_STATUS_SUCCESS) {
		std::printf("PROBE pjrt backend %s: the direct run failed\n", which.c_str());
		return false;
	}
	const Outs want = read_outs(direct);
	if (output_is_dead(want, "backend " + arg_which)) {
		return false;
	}
	const int64_t t_direct = rdc::host_usec();
	{
		for (const std::pair<std::string, std::vector<float>> &o : want) {
			size_t nz = 0;
			double sum = 0.0;
			for (float f : o.second) {
				nz += f != 0.0f;
				sum += std::fabs(double(f));
			}
			std::printf("PROBE pjrt backend %s output %-14s n=%zu nonzero=%zu mean_abs=%.4e\n", which.c_str(), o.first.c_str(), o.second.size(), nz,
					o.second.empty() ? 0.0 : sum / double(o.second.size()));
		}
	}

	pjrt_rd::attach(direct.be);
	g_api = GetPjrtApi();
	if (!ggml_backend_pjrt_attach(g_api)) {
		std::printf("PROBE pjrt backend %s: attach failed: %s\n", which.c_str(), ggml_backend_pjrt_last_error().c_str());
		return false;
	}
	bool ok = true;
	{
		Net p;
		p.be = ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_pjrt_reg(), 0), nullptr);
		p.rd = false;
		if (p.be == nullptr) {
			std::printf("PROBE pjrt backend %s: no PJRT device\n", which.c_str());
			return false;
		}
		build(p, f32);
		if (!allocate(p)) {
			std::printf("PROBE pjrt backend %s: allocation failed: %s\n", which.c_str(), ggml_backend_pjrt_last_error().c_str());
			return false;
		}
		const int64_t t_alloc = rdc::host_usec();
		const ggml_status st = ggml_backend_graph_compute(p.be, p.gf);
		const Outs got = read_outs(p);
		size_t differing = 0;
		const bool same = st == GGML_STATUS_SUCCESS && identical(got, want, &differing);
		const std::string after_first = ggml_backend_pjrt_stats();
		std::printf("PROBE pjrt backend %s first compute: status=%d identical=%d differing=%zu leaves=%zu\n", which.c_str(), int(st),
				same ? 1 : 0, differing, p.leaves.size());
		std::printf("PROBE pjrt backend %s stats after the first compute: %s\n", which.c_str(), after_first.c_str());

		const int64_t up1 = [&]() {
			const std::string s = after_first;
			const size_t k = s.find("uploads=");
			return int64_t(std::atoll(s.c_str() + k + 8));
		}();
		const ggml_status st2 = ggml_backend_graph_compute(p.be, p.gf);
		const Outs got2 = read_outs(p);
		const std::string after_second = ggml_backend_pjrt_stats();
		const size_t k2 = after_second.find("uploads=");
		const int64_t up2 = int64_t(std::atoll(after_second.c_str() + k2 + 8));
		const bool again = st2 == GGML_STATUS_SUCCESS && identical(got2, want, nullptr) && up2 == up1;
		std::printf("PROBE pjrt backend %s second compute: identical=%d uploads %lld -> %lld (must not grow)\n", which.c_str(),
				identical(got2, want, nullptr) ? 1 : 0, (long long)up1, (long long)up2);

		size_t tried = 0, moved = 0, restored = 0;
		std::string moved_names;
		for (size_t i = 0; i < p.leaves.size() && tried < 40; ++i) {
			ggml_tensor *w = p.leaves[i].t;
			if (w == p.residual_in || ggml_nbytes(w) < 64 || w->type == GGML_TYPE_I32) {
				continue;
			}
			uint8_t keep[4];
			ggml_backend_tensor_get(w, keep, 8, 4);
			// 1.0 in the leaf's own type: a flipped exponent bit on a zero weight is 1e-19 and vanishes in an f32 sum.
			uint8_t bent[4] = { 0x00, 0x00, 0x80, 0x3f };
			if (w->type == GGML_TYPE_BF16) {
				bent[0] = 0x80;
				bent[1] = 0x3f;
				bent[2] = 0x80;
				bent[3] = 0x3f;
			}
			ggml_backend_tensor_set(w, bent, 8, 4);
			ggml_backend_graph_compute(p.be, p.gf);
			const bool changed = !identical(read_outs(p), want, nullptr);
			{
				const Outs now = read_outs(p);
				double worst = 0.0;
				for (size_t k = 0; k < now[0].second.size(); ++k) {
					worst = std::max(worst, std::fabs(double(now[0].second[k]) - double(want[0].second[k])));
				}
				std::printf("PROBE pjrt backend %s leaf %-34s type=%s bytes=%zu changed=%d worst_abs_diff=%.3e\n", which.c_str(), w->name,
						ggml_type_name(w->type), ggml_nbytes(w), changed ? 1 : 0, worst);
			}
			ggml_backend_tensor_set(w, keep, 8, 4);
			ggml_backend_graph_compute(p.be, p.gf);
			const bool back = identical(read_outs(p), want, nullptr);
			++tried;
			moved += changed ? 1 : 0;
			restored += back ? 1 : 0;
			if (changed && moved_names.size() < 200) {
				moved_names += std::string(moved_names.empty() ? "" : ", ") + w->name;
			}
		}
		std::printf("PROBE pjrt backend %s control: a 4-byte write to %zu weights changed the result for %zu (must be > 0); writing the bytes back "
					"restored it for %zu of %zu; moved by: %s\n", which.c_str(), tried, moved, restored, tried, moved_names.c_str());
		const std::string end_stats = ggml_backend_pjrt_stats();
		std::printf("PROBE pjrt backend %s final stats: %s\n", which.c_str(), end_stats.c_str());
		ok = same && again && moved > 0 && restored == tried && tried > 0;
		std::printf("PROBE pjrt backend %s SUMMARY identical=%d second_compute_clean=%d changed=%zu/%zu restored=%zu/%zu alloc_ms=%.1f "
					"first_compute_ms=%.1f direct_ms=%.1f\n", which.c_str(), same ? 1 : 0, again ? 1 : 0, moved, tried, restored, tried,
				(t_alloc - t_direct) / 1000.0, (rdc::host_usec() - t_alloc) / 1000.0, (t_direct - t0) / 1000.0);
		ggml_backend_free(p.be);
		p.be = nullptr;
	}
	ggml_backend_pjrt_detach();
	pjrt_rd::attach(nullptr);
	ggml_backend_free(direct.be);
	direct.be = nullptr;
	return ok;
}


enum class StablehloDtype { F32, BF16 };

struct StablehloArg {
	StablehloDtype dtype;
	int count;
	const unsigned char *bytes;
	size_t size;
};

struct StablehloCase {
	const char *name;
	const char *hlo;
	const char *bad;
	const StablehloArg *args;
	int nargs;
	const unsigned char *out;
	size_t out_size;
	StablehloDtype out_dtype;
	const char *note;
	const char *xfail; // why the output is not bit-equal to XLA CPU's; nullptr when it must be
};

#include "stablehlo_cases.inc"

float element(const unsigned char *p, StablehloDtype d, size_t i) {
	if (d == StablehloDtype::F32) {
		float f;
		std::memcpy(&f, p + i * 4, 4);
		return f;
	}
	uint16_t h;
	std::memcpy(&h, p + i * 2, 2);
	const uint32_t w = uint32_t(h) << 16;
	float f;
	std::memcpy(&f, &w, 4);
	return f;
}

// The bits of a float mapped so that adjacent values are adjacent integers.
int64_t ordered(float f, StablehloDtype d) {
	uint32_t w;
	std::memcpy(&w, &f, 4);
	if (d == StablehloDtype::BF16) {
		w >>= 16;
		const int64_t sign = (w & 0x8000u) != 0 ? -1 : 1;
		return sign * int64_t(w & 0x7fffu);
	}
	const int64_t sign = (w & 0x80000000u) != 0 ? -1 : 1;
	return sign * int64_t(w & 0x7fffffffu);
}

struct Compared {
	size_t n = 0;
	size_t differing = 0;
	int64_t max_ulp = 0;
	double rel_l2 = 0.0;
};

Compared compare_to(const unsigned char *want, const std::vector<uint8_t> &got, size_t want_size, StablehloDtype d) {
	Compared c;
	if (got.size() != want_size) {
		c.n = want_size;
		c.differing = want_size;
		c.max_ulp = INT64_MAX;
		c.rel_l2 = 1.0;
		return c;
	}
	const size_t esz = d == StablehloDtype::F32 ? 4 : 2;
	c.n = want_size / esz;
	double num = 0.0;
	double den = 0.0;
	for (size_t i = 0; i < c.n; ++i) {
		const float w = element(want, d, i);
		const float g = element(got.data(), d, i);
		if (std::memcmp(want + i * esz, got.data() + i * esz, esz) != 0) {
			++c.differing;
			c.max_ulp = std::max<int64_t>(c.max_ulp, std::llabs(ordered(w, d) - ordered(g, d)));
		}
		num += (double(g) - double(w)) * (double(g) - double(w));
		den += double(w) * double(w);
	}
	c.rel_l2 = den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
	return c;
}

struct Refusal {
	const char *name;
	const char *body; // the statements of @main, after its parameter %arg0: tensor<4xf32>
	PJRT_Error_Code code;
};

// Valid StableHLO outside what the lowering handles, and malformed text, must each come back as a
// specific error from Compile and never reach ggml.
size_t probe_refusals(PJRT_Client *client, size_t *total) {
	static const Refusal kRefusals[] = {
		{ "unsupported_op", "%0 = stablehlo.sine %arg0 : tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "hex_constant", "%0 = stablehlo.constant dense<0x7FC00000> : tensor<f32>\n    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>\n    return %1 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "reduce_multiply", "%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.multiply across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>\n    return %1 : tensor<f32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "reduce_init_not_identity", "%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>\n    return %1 : tensor<f32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "reduce_bad_dimension", "%0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [3] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>\n    return %1 : tensor<f32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "maximum_of_values", "%0 = stablehlo.maximum %arg0, %arg0 : tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "pad_nonzero_value", "%0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n    %1 = stablehlo.pad %arg0, %0, low = [1], high = [0], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<5xf32>\n    return %1 : tensor<5xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "pad_interior", "%0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n    %1 = stablehlo.pad %arg0, %0, low = [0], high = [0], interior = [1] : (tensor<4xf32>, tensor<f32>) -> tensor<7xf32>\n    return %1 : tensor<7xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "slice_past_end", "%0 = stablehlo.slice %arg0 [2:6] : (tensor<4xf32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "concatenate_rank_mismatch", "%0 = stablehlo.concatenate %arg0, %arg0, dim = 3 : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>\n    return %0 : tensor<8xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "shape_mismatch", "%0 = stablehlo.add %arg0, %arg0 : tensor<5xf32>\n    return %0 : tensor<5xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "wrong_declared_type", "%0 = stablehlo.negate %arg0 : tensor<5xf32>\n    return %0 : tensor<5xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "undefined_operand", "%0 = stablehlo.add %arg0, %nope : tensor<4xf32>\n    return %0 : tensor<4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "no_return", "%0 = stablehlo.negate %arg0 : tensor<4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "returns_parameter", "return %arg0 : tensor<4xf32>", PJRT_Error_Code_UNIMPLEMENTED },
		{ "garbage_statement", "this is not a statement", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "bad_broadcast", "%0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<4x4xf32>\n    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<4xf32>) -> tensor<4x4xf32>\n    return %1 : tensor<4x4xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
		{ "reshape_count", "%0 = stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<5xf32>\n    return %0 : tensor<5xf32>", PJRT_Error_Code_INVALID_ARGUMENT },
	};
	static const char *kWholeModules[][2] = {
		{ "integer_type", "module @m {\n  func.func public @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>) {\n    %0 = stablehlo.negate %arg0 : tensor<4xi32>\n    return %0 : tensor<4xi32>\n  }\n}" },
		{ "rank_five", "module @m {\n  func.func public @main(%arg0: tensor<1x1x1x1x4xf32>) -> (tensor<1x1x1x1x4xf32>) {\n    %0 = stablehlo.negate %arg0 : tensor<1x1x1x1x4xf32>\n    return %0 : tensor<1x1x1x1x4xf32>\n  }\n}" },
		{ "dynamic_shape", "module @m {\n  func.func public @main(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {\n    %0 = stablehlo.negate %arg0 : tensor<?xf32>\n    return %0 : tensor<?xf32>\n  }\n}" },
		{ "two_functions", "module @m {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    %0 = stablehlo.negate %arg0 : tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n  func.func private @helper(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    return %arg0 : tensor<4xf32>\n  }\n}" },
	};
	size_t held = 0;
	std::vector<std::pair<std::string, std::pair<std::string, PJRT_Error_Code> > > all;
	for (const Refusal &r : kRefusals) {
		all.push_back({ r.name, { std::string("module @m {\n  func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {\n    ") + r.body + "\n  }\n}", r.code } });
	}
	for (const char *const *w : kWholeModules) {
		all.push_back({ w[0], { w[1], std::strstr(w[0], "two_functions") != nullptr || std::strstr(w[0], "dynamic") != nullptr || std::strstr(w[0], "integer") != nullptr || std::strstr(w[0], "rank") != nullptr ? PJRT_Error_Code_UNIMPLEMENTED : PJRT_Error_Code_INVALID_ARGUMENT } });
	}
	for (const std::pair<std::string, std::pair<std::string, PJRT_Error_Code> > &r : all) {
		PJRT_Program prog;
		std::memset(&prog, 0, sizeof prog);
		prog.struct_size = PJRT_Program_STRUCT_SIZE;
		prog.code = const_cast<char *>(r.second.first.data());
		prog.code_size = r.second.first.size();
		prog.format = "mlir";
		prog.format_size = 4;
		PJRT_ARGS(PJRT_Client_Compile_Args, cc);
		cc.client = client;
		cc.program = &prog;
		PJRT_Error_Code got = PJRT_Error_Code_OK;
		const std::string msg = take(g_api->PJRT_Client_Compile(&cc), &got);
		const bool ok = !msg.empty() && got == r.second.second;
		held += ok ? 1 : 0;
		std::printf("PROBE pjrt stablehlo refusal %-20s %s code=%d wanted=%d %.100s\n", r.first.c_str(), ok ? "held  " : "FAILED", int(got), int(r.second.second), msg.c_str());
	}
	*total = all.size();
	return held;
}

// Real StableHLO from JAX, compiled and executed through the C API, against the output XLA's CPU
// backend gave for the same module. A mutated module is the control: it must be refused or give
// another output.
bool probe_stablehlo(const std::string &which) {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		std::printf("PROBE pjrt stablehlo: no RD device\n");
		return false;
	}
	ggml_backend_t be = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
	pjrt_rd::attach(be);
	g_api = GetPjrtApi();
	PJRT_Client *client = make_client();
	bool ok = client != nullptr;
	size_t ran = 0, bit_equal = 0, expected_failures = 0, controls_held = 0;
	{
		std::vector<uint8_t> flipped(kStablehloCases[0].out, kStablehloCases[0].out + kStablehloCases[0].out_size);
		flipped[0] ^= 1;
		const Compared self = compare_to(kStablehloCases[0].out, flipped, flipped.size(), kStablehloCases[0].out_dtype);
		std::printf("PROBE pjrt stablehlo compare control: one flipped bit is seen as %zu differing element(s) (must be 1)\n", self.differing);
		ok = ok && self.differing == 1;
	}
	for (const StablehloCase &c : kStablehloCases) {
		if (client == nullptr || (which != "all" && which != c.name)) {
			continue;
		}
		std::vector<PJRT_Buffer *> bufs;
		for (int k = 0; k < c.nargs; ++k) {
			bufs.push_back(upload(client, c.args[k].bytes, c.args[k].size));
		}
		const Run good = execute(client, c.hlo, bufs, 1);
		Compared cmp;
		if (good.ok && good.out.size() == 1) {
			cmp = compare_to(c.out, good.out[0], c.out_size, c.out_dtype);
		} else {
			cmp.n = c.out_size;
			cmp.differing = c.out_size;
			cmp.max_ulp = INT64_MAX;
			cmp.rel_l2 = 1.0;
		}
		const Run bad = execute(client, c.bad, bufs, 1);
		const bool control = !bad.ok || bad.out.size() != 1 || compare_to(c.out, bad.out[0], c.out_size, c.out_dtype).differing != 0;
		const bool bits = good.ok && cmp.differing == 0;
		const double bar = c.out_dtype == StablehloDtype::F32 ? 1e-4 : 1e-3; // the oracle bar of the graph probes
		const bool functional = good.ok && cmp.rel_l2 <= bar;
		++ran;
		bit_equal += bits ? 1 : 0;
		expected_failures += c.xfail != nullptr ? 1 : 0;
		controls_held += control ? 1 : 0;
		std::printf("PROBE pjrt stablehlo %-22s ran=%d bit_equal=%d differing=%zu/%zu max_ulp=%lld rel_l2=%.3e control=%s%s%s\n", c.name,
				good.ok ? 1 : 0, bits ? 1 : 0, cmp.differing, cmp.n, (long long)cmp.max_ulp, cmp.rel_l2, control ? "held" : "FAILED",
				good.ok ? "" : " error=", good.ok ? "" : good.error.c_str());
		if (c.xfail != nullptr) {
			std::printf("PROBE pjrt stablehlo %-22s expected not bit-equal (%s); functional bar %.0e %s\n", c.name, c.xfail, bar,
					bits ? "NOT NEEDED: it is bit-equal now, so remove it from XFAIL" : (functional ? "met" : "NOT MET"));
			ok = ok && !bits && functional && control;
		} else {
			ok = ok && bits && control;
		}
		for (PJRT_Buffer *b : bufs) {
			drop(b);
		}
	}
	size_t refusals = 0, refusals_held = 0;
	if (client != nullptr && which == "all") {
		refusals_held = probe_refusals(client, &refusals);
		ok = ok && refusals_held == refusals;
	}
	std::printf("PROBE pjrt stablehlo refusals held=%zu/%zu\n", refusals_held, refusals);
	std::printf("PROBE pjrt stablehlo SUMMARY cases=%zu bit_equal=%zu expected_not_bit_equal=%zu controls_held=%zu\n", ran, bit_equal, expected_failures, controls_held);
	if (client != nullptr) {
		destroy_client(client);
	}
	pjrt_rd::attach(nullptr);
	ggml_backend_free(be);
	return ok && ran > 0;
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
	if (arg.compare(0, 10, "stablehlo:") == 0) {
		return probe_stablehlo(arg.substr(10));
	}
	if (arg.compare(0, 5, "cast:") == 0) {
		return probe_cast(arg.substr(5));
	}
	if (arg.compare(0, 7, "bisect:") == 0) {
		return probe_bisect(arg.substr(7));
	}
	if (arg.compare(0, 8, "backend:") == 0) {
		return probe_backend(arg.substr(8));
	}
	return probe_graph_through_pjrt(arg);
}

} // namespace probes
