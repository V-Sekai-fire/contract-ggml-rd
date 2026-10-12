// web/test: the G3.graph nets (guest/ggml_test/graph_nets.cpp) on ggml-rd's
// WebGPU device against ggml-cpu, both in this one wasm module.
//
//   rd native vs cpu native   rel-L2 <= 1e-3   (tests/ggml_graph_oracle's bounds)
//   rd f32    vs cpu f32      rel-L2 <= 1e-4
//   control: GGML_RD_FAULT=1 (a source of every dispatch moved) must fail
//
// web_main takes graph names separated by spaces (default: qwen sconv
// kimodo_denoiser dit); call it with ccall(..., {async: true}), since the GPU
// waits suspend it. The last line is "RESULT: PASS" or "RESULT: FAIL".
// kimodo_text is not in the default set: its f32 arm holds a 224 MiB tensor,
// over the 128 MiB storage binding every WebGPU device offers.
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <emscripten.h>

#include "ggml-cpu.h"
#include "ggml-rd.h"
#include "graph_nets.h"
#include "rd_compute.h"

using namespace graph_nets;

namespace {

rdc::Device g_dev;

ggml_backend_t rd_backend() {
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	return ggml_backend_reg_dev_count(reg) ? ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr) : nullptr;
}

bool run_net(const BuildFn &build, bool rd, bool f32_arm, Outs &out, std::string *why) {
	Net n;
	n.be = rd ? rd_backend() : ggml_backend_cpu_init();
	n.rd = rd;
	if (n.be != nullptr && !rd) {
		ggml_backend_cpu_set_n_threads(n.be, 1); // no pthreads: runs without cross-origin isolation
	}
	if (n.be == nullptr) {
		*why = "no backend";
		return false;
	}
	bool ok = true;
	try {
		build(n, f32_arm);
		if (!allocate(n)) {
			throw std::runtime_error("allocation");
		}
		const ggml_status st = ggml_backend_graph_compute(n.be, n.gf);
		if (st != GGML_STATUS_SUCCESS) {
			throw std::runtime_error("graph_compute status " + std::to_string(int(st)));
		}
		out = read_outs(n);
	} catch (const std::exception &e) {
		*why = e.what();
		ok = false;
	}
	ggml_backend_free(n.be);
	n.be = nullptr;
	return ok;
}

} // namespace

bool all_of_finite(const Outs &o) {
	for (const auto &x : o) {
		if (!finite(x.second)) {
			return false;
		}
	}
	return !o.empty();
}

extern "C" EMSCRIPTEN_KEEPALIVE int web_main(const char *names) {
	std::vector<std::string> graphs;
	std::istringstream in(names ? names : "");
	for (std::string g; in >> g;) {
		graphs.push_back(g);
	}
	if (graphs.empty()) {
		graphs = { "qwen", "sconv", "kimodo_denoiser", "dit" };
	}
	unsetenv("GGML_RD_CPU_FALLBACK");
	if (!g_dev.open()) {
		std::printf("no WebGPU device: %s\nRESULT: FAIL\n", g_dev.error().c_str());
		return 1;
	}
	ggml_backend_rd_attach(&g_dev, 0);
	std::printf("ggml-rd web: %s, max buffer %s MiB\n", g_dev.device_name().c_str(), std::getenv("GGML_RD_MAX_BUFFER_MB"));

	bool pass = true;
	for (const std::string &g : graphs) {
		BuildFn build;
		std::string native, resid;
		if (!graph_builder(g, 0, build, native, resid)) {
			std::printf("unknown graph %s\n", g.c_str());
			pass = false;
			continue;
		}
		for (const bool f32_arm : { false, true }) {
			const char *arm = f32_arm ? "f32" : native.c_str();
			const double bound = f32_arm ? 1e-4 : 1e-3;
			Outs ref, rd, faulted;
			std::string why;
			const double t0 = double(rdc::host_usec());
			bool ok = run_net(build, false, f32_arm, ref, &why);
			const double t1 = double(rdc::host_usec());
			ok = ok && run_net(build, true, f32_arm, rd, &why);
			const double t2 = double(rdc::host_usec());
			double worst = -1.0, worst_fault = -1.0;
			if (ok) {
				worst = compare(g.c_str(), arm, rd, ref, resid, false);
				setenv("GGML_RD_FAULT", "1", 1);
				std::string fwhy;
				const bool fok = run_net(build, true, f32_arm, faulted, &fwhy);
				unsetenv("GGML_RD_FAULT");
				worst_fault = fok ? compare(g.c_str(), "fault", faulted, ref, resid, false) : -1.0;
			}
			const bool good = ok && worst >= 0.0 && worst <= bound && all_of_finite(rd);
			const bool caught = !(worst_fault >= 0.0 && worst_fault <= bound);
			std::printf("%s %-16s %-5s rel_l2=%.3e (max %.0e) cpu_ms=%.0f webgpu_ms=%.0f control(fault)=%s rel_l2=%.3e%s%s\n",
					good ? "OK  " : "FAIL", g.c_str(), arm, worst, bound, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0,
					caught ? "caught" : "MISSED", worst_fault, why.empty() ? "" : " error: ", why.c_str());
			pass = pass && good && caught;
		}
	}
	std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}
