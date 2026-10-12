// The web test's references, computed natively (ggml-cpu, every core): each
// G3.graph net's outputs per arm as raw little-endian f32, so the browser
// runs only WebGPU and compares against these.
//
//   ggml_rd_web_refs <out dir> [graphs...]   -> <out>/<graph>/<arm>/<output>.f32 + <out>/refs.json
//
// arm is "native" or "f32". The last line is "RESULT: PASS" or "RESULT: FAIL".
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "graph_nets.h"

using namespace graph_nets;

int main(int argc, char **argv) {
	if (argc < 2) {
		std::printf("usage: ggml_rd_web_refs <out dir> [graphs...]\nRESULT: FAIL\n");
		return 1;
	}
	const std::filesystem::path out = argv[1];
	std::vector<std::string> graphs(argv + 2, argv + argc);
	if (graphs.empty()) {
		graphs = { "qwen", "sconv", "kimodo_denoiser", "dit" };
	}
	std::string manifest = "[";
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
			const std::string arm = f32_arm ? "f32" : "native";
			Net n;
			n.be = ggml_backend_cpu_init();
			ggml_backend_cpu_set_n_threads(n.be, int(std::thread::hardware_concurrency()));
			Outs o;
			try {
				build(n, f32_arm);
				if (!allocate(n) || ggml_backend_graph_compute(n.be, n.gf) != GGML_STATUS_SUCCESS) {
					throw std::runtime_error("compute");
				}
				o = read_outs(n);
			} catch (const std::exception &e) {
				std::printf("%s %s: %s\n", g.c_str(), arm.c_str(), e.what());
				pass = false;
			}
			ggml_backend_free(n.be);
			n.be = nullptr;
			const std::filesystem::path dir = out / g / arm;
			std::filesystem::create_directories(dir);
			for (const auto &x : o) {
				std::ofstream f(dir / (x.first + ".f32"), std::ios::binary);
				f.write(reinterpret_cast<const char *>(x.second.data()), std::streamsize(x.second.size() * sizeof(float)));
				manifest += std::string(manifest.size() > 1 ? "," : "") + "{\"graph\":\"" + g + "\",\"arm\":\"" + arm +
						"\",\"native\":\"" + native + "\",\"residual_out\":\"" + resid + "\",\"output\":\"" + x.first +
						"\",\"n\":" + std::to_string(x.second.size()) + "}";
			}
			std::printf("%s %s: %zu outputs\n", g.c_str(), arm.c_str(), o.size());
		}
	}
	std::ofstream(out / "refs.json") << manifest << "]\n";
	std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}
