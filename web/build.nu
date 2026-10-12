# Build ggml-rd's web module with emscripten and run its graph test on WebGPU (Deno).
#   pixi run nu web/build.nu <build dir> [graphs...]
def main [build: path, ...graphs: string] {
  const here = path self .
  if not ($build | path join build.ninja | path exists) {
    ^emcmake cmake -S $here -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release
  }
  ^cmake --build $build --target ggml_rd_web_test
  cd $build
  ^deno run --unstable-webgpu -A ($here | path join test run_test.mjs) ./ggml_rd_web_test.mjs ...$graphs
}
