# Build ggml-rd's web module with emscripten, compute the test's references
# natively (web/refs, every core), and run the test on WebGPU (Deno). The
# browser page is the same build dir served over HTTP: index.html.
#   pixi run nu web/build.nu <build dir> [graphs...]
def main [build: path, ...graphs: string] {
  const here = path self .
  if not ($build | path join build.ninja | path exists) {
    ^emcmake cmake -S $here -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release
  }
  ^cmake --build $build --target ggml_rd_web_test
  let rb = ($build | path join refs-build)
  if not ($rb | path join build.ninja | path exists) {
    ^cmake -S ($here | path join refs) -B $rb -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  }
  ^cmake --build $rb
  ^($rb | path join ggml_rd_web_refs) ($build | path join refs) ...$graphs
  for f in [index.html worker.mjs refs.mjs] {
    cp ($here | path join test $f) $build
  }
  cd $build
  ^deno run --unstable-webgpu -A ($here | path join test run_test.mjs) ./ggml_rd_web_test.mjs ./refs ...$graphs
}
