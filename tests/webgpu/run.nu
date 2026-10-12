# ggml-rd kernels: the host L2 (cpp emits vs ggml-cpu), then the WGSL emits on WebGPU vs ggml-cpu.
#   pixi run nu tests/webgpu/run.nu <build dir with ggml_rd_l2> <scratch dir>
def main [build: path, scratch: path] {
  const here = path self .
  let root = ($here | path join .. .. | path expand)
  let l2 = ($build | path join ggml_rd_l2)
  let cases = ($scratch | path join cases)
  let wgsl = ($scratch | path join wgsl)
  rm -rf $cases $wgsl
  mkdir $cases $wgsl

  let host = (do { ^$l2 } | complete)
  print ($host.stdout | lines | last 2 | str join "\n")

  let exp = (do { ^$l2 $"--export=($cases)" } | complete)
  print ($exp.stdout | lines | where $it =~ '^FAIL' | first 10 | str join "\n")

  let slang = ($root | path join kernels ggml slang)
  for f in (ls $slang | where name =~ '\.slang$' | get name) {
    let k = ($f | path parse | get stem)
    ^slangc $f -target wgsl -entry main -stage compute -o ($wgsl | path join $"($k).wgsl")
  }
  let r = (do { ^deno run --unstable-webgpu -A ($here | path join run.mjs) $cases $wgsl ($root | path join kernels ggml kernels.txt) } | complete)
  $r.stdout | save -f ($scratch | path join webgpu.log)
  print ($r.stdout | lines | where $it =~ '^(FAIL|summary|kernels|RESULT|ggml-rd)' | str join "\n")
  print ($r.stderr | lines | last 5 | str join "\n")
}
