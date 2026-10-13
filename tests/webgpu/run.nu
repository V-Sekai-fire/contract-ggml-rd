# ggml-rd kernels: the host L2 (cpp emits vs ggml-cpu), then the WGSL emits on WebGPU vs ggml-cpu.
#   pixi run nu tests/webgpu/run.nu <build dir with ggml_rd_l2> <scratch dir>
#
# Test shapes:
#   unit            every case's WGSL emit on WebGPU matches ggml-cpu (run.mjs)
#   reverse control every case again with its last dispatch left out must FAIL (run.mjs);
#                   the host L2 swaps src strides (ggml_rd_l2 --control=swap-nb)
#   contrapositive  a failure is never reported as success: a broken .slang makes slangc
#                   exit nonzero with no .wgsl written, a broken .wgsl makes its cases print
#                   an error and run.mjs exit nonzero, and this script exits nonzero on any FAIL
def main [build: path, scratch: path] {
  const here = path self .
  let root = ($here | path join .. .. | path expand)
  let l2 = ($build | path join ggml_rd_l2)
  let cases = ($scratch | path join cases)
  let wgsl = ($scratch | path join wgsl)
  let kernels_txt = ($root | path join kernels ggml kernels.txt)
  rm -rf $cases $wgsl
  mkdir $cases $wgsl
  mut failed = []

  let host = (do { ^$l2 } | complete)
  print ($host.stdout | lines | last 2 | str join "\n")
  if $host.exit_code != 0 { $failed = ($failed | append "host L2") }

  let exp = (do { ^$l2 $"--export=($cases)" } | complete)
  print ($exp.stdout | lines | where $it =~ '^FAIL' | first 10 | str join "\n")
  if $exp.exit_code != 0 { $failed = ($failed | append "case export") }

  let slang = ($root | path join kernels ggml slang)
  let emits = (ls $slang | where name =~ '\.slang$' | get name | each {|f|
    let k = ($f | path parse | get stem)
    let out = ($wgsl | path join $"($k).wgsl")
    let r = (do { ^slangc $f -target wgsl -entry main -stage compute -o $out } | complete)
    {kernel: $k, ok: ($r.exit_code == 0 and ($out | path exists)), error: ($r.stderr | lines | where $it =~ 'error' | first 1 | str join '')}
  })
  let bad_emits = ($emits | where not ok)
  print $"slang to wgsl: ($emits | length) kernels, failed ($bad_emits | length)"
  if ($bad_emits | is-not-empty) {
    print ($bad_emits | table)
    $failed = ($failed | append "slang to wgsl")
  }

  # unit and reverse control
  let r = (do { ^deno run --unstable-webgpu -A ($here | path join run.mjs) $cases $wgsl $kernels_txt } | complete)
  $r.stdout | save -f ($scratch | path join webgpu.log)
  print ($r.stdout | lines | where $it =~ '^(FAIL|summary|kernels|RESULT|ggml-rd)' | str join "\n")
  print ($r.stderr | lines | last 5 | str join "\n")
  let unit_pass = ($r.exit_code == 0 and ($r.stdout | lines | last 1 | str join '') == 'RESULT: PASS')
  if not $unit_pass { $failed = ($failed | append "webgpu unit") }

  # contrapositive 1: a broken .slang must fail loudly and write nothing
  let bad_slang = ($scratch | path join planted_broken.slang)
  let bad_slang_out = ($scratch | path join planted_broken.wgsl)
  rm -f $bad_slang_out
  "[numthreads(1, 1, 1)]\nvoid main() { undeclared_thing = 1; }\n" | save -f $bad_slang
  let s = (do { ^slangc $bad_slang -target wgsl -entry main -stage compute -o $bad_slang_out } | complete)
  let slang_contra = ($s.exit_code != 0 and not ($bad_slang_out | path exists) and ($s.stderr =~ 'error'))
  print $"contrapositive \(broken slang rejected, no wgsl written\): (if $slang_contra { 'PASS' } else { 'FAIL' })"
  if not $slang_contra { $failed = ($failed | append "contrapositive slang") }

  # contrapositive 2: one kernel's WGSL broken -> its cases print an error and the run exits nonzero
  let victim = (open --raw $kernels_txt | lines | where $it != '' and not ($it starts-with '#') | first)
  let wgsl_bad = ($scratch | path join wgsl-planted)
  rm -rf $wgsl_bad
  cp -r $wgsl $wgsl_bad
  "this is not wgsl" | save -f ($wgsl_bad | path join $"($victim).wgsl")
  let b = (do { ^deno run --unstable-webgpu -A ($here | path join run.mjs) $cases $wgsl_bad $kernels_txt } | complete)
  let err_lines = ($b.stdout | lines | where {|l| ($l starts-with 'FAIL') and ($l =~ $"error: ($victim)") } | length)
  let wgsl_contra = ($b.exit_code != 0 and $err_lines > 0 and ($b.stdout | lines | last 1 | str join '') == 'RESULT: FAIL')
  print $"contrapositive \(broken wgsl for ($victim): ($err_lines) cases report an error, run fails\): (if $wgsl_contra { 'PASS' } else { 'FAIL' })"
  if not $wgsl_contra { $failed = ($failed | append "contrapositive wgsl") }

  if ($failed | is-empty) {
    print "ALL: PASS"
  } else {
    print $"ALL: FAIL \(($failed | str join ', ')\)"
    exit 1
  }
}
