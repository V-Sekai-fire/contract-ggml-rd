# Workgroup memory per WGSL kernel against WebGPU's default limit (16384 bytes).
#   pixi run nu web/wg_census.nu <dir of .wgsl>
def wg_bytes [line: string] {
  let ty = ($line | parse --regex 'array<(?<ty>f32|u32|i32|f16)' | get ty | first)
  let dims = ($line | parse --regex 'i32\((?<n>\d+)\)' | get n | each {|n| $n | into int })
  let unit = (if $ty == "f16" { 2 } else { 4 })
  $dims | reduce --fold $unit {|n, acc| $acc * $n }
}

def main [dir: path] {
  let rows = (ls $dir | where name =~ '\.wgsl$' | each {|f|
    let lines = (open --raw $f.name | lines | where $it =~ 'var<workgroup>')
    let bytes = (if ($lines | is-empty) { 0 } else { $lines | each {|l| wg_bytes $l } | math sum })
    { kernel: ($f.name | path parse | get stem), workgroup_bytes: $bytes }
  })
  let over = ($rows | where workgroup_bytes > 16384 | sort-by workgroup_bytes -r)
  print $"kernels ($rows | length), using workgroup memory ($rows | where workgroup_bytes > 0 | length), over 16384: ($over | length)"
  print ($over | table)
  print ($rows | where workgroup_bytes > 0 | where workgroup_bytes <= 16384 | sort-by workgroup_bytes -r | first 6 | table)
}
