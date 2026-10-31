import Lake
open Lake DSL

package rsqrtWitness

require «plausible-witness-dag» from git
  "https://github.com/V-Sekai-fire/plausible-witness-dag" @ "b150a7ff29e51b65074bbe71b63d6954eb80caa5"

lean_lib RsqrtTable where
  roots := #[`RsqrtTable]

@[default_target]
lean_exe rsqrt_witness where
  root := `Main
