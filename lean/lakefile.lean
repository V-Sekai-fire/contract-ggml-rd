import Lake
open Lake DSL

package Ggml where

-- Every dependency is pinned in a V-Sekai-fire repo or fork.
require LeanSlang from git
  "https://github.com/V-Sekai-fire/contract-lean-slang.git" @ "60532aef8ed70cc669ecab481182d0636c9e1ac3"

-- The ggml-rd op kernels (Cut 3): Ggml.SlangCodegen.*, one family per
-- module, all on the fixed layout of Ggml.SlangCodegen.Common.
lean_lib Ggml

lean_exe emit_ggml where
  root := `EmitGgml
