# contract-ggml-rd

ggml's backend on Godot's RenderingDevice for godot-sandbox guests: Lean-emitted kernels, a CPU fallback, L2 and graph oracles.

Split out of `interactor-dress-on` at `310b52e` with its history (`git subtree`). It sits at `2-contract/ggml-rd` in the goal manifest (`contract-manifest-taskweft`), and finds the repositories it builds against as sibling checkouts at their manifest paths. `transport-meshing-pen` builds the guest ELFs (`build.sh`, `tools/build.exs`).
