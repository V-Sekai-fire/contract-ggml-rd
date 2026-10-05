# contract-ggml-rd

The ggml backend on the engine's rendering device for sandbox guests: Lean-emitted kernels, a CPU fallback, and the oracles that check them.

## What it is for

It is the route by which ggml reaches the GPU inside sandbox guests, so no guest carries a tensor runtime of its own. RFD 2290 owns that decision.

## Build

```sh
kernels/ggml/gen.sh
```

It emits the kernels from Lean, checks them against the committed `slang/` and compiles them for both targets; the header of `gen.sh` names its modes. `transport-meshing-pen`'s build compiles the guest programs, finding this repository and its siblings at their goal-manifest paths.

## Licence

The repository does not state a licence of its own; `lean/CITATION.cff` names MIT for the Lean tree.
