# Neural Engine evaluation

## Decision

Do **not** select the Apple Neural Engine for the current GTCRN Core AI assets.
The graph does not successfully specialize and load on ANE, so it cannot meet
the project's gate of equal-or-better latency with stable, full-model residency.
Use explicit CPU specialization as the conservative default; GPU specialization
is also numerically validated.

This is an implementation result, not a claim that GTCRN can never run on ANE.
An ANE version would require re-authoring and revalidating the recurrent graph,
not merely changing a compute-unit preference.

## Gate

ANE is accepted only if all of the following are true:

1. The static float16 asset specializes and loads without warnings or a process
   abort.
2. The complete recurrent graph stays on ANE. Partial placement is insufficient
   for this 48.2K-parameter model because CPU/ANE transfers can dominate its
   already tiny compute cost.
3. Consecutive-frame output parity remains at least 40 dB PSNR against PyTorch.
4. Warm steady-state p50 and p95 inference latency are no worse than the fastest
   already-valid CPU/GPU specialization on the target device.

## Environment and method

- Apple M4 Max, 64 GB
- macOS 27.0 (26A5406e)
- Python 3.12.13
- `coreai-core==1.0.0b2`
- `coreai-torch==0.4.1`
- DNS3 float16 streaming asset
- 64 warm-up frames followed by 512 timed consecutive frames
- recurrent outputs fed directly into the next call
- 16 ms audio frame period
- timing covers the Core AI function call and state outputs, but excludes model
  load, STFT, iSTFT, and audio I/O

The benchmark is reproducible with:

```sh
uv run gtcrn-coreai-benchmark \
  exports/gtcrn_dns3_float16_streaming.aimodel \
  --compute-unit cpu
```

Run accelerator experiments in separate processes. An explicit ANE attempt is:

```sh
uv run gtcrn-coreai-benchmark \
  exports/gtcrn_dns3_float16_streaming.aimodel \
  --compute-unit neural-engine
```

Be aware that the tested beta Core AI/macOS runtime aborts the Python process on
this failure instead of returning a catchable Python exception.

## Results

The successful CPU baseline measured:

| Compute unit | Mean | p50 | p95 | p99 | Maximum | Mean / 16 ms budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 0.232 ms | 0.231 ms | 0.240 ms | 0.248 ms | 0.499 ms | 1.45% |
| ANE | No result | No result | No result | No result | No result | Load failed |

The CPU result is a representative local run, not a cross-device guarantee.
Existing parity measurements separately show that both CPU and GPU produce more
than 56 dB minimum PSNR across the bundled checkpoints.

ANE/default specialization attempted to compile decomposed PReLU and GRU
regions, emitted unsupported-pattern diagnostics, then failed to load the ANE
program with Core AI error Code 53 (`Program load failed - no memory`). The
runtime subsequently aborted. Therefore an ANE performance comparison is not
defined: no valid ANE inference completed.

Enabling the OS runtime's private debug-residency metadata also proved unstable
on this beta stack. It exposed GPU residency for sampled operations, but a later
debug specialization trapped in `CoreAIRuntime` and left subsequent model-load
attempts failing until the OS service is reset. The repository benchmark
intentionally does not enable that debug mode.

## What would be required for ANE

The main blockers are the decomposed recurrent and activation patterns. A real
ANE port would likely replace the exported GRU/PReLU regions with an
ANE-oriented, static-shape formulation (typically convolution/matrix primitives
and explicit recurrence), then remap the original weights and repeat full-stream
parity and audio-quality validation.

That rewrite is not selected here because:

- changing only the compute-unit preference does not produce a loadable model;
- partial placement would add transfer overhead to a CPU path already far below
  the real-time budget; and
- a recurrent rewrite is a new model-authoring effort with meaningful numerical
  and audio-quality risk, not a safe packaging optimization.

Revisit this decision when the Core AI runtime/compiler changes or when a fully
ANE-oriented recurrent implementation can be tested on every deployment device.
