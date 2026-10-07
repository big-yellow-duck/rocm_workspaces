# ROCm 10 / gfx1152 PyTorch workspace

Native Pixi environment for AMD Radeon 860M (`gfx1152`), matching the
other GPU workspaces: Python 3.12, ROCm 10.0.0, AMD PyTorch 2.13.0,
GCC 14, and development tools. GPU kernel packages target gfx1152.

From the repository root:

```bash
cd 10/gfx1152
pixi install --locked
ln -sfn .pixi/envs/default .venv
pixi run check-smi
pixi run check-torch
pixi run smoke
```

Use `pixi run` or `pixi shell` to activate SDK, compiler, library, and
SDK-provided AMD SMI paths. The optional `.venv` link supports editors;
tasks use the activated Python directly. The host must provide an AMD GPU
driver and access to `/dev/kfd` and the GPU render node. No architecture
override is required. The SDK is installed inside the environment.

`smoke` checks the actual gfx1152 architecture, FP32 matrix multiplication
against a CPU reference, and autograd gradients. It then runs 2048×2048
GPU matrix multiplication for approximately 15 seconds while sampling
AMD SMI usage, power, and temperature. Unsupported telemetry fields may
appear as N/A; command failures fail the smoke check.

This workspace sets up PyTorch and build tools. It does not install vLLM.
Local environments, editor links, and logs are ignored by Git.

## Verified on 2026-10-07

On Fedora/Bazzite with Linux 7.2.7 and AMD Radeon 860M:

- `pixi install --locked`: passed.
- Python 3.12.15, ROCm package 10.0.0, Torch 2.13.0+rocm10.0.0.
- Torch reports HIP 7.15.26333; this is the wheel's embedded HIP build
  version. SDK AMD SMI reports ROCm 10.0.0.
- `check-smi`: native gfx1152, eight compute units.
- `check-torch`: GPU tensor operation passed.
- `smoke`: CPU/GPU FP32 comparison, autograd, finite results, and AMD SMI
  monitoring passed; 434 GEMMs in 15.011 seconds.
- AMD SMI sampled GPU activity up to 100%, GPU temperature approximately
  64–69°C under load, and socket power approximately 25–33 W.
- `uv pip check`: all 43 installed packages compatible.

The workload is a functional smoke check, not a controlled performance
benchmark. No vLLM or model inference validation was performed.

## FlyDSL vector addition

Install the pinned published wheel into the same environment using uv:

```bash
pixi run install-flydsl
pixi run vectoradd
```

FlyDSL 0.3.4.1 declares no Python dependencies; the installer uses `--no-deps`
to preserve the Pixi-managed ROCm/PyTorch stack. Re-run `install-flydsl` after
`pixi install`, which may remove packages installed separately with uv.

The vector-add kernel is adapted from
[the upstream v0.3.4.1 example](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/01-vectorAdd.py),
retaining its Apache-2.0 attribution. The runner checks native gfx1152 and
compares GPU results with PyTorch for 1×4, 8×64, and 100×1000 FP32 inputs,
including partial tiles. It synchronizes the input and kernel streams and
raises on incorrect results.

Validated on 2026-10-07: uv installed FlyDSL 0.3.4.1; the upstream
100×1000 example and all three workspace vector-add cases passed.

## Tuned 1024×1024 vector addition

```bash
pixi run vectoradd-optimized
pixi run bench-vectoradd --require-speedup
# Sweep block size, vector width, grid geometry, and cache policy.
pixi run tune-vectoradd --layout asm
# Recheck the best ten candidates from a saved sweep, with longer samples.
pixi run tune-vectoradd --finalists logs/vectoradd_tuning.json --rounds 24 --repeats 30
```

The selected gfx1152 preset uses **64 threads per block, grid (8192, 1, 1),
and two FP32 values per thread (64-bit vector loads/stores)**. Both reads and
writes use the GFX11 `slc` cache modifier. The FlyDSL kernel uses scalar base
addresses with unsigned 32-bit byte offsets, avoiding per-thread 64-bit
address construction. It waits for both input reads before adding the values;
early-clobber constraints protect the address register during those reads.
The ordinary tiled example remains available as `pixi run vectoradd`.

Use the checked helper from the workspace's `scripts/` directory:

```python
import torch
from flydsl_vectoradd import add_1024

a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
out = torch.empty_like(a)
add_1024(a, b, out)
```

This forward-only helper requires contiguous, 16-byte-aligned 1024×1024 FP32
tensors on the same gfx1152 GPU. It honors the current PyTorch stream, or an
explicit `stream=` argument. `vectoradd-optimized` checks exact results and
eight dependent kernel launches on a non-default stream.

### Performance verification on 2026-10-07

The search covered blocks of 32–1024 threads; scalar, 64-bit, and 128-bit
native loads; wider logical vectors; contiguous and striped layouts;
smaller grids processing several tiles per block; and cache modifiers.
The selected scalar-address kernel passed exact comparison with PyTorch.

After the sweep and a separate 16-round finalist check, an independent
24-round comparison of three finalists selected this preset:

| Block threads | Grid blocks | Vector FP32 values | Load/store flags | Median paired speedup |
| --- | --- | --- | --- | --- |
| **64** | **8192** | **2** | **slc / slc** | **1.01777×** |
| 128 | 4096 | 2 | glc slc / slc | 1.01763× |
| 256 | 2048 | 2 | slc dlc / slc | 1.01534× |

The selected preset won 23 of 24 paired rounds. Its median GPU time was
126.520 µs versus 128.616 µs for `torch.add(a, b, out=c)`; the median of
paired ratios gives a **1.8% improvement**. A resampling interval for the
paired median ratio was 1.01566–1.02024 (95%, 20,000 resamples, seed 42).
The leading settings are close; this is the best measured preset in this
search, not a guarantee of identical rankings on another system.
Two further independent 15-round checks of the selected preset won all
15 paired rounds each, with median paired speedups of 1.00930× and 1.01519×.

All kernels compile and pass exact-result checks before timing. Each sample
uses HIP events around 30 replays of a 100-node GPU graph (3000 additions),
with warmup on the same stream. Candidate and baseline order are randomized,
and both use the same inputs and preallocated output. These are GPU execution
timings; Python dispatch, first-use JIT compilation, and allocation are excluded.
Early short-sample cache-hint gains did not reproduce in longer checks and
were discarded. A rotating-buffer experiment was too variable to establish
a reliable speedup. Local logs and raw measurements remain under `logs/`.
