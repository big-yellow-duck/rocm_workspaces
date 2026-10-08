# Vector add: naive and winner

Two kernels for `C = A + B`, using contiguous 1024×1024 FP32 tensors on gfx1152.

| File | Kernel in the profiler | Implementation |
| --- | --- | --- |
| [naive.py](naive.py) | `naive_kernel_0` | Original FlyDSL example: masked 8×64 tiles, 128 threads/block, 128-bit copies |
| [winner.py](winner.py) | `winner_kernel_0` | 64 threads/block, 8192 blocks, 64-bit loads/stores, scalar bases and `slc` |

Run these commands from `10/gfx1152`:

```bash
pixi run vectoradd                 # Check both kernels against Torch
pixi run profile-vectoradd         # Capture both warmed kernels
pixi run view-vectoradd naive      # Open the naive trace
pixi run view-vectoradd winner     # Open the winner trace
```

`run.py` checks exact results and dependent launches on a non-default stream.
`profile.py` warms each kernel 50 times, checks its output, then captures one
dispatch with rocprofv3. It saves decoded ATT traces, kernel timing CSVs,
instruction statistics, and matching source snapshots under `profiles/`.
The viewer automatically opens the latest capture. It uses the existing
`~/rocprof-viewer` installation.

ATT targets WGP 1 / SIMD 3. Profiling temporarily uses `profile_standard`
power mode and restores the original mode afterward. Profiles are for
examining execution; profiler timings use different conditions from the
earlier paired GPU-graph benchmark.

The naive kernel retains Apache-2.0 attribution from
[FlyDSL v0.3.4.1's vector-add example](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/01-vectorAdd.py).
