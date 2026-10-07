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
