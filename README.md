# ROCm development workspaces

GPU-specific Pixi environments for building and running vLLM with AMD ROCm.
The ROCm 10 workspaces use Python 3.12, ROCm 10.0.0, and AMD PyTorch 2.13.0.
Pixi manages the ROCm/PyTorch stack and development tools; uv installs vLLM
in editable mode and resolves its runtime dependencies from the checkout.

| GPU target | Workspace |
| --- | --- |
| gfx1100 | [Setup and usage](10/gfx1100/README.md) |
| gfx1151 — AMD Radeon 8060S | [Setup, usage, and verification](10/gfx1151/README.md) |
| gfx1152 — AMD Radeon 860M | [Setup and GPU validation](10/gfx1152/README.md) |
| gfx1201 — Radeon AI PRO R9700 | [Setup, usage, and verification](10/gfx1201/README.md) |

## gfx1151 quick start

Run from the repository root:

```bash
cd 10/gfx1151
# Clone once; skip this command if vllm/ already exists.
git clone --single-branch --branch perf/rocm-segmented-attn \
  https://github.com/big-yellow-duck/vllm.git vllm
pixi install --locked
ln -sfn .pixi/envs/default .venv
pixi run check-torch
pixi run install-vllm
pixi run smoke
pixi run smoke-segmented
pixi run test-segmented
```

Use `pixi run` or `pixi shell` to activate the ROCm SDK and compiler paths.
After updating vLLM or running `pixi install`, run `pixi run install-vllm`
to reconcile its dependencies. The installer preserves the Pixi-managed stack
while allowing vLLM's runtime dependencies to follow its requirements.
Preview resolution with `pixi run install-vllm --dry-run`.

The gfx1151 checkout uses `big-yellow-duck/vllm`'s `perf/rocm-segmented-attn`
branch. On 2026-10-02, its native build, dependency compatibility check,
GPU tensor check, and Qwen2.5-0.5B-Instruct generation with both default and
segmented attention passed. Focused tests reported 69 passed and 31 skipped.
Generation was verified with eager execution and FP16 KV; graph capture,
startup autotuning, speculative decoding, and performance were not evaluated.
See the workspace README for the tested revision and local log locations.

The older shared environment directly under `10/` is retained as legacy
configuration. Use the GPU-specific workspaces for current setup instructions.
