# ROCm 10 / gfx1151 vLLM workspace

Native development environment for the AMD Radeon 8060S (`gfx1151`).
Pixi manages Python 3.12, ROCm 10.0.0, AMD PyTorch 2.13.0, GCC 14,
and build and test tools. vLLM resolves its own runtime dependencies with uv;
these requirements are not duplicated in the Pixi manifest or lockfile.

```bash
cd /home/akk/jeff/rocm_workspaces/10/gfx1151
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

Skip cloning when `vllm/` already exists. `.venv` links to the Pixi environment.
Run commands through `pixi run` or `pixi shell` to activate the ROCm SDK,
SDK-provided AMD SMI binding, compiler, and library paths. Native builds target
only gfx1151; `MAX_JOBS` defaults to 4.

`install-vllm` installs the checkout in editable mode with dependency resolution
and `--no-build-isolation`. Constraints preserve installed Pixi-managed Python
packages, including ROCm/PyTorch and build tools. Runtime dependencies follow
the branch's requirements. The installer uses PyPI and AMD's wheel index with
`unsafe-best-match`, matching Pixi. It applies the gfx1201 workspace's ROCm 10
rocPRIM header workaround inside the SDK environment.

The branch's runtime ROCm requirements do not pin PyTorch, so the installer does
not run `use_existing_torch.py` or rewrite checkout dependency files. Its separate
build requirements contain an older Torch stack; build isolation stays disabled
so the existing Pixi stack is used instead. AMD SMI comes from the ROCm SDK;
there is no separate vendored binding.

After `pixi install` or updating the checkout, run `pixi run install-vllm`
to reconcile runtime dependencies. Preview resolution with
`pixi run install-vllm --dry-run`.

Smoke tests download `Qwen/Qwen2.5-0.5B-Instruct` by default and require a
nonempty completion. Set `VLLM_SMOKE_MODEL` for another model. They use eager
execution; `smoke-segmented` disables startup autotuning and selects
`ROCM_SEGMENTED_ATTN`. gfx1151 supports FP16/BF16 KV with this backend;
FP8 KV requires gfx12. `smoke-spec` adds n-gram speculative decoding.
Logs from local verification are under `logs/`.

## Verified on 2026-10-02

Checkout: `perf/rocm-segmented-attn` at
`f679f4e539ed8ebfb54bcc8c79d7c8f69df03650` (clean tracked files).

- Native editable installation passed.
- `uv pip check`: all 234 installed packages compatible.
- `check-torch`: Torch `2.13.0+rocm10.0.0`, ROCm `10.0.0`, GPU tensor check passed.
- `smoke`: Qwen2.5-0.5B-Instruct generation passed with `ROCM_ATTN`.
- `smoke-segmented`: generation passed with `ROCM_SEGMENTED_ATTN` and FP16 KV.
- Both completions: `Paris. It is the largest city in`.
- `test-segmented`: 69 passed, 31 skipped, 62 deselected; unsupported GPU cases skipped.
- Installer resolver dry run passed and proposed replacing only editable vLLM.

Generation checks used eager execution. Graph capture, startup autotuning,
speculative decoding, and performance were not validated in this run.

The older `10/vllm` and `10/vllm-pr-52619` source trees and the old environment's
editable vLLM installation were removed. Their local patches, checkout hashes,
and untracked helper scripts were saved outside the workspace in
`/home/akk/jeff/vllm-cleanup-backup-20261002`.
