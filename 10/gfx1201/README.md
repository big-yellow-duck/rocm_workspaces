# ROCm 10 / gfx1201 vLLM workspace

Native development environment for the Radeon AI PRO R9700 GPUs. Adapted from
`../gfx1100`: Python 3.12, ROCm 10.0.0, AMD PyTorch 2.13.0, GCC 14, and
`device-gfx1201` libraries. Pixi locks the ROCm/PyTorch stack plus build, lint, and test tools. vLLM
resolves and installs its own runtime dependencies with uv.

```bash
cd /home/ellm/jeff/rocm_workspaces/10/gfx1201
pixi install --locked
ln -sfn .pixi/envs/default .venv
pixi run check-torch
pixi run install-vllm
pixi run smoke-segmented
pixi run test-segmented
```

`.venv` links to `.pixi/envs/default` so scripts use the Pixi interpreter
explicitly. Run commands through `pixi run` or `pixi shell` to activate the
ROCm SDK, AMD SMI binding, compiler, and library paths. Build commands target
only `gfx1201`; `MAX_JOBS` defaults to 4 and can be overridden.

`install-vllm` builds `vllm/` in editable mode using the installed PyTorch and
`--no-build-isolation`. Runtime dependency resolution stays enabled. The script
constrains explicitly Pixi-managed Python packages to their installed versions,
so transitive dependencies cannot replace the ROCm/PyTorch or build-tool stack.
Runtime packages such as transformers and fastapi are governed by vLLM's
requirements, not duplicated in `pixi.toml` or `pixi.lock`.

This checkout's `requirements/rocm.txt` and `requirements/common.txt` do not pin
PyTorch. `tools/use_existing_torch.py --prefix` is therefore unnecessary here;
its purpose is to remove Torch-family requirements from source files, not to
freeze installed versions. `--no-build-isolation` uses the build dependencies
already installed by Pixi. The script preserves tracked dependency files.

The build applies the same local ROCm 10 rocPRIM header workaround as the gfx1100
workspace; this changes only the SDK inside the environment.

The checkout is a Git worktree moved from `vllm-rocm-segmented-scoped`, on
`fix/rocm-segmented-backend-scope`, based on PR #59132 head
`a5db6e20e606a50be0ede60026b32051a2891804`. Its uncommitted refactor is preserved.
The parent Git repository remains at `/home/ellm/jeff/vllm-rocm-segmented-attn`.
`logs/backend-scope-wip.patch` is a snapshot of the edits before environment setup.

The smoke task disables startup autotuning for a quick engine check. Set
`VLLM_SMOKE_MODEL` to select another model; the default is
`Qwen/Qwen2.5-0.5B-Instruct`. The gfx1201 backend supports native FP16/BF16 KV
and FP8 KV. Kernel tests cover both.

## Refactor in progress

The intended PR removes changes to `gpu_worker.py`, `gpu/model_runner.py`, and
`worker/workspace.py`. Startup tuning moves to the attention post-load hook;
segmented scratch uses the existing persistent-resource API for runtime lane
and microbatch isolation. The refactor remains in progress: focused regression tests and eager
speculative generation pass, but real startup autotuning, graph capture at the
model level, and DFlash/DSpark validation remain before shipping. Nothing has been committed or pushed.

The activation script also supplies the Conda sysroot to HIP Clang. `patch` is
included for vLLM's PyTorch stable-header fix. `test-segmented` selects the
segmented tests only; other backend tests can require optional AITER packages.

`pixi run smoke-spec` enables n-gram speculative decoding with segmented
attention. For another draft method, set `VLLM_SMOKE_SPEC_CONFIG` to its JSON
configuration and run `smoke-segmented`. Set `VLLM_SMOKE_KV_CACHE_DTYPE=fp8` to
check FP8 KV on gfx1201. These are short startup/generation checks, not model
quality evaluations.


## Validated on 2026-09-29

- `pixi run check-torch`: both gfx1201 Radeon AI PRO R9700 GPUs passed.
- `pixi run install-vllm`: native editable build passed (about nine minutes).
- `pixi run test-segmented`: 93 passed, 62 unrelated tests deselected.
- `VLLM_SMOKE_MODEL=Qwen/Qwen3-0.6B pixi run smoke-segmented`: passed.
- The same smoke with `VLLM_SMOKE_KV_CACHE_DTYPE=fp8`: passed.
- `VLLM_SMOKE_MODEL=Qwen/Qwen3-0.6B pixi run smoke-spec`: passed (V1 runner).
- Draft-model speculative decoding with Qwen3-0.6B as both target and draft:
  passed (V2 runner, segmented attention on both models).
- Workspace script lint and refactor-file lint passed.

The draft-model check used:

```bash
VLLM_SMOKE_MODEL=Qwen/Qwen3-0.6B \
VLLM_SMOKE_SPEC_CONFIG='{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3,"attention_backend":"ROCM_SEGMENTED_ATTN"}' \
pixi run smoke-segmented
```

Smoke checks used eager execution with startup autotuning disabled. They are
not graph-capture, tuning-performance, or model-quality evaluations. Logs are
under `logs/`; the first pre-install test attempts were superseded by the
successful post-install `test-segmented.log` run.


After `pixi install` or changing vLLM revisions, run `pixi run install-vllm`
to reconcile vLLM's dependencies. Preview the resolver without installing with
`pixi run install-vllm --dry-run`. No `uv pip sync` or separate runtime lockfile
is needed. Pixi still locks transitive dependencies required by its own stack.

## Dependency ownership check

After trimming the manifest, the editable-install resolver dry run succeeded,
and vLLM's `requirements/rocm.txt` was installed with the same stack constraints
and index settings. The existing native build was retained. `uv pip check`
reported all installed packages compatible, and both GPU checks passed.
The native ROCm/PyTorch versions remained unchanged. Results for this environment
are in `logs/test-trimmed-env.log` and `logs/smoke-trimmed-env.log`.

uv uses the same `unsafe-best-match` index policy as Pixi: AMD's index also
contains NumPy, but its listed version need not satisfy vLLM's Numba constraint.
The resolver therefore considers compatible versions from PyPI as well.
