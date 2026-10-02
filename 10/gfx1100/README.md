# ROCm 10 / gfx1100 vLLM development workspace

This workspace targets `gfx1100` GPUs with Python 3.12, ROCm 10.0.0,
AMD PyTorch 2.13.0, and GCC 14. Pixi manages the ROCm/PyTorch stack plus
build, lint, and test tools. uv resolves vLLM's runtime dependencies directly
from the editable checkout; they are not duplicated in `pixi.toml`.

ROCm needs read/write access to `/dev/kfd` and the DRM render nodes. On a new
host, add your user to the `render` group once, then log out and back in:

```bash
sudo usermod -aG render "$USER"
```

```bash
git clone --branch perf/rocm-segmented-attn \
  https://github.com/big-yellow-duck/vllm.git vllm-rocm-segmented-attn
cd vllm-rocm-segmented-attn
git checkout 4f208193293dde825c367f760559ba0faa727cfb
cd ..
pixi install --locked
ln -sfn .pixi/envs/default .venv
pixi run check-torch
pixi run install-vllm
pixi run smoke
pixi run smoke-segmented
```

`.venv` links to the Pixi environment. Use `pixi run` or `pixi shell` to
activate the ROCm SDK, AMD SMI binding, compiler, and library paths.

`install-vllm` uses `--no-build-isolation` and constrains explicitly
Pixi-managed Python packages to their installed versions. uv installs vLLM's
runtime requirements while preserving the ROCm/PyTorch and build-tool stack.
vLLM's own version constraints still apply. The resolver uses PyPI and AMD's
index with `unsafe-best-match`, matching Pixi's index policy.

This revision's ROCm runtime requirements do not pin PyTorch, so the installer
does not run `use_existing_torch.py` or modify tracked dependency files.
Existing edits made by earlier runs of that helper are preserved. The build
patches the ROCm 10 rocPRIM header inside the environment and exposes the
AMD SMI Python binding bundled with the SDK. `MAX_JOBS` defaults to 4.

After `pixi install` or changing vLLM revisions, run `pixi run install-vllm`
to reconcile runtime dependencies. Preview changes with
`pixi run install-vllm --dry-run`.

Set `VLLM_SMOKE_MODEL` to choose a different model. The default is
`Qwen/Qwen2.5-0.5B-Instruct`; the smoke check requires a nonempty completion.
`smoke-segmented` selects `ROCM_SEGMENTED_ATTN` with FP16 KV cache and
startup autotuning disabled. FP8 KV requires `gfx12` hardware.

## Validated on 2026-10-02

- Fresh `pixi install --locked` and native editable `install-vllm`: passed.
- `check-torch`: passed on AMD Radeon RX 7900 XTX; PyTorch remains
  `2.13.0+rocm10.0.0` and ROCm remains `10.0.0`.
- `uv pip check`: all 234 installed packages compatible.
- `smoke` and `smoke-segmented`: passed with Qwen2.5-0.5B-Instruct;
  both returned `Paris. It is the largest city in`.

These smoke checks use eager execution and FP16 KV cache. Logs are in `logs/`.
The running session had not picked up the account's existing render-group
membership, so GPU checks used `sg render -c 'pixi run <task>'`.

The workspace uses the fresh, validated `.pixi/envs/default` environment;
`.venv` points to it. The previous environment backup has been removed.
