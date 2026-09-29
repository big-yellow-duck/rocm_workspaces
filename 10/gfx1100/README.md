# ROCm 10 vLLM development workspace

This workspace targets the local `gfx1100` GPU. `pixi.lock` pins Python 3.12,
ROCm 10.0.0, AMD PyTorch 2.13.0, GCC 14, and the fork's Python requirements.
The vLLM fork is an editable checkout.

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
pixi run check-torch
pixi run install-vllm
pixi run smoke
pixi run smoke-segmented
```

The `install-vllm` task runs the fork's `use_existing_torch.py --prefix` to
remove its incompatible PyTorch pins, then builds the checkout in editable
mode without resolving dependencies. Pixi installs all Python and build
dependencies first. The task also patches the ROCm 10 rocPRIM header inside
the Pixi environment so HIP can compile its device-side placement-new path.
The activation script exposes the AMD SMI 27 Python binding bundled with the
ROCm SDK. That helper changes tracked dependency files in
`vllm-rocm-segmented-attn/`; this is expected. Re-run the task after changing
build code.

Set `VLLM_SMOKE_MODEL` to use a different Hugging Face model for the smoke
test. The default downloads `Qwen/Qwen2.5-0.5B-Instruct` and checks that a
short prompt yields a nonempty completion.

`smoke-segmented` selects the fork's `ROCM_SEGMENTED_ATTN` backend on this
`gfx1100` GPU with the normal FP16 KV cache. The backend allows FP16/BF16 KV
on `gfx1x`; its `gfx12` restriction applies to FP8 KV.
