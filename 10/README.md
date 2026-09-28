# ROCm 10 vLLM development workspace

This workspace targets the local `gfx1151` GPU. `pixi.lock` pins Python 3.12,
ROCm 10.0.0, AMD PyTorch 2.13.0, and the fork's Python requirements. The vLLM
fork is an editable checkout.

```bash
git clone --branch perf/rocm-segmented-attn \
  https://github.com/big-yellow-duck/vllm.git vllm
cd vllm && git checkout 4f208193293dde825c367f760559ba0faa727cfb && cd ..
pixi install --locked
pixi run check-torch
pixi run install-vllm
pixi run smoke
pixi run smoke-segmented
```

The `install-vllm` task runs the fork's `use_existing_torch.py --prefix` to
remove its incompatible PyTorch pins, then builds the checkout in editable
mode without resolving dependencies. Pixi installs all Python and build
dependencies first. That helper changes tracked dependency files in `vllm/`;
this is expected. Re-run the task after changing build code.

The local `vendor/amdsmi` package is the AMD SMI 27.0.0 Python binding shipped
with ROCm 10.0.0. The fork's older `amdsmi==7.0.2` build pin cannot load
ROCm 10's `libamd_smi.so.27`.

Set `VLLM_SMOKE_MODEL` to use a different Hugging Face model for the smoke
test. The default downloads `Qwen/Qwen2.5-0.5B-Instruct` and checks that a
short prompt yields a nonempty completion.

`smoke-segmented` selects the fork's `ROCM_SEGMENTED_ATTN` backend on this
`gfx1151` GPU with the normal FP16 KV cache. The backend allows FP16/BF16 KV
on `gfx1x`; its `gfx12` restriction applies to FP8 KV.
