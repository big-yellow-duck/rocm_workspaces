# ROCm 10 workspaces

Use the workspace matching your GPU:

- [gfx1100](gfx1100/README.md)
- [gfx1151 — AMD Radeon 8060S](gfx1151/README.md)
- [gfx1201 — Radeon AI PRO R9700](gfx1201/README.md)

The [project README](../README.md) includes the current gfx1151 quick start
and verification summary. GPU-specific workspaces pin the ROCm/PyTorch stack
and provide build tools through Pixi, while vLLM resolves its runtime
dependencies with uv.

The manifest and scripts directly in this directory are legacy configuration.
The old shared environment's vLLM installation and the `vllm/` and
`vllm-pr-52619/` source trees were removed; the current gfx1151 installation
and `perf/rocm-segmented-attn` checkout are under `gfx1151/`.
