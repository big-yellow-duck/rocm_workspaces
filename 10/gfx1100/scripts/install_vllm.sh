#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${workspace_dir}/vllm-rocm-segmented-attn"

if [[ ! -f "${repo_dir}/use_existing_torch.py" ]]; then
  echo "Missing ${repo_dir}; clone perf/rocm-segmented-attn into vllm-rocm-segmented-attn first." >&2
  exit 1
fi

cd "${repo_dir}"
python "${workspace_dir}/scripts/patch_rocprim.py"
python use_existing_torch.py --prefix
export MAX_JOBS="${MAX_JOBS:-4}"
uv pip install --python "$(command -v python)" --no-build-isolation \
  --no-deps -e .
