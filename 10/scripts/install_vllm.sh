#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${workspace_dir}/vllm"

if [[ ! -f "${repo_dir}/use_existing_torch.py" ]]; then
  echo "Missing ${repo_dir}; clone perf/rocm-segmented-attn into vllm first." >&2
  exit 1
fi

cd "${repo_dir}"
python use_existing_torch.py --prefix
uv pip install --python "$(command -v python)" --no-build-isolation \
  --no-deps -e .
