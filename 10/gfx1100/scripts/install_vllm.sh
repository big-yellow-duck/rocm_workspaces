#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="${workspace_dir}/vllm-rocm-segmented-attn"
if [[ ! -f "${repo_dir}/setup.py" ]]; then
  echo "Missing vLLM source checkout at ${repo_dir}" >&2
  exit 1
fi
cd "${repo_dir}"
"${workspace_dir}/.venv/bin/python" "${workspace_dir}/scripts/patch_rocprim.py"
export MAX_JOBS="${MAX_JOBS:-4}"
# Preserve the Pixi-managed stack while uv resolves vLLM's runtime dependencies.
constraints_file="$(mktemp)"
trap 'rm -f "${constraints_file}"' EXIT
"${workspace_dir}/.venv/bin/python" - "${workspace_dir}/pixi.toml" > "${constraints_file}" <<'PYTHON'
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

manifest = tomllib.loads(Path(sys.argv[1]).read_text())
for package in manifest["pypi-dependencies"]:
    print(f"{package}=={version(package)}")
PYTHON
uv pip install --python "${workspace_dir}/.venv/bin/python" \
  --no-build-isolation --constraints "${constraints_file}" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
  --index-strategy unsafe-best-match \
  -e . "$@"
