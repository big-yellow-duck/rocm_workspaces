#!/usr/bin/env bash
set -euo pipefail

# FlyDSL 0.3.4.1 has no declared Python dependencies. Keep Pixi's stack intact.
uv pip install --python "${CONDA_PREFIX}/bin/python" \
  --index-url https://pypi.org/simple --no-deps 'flydsl==0.3.4.1'
