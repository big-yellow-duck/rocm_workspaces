#!/usr/bin/env bash
set -euo pipefail
example_root=$(cd "$(dirname "$0")" && pwd)
selection=${1:-winner}
ui_path=$(python3 - "$example_root" "$selection" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
case = sys.argv[2]
if case not in ("naive", "winner"):
    raise SystemExit("Choose naive or winner")
latest = root / "profiles/latest.json"
if not latest.is_file():
    raise SystemExit("Run pixi run profile-vectoradd first")
ui = root / json.loads(latest.read_text())[case]
if not (ui / "filenames.json").is_file():
    raise SystemExit(f"Missing profile: {ui}")
print(ui)
PY
)
exec pixi run --manifest-path "$HOME/rocprof-viewer/pixi.toml" ui "$ui_path"
