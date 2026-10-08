"""Capture the naive and winner kernels for ROCprof Compute Viewer."""

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    capture = root / "profiles" / datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    source = capture / "source"
    source.mkdir(parents=True, exist_ok=False)
    for name in ("naive.py", "winner.py", "run.py"):
        shutil.copy2(root / name, source / name)
    rocprof = shutil.which("rocprofv3")
    if rocprof is None:
        raise SystemExit("Run this with pixi run profile-vectoradd")
    power = Path("/sys/class/drm/card1/device/power_dpm_force_performance_level")
    original = power.read_text().strip()
    index = {}
    try:
        power.write_text("profile_standard\n")
        for case in ("naive", "winner"):
            output = capture / case
            output.mkdir()
            command = [
                rocprof,
                "--disable-signal-handlers",
                "true",
                "--selected-regions",
                "--att",
                "--att-library-path",
                str(Path(os.environ["ROCM_PATH"]) / "lib"),
                "--att-target-cu",
                "1",
                "--att-simd-select",
                "3",
                "--att-shader-engine-mask",
                "0x1",
                "--att-buffer-size",
                "100663296",
                "--att-gpu-index",
                "0",
                "--kernel-trace",
                "--output-format",
                "csv",
                "--output-directory",
                str(output),
                "--output-file",
                case,
                "--",
                sys.executable,
                str(source / "run.py"),
                "--case",
                case,
                "--profile",
                "--metadata",
                str(output / "workload.json"),
            ]
            (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            print(f"Profiling {case}...", flush=True)
            with (output / "rocprof.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=capture,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            workload = json.loads((output / "workload.json").read_text())
            assert workload["exact_result_passed"]
            raw = list(output.glob("*.att"))
            assert raw and any(path.stat().st_size > 0 for path in raw), (
                "Empty ATT trace"
            )
            ui = next(output.glob("ui_output_*/filenames.json")).parent
            files = json.loads((ui / "filenames.json").read_text())
            assert files["gfxip"] == 11 and files["wave_filenames"]
            waves = [
                name
                for shaders in files["wave_filenames"].values()
                for slots in shaders.values()
                for instances in slots.values()
                for name, *_ in instances.values()
            ]
            for name in waves:
                json.loads((ui / name).read_text())
            code = json.loads((ui / "code.json").read_text())["code"]
            assert any(row[0].startswith(f"; {case}_kernel") for row in code)
            with (output / f"{case}_kernel_trace.csv").open() as handle:
                dispatches = list(csv.DictReader(handle))
            assert (
                len(dispatches) == 1
                and f"{case}_kernel" in dispatches[0]["Kernel_Name"]
            )
            stats = next(output.glob("stats_*.csv"))
            with stats.open() as handle:
                instructions = list(csv.DictReader(handle))
            assert instructions and any(
                int(row["Hitcount"]) > 0 for row in instructions
            )
            (output / "instructions.txt").write_text(
                "\n".join(row[0] for row in code) + "\n"
            )
            index[case] = str(ui.relative_to(root))
            print(f"Ready: {case} ({len(waves)} decoded waves)", flush=True)
    finally:
        power.write_text(original + "\n")
        print(f"Restored GPU power mode: {power.read_text().strip()}", flush=True)
    (root / "profiles/latest.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        "Open with: pixi run view-vectoradd naive  /  pixi run view-vectoradd winner",
        flush=True,
    )


if __name__ == "__main__":
    main()
