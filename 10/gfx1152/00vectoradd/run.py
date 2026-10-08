"""Validate the naive and winner kernels on 1024x1024 FP32 inputs."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import naive
import torch
import winner


def profiler_control():
    roctx = ctypes.CDLL(
        str(Path(os.environ["ROCM_PATH"]) / "lib/librocprofiler-sdk-roctx.so")
    )
    for name in ("roctxProfilerPause", "roctxProfilerResume"):
        function = getattr(roctx, name)
        function.argtypes = [ctypes.c_uint64]
        function.restype = ctypes.c_int
    roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
    roctx.roctxRangePushA.restype = ctypes.c_int
    roctx.roctxRangePop.argtypes = []
    roctx.roctxRangePop.restype = ctypes.c_int
    return roctx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("naive", "winner", "both"), default="both")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.profile and (args.case == "both" or args.metadata is None):
        parser.error("profiling requires one case and --metadata")
    roctx = profiler_control() if args.profile else None
    if roctx:
        roctx.roctxProfilerPause(0)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1152"
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    cases = ("naive", "winner") if args.case == "both" else (args.case,)
    with torch.cuda.stream(stream):
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        expected = a + b
        for case in cases:
            kernel = {"naive": naive.add, "winner": winner.add}[case]
            out.fill_(float("nan"))
            for _ in range(50 if args.profile else 1):
                kernel(a, b, out, stream=stream)
            stream.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            if roctx:
                roctx.roctxProfilerResume(0)
                roctx.roctxRangePushA(case.encode())
                try:
                    kernel(a, b, out, stream=stream)
                    stream.synchronize()
                finally:
                    roctx.roctxRangePop()
                    roctx.roctxProfilerPause(0)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
            else:
                # Dependent launches check stream ordering and memory visibility.
                scratch = torch.empty_like(out)
                chained_expected = expected.clone()
                for _ in range(4):
                    kernel(out, b, scratch, stream=stream)
                    out, scratch = scratch, out
                    chained_expected = chained_expected + b
                stream.synchronize()
                torch.testing.assert_close(out, chained_expected, rtol=0, atol=0)
            print(f"PASS: {case}, 1024x1024 FP32", flush=True)
    if args.metadata:
        args.metadata.write_text(
            json.dumps(
                {
                    "case": args.case,
                    "shape": [1024, 1024],
                    "dtype": "float32",
                    "gpu": props.name,
                    "arch": props.gcnArchName,
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "warmup": 50,
                    "profiled_launches": 1,
                    "exact_result_passed": True,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
