"""Validate native gfx1152 execution and sample AMD SMI during GPU work."""

import json
import subprocess
import threading
import time
from importlib.metadata import version

import torch


def smi(*args: str) -> None:
    subprocess.run(["amd-smi", *args], check=True)


def main() -> None:
    smi("version")
    smi("static", "--asic")
    assert torch.__version__.startswith("2.13.0+rocm10.0.0"), torch.__version__
    assert version("rocm") == "10.0.0", version("rocm")
    assert torch.version.hip is not None, "PyTorch lacks HIP support"
    assert torch.cuda.is_available(), "ROCm GPU unavailable"
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1152", props.gcnArchName
    print(f"Torch {torch.__version__}; HIP {torch.version.hip}; {props}", flush=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    size = 512
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    expected = a @ b
    gpu_a, gpu_b = a.cuda(), b.cuda()
    actual = gpu_a @ gpu_b
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    # Exercise autograd as well as elementwise kernels and BLAS.
    x = torch.linspace(-1, 1, 4096, device="cuda", requires_grad=True)
    x.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    # Use a larger GEMM for a bounded monitoring window (~15 seconds).
    large_a = torch.randn(2048, 2048, device="cuda")
    large_b = torch.randn_like(large_a)
    torch.cuda.synchronize()
    failures: list[Exception] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            try:
                smi("metric", "--usage", "--power", "--temperature")
            except (subprocess.CalledProcessError, OSError) as exc:
                failures.append(exc)
                return
            stop.wait(2)

    worker = threading.Thread(target=monitor)
    worker.start()
    iterations = 0
    start = time.monotonic()
    try:
        while time.monotonic() - start < 15:
            output = large_a @ large_b
            torch.cuda.synchronize()
            iterations += 1
        assert torch.isfinite(output).all().item()
    finally:
        stop.set()
        worker.join()
    assert not failures, failures
    elapsed = time.monotonic() - start
    print(
        json.dumps(
            {
                "status": "PASS",
                "architecture": props.gcnArchName,
                "gemm_size": 2048,
                "iterations": iterations,
                "elapsed_seconds": round(elapsed, 3),
                "checks": [
                    "FP32 GEMM versus CPU",
                    "autograd",
                    "finite GPU results",
                    "AMD SMI monitoring",
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
