import torch

print(f"torch={torch.__version__}, HIP={torch.version.hip}")
assert torch.__version__.startswith("2.13.0+rocm10.0.0")
assert torch.version.hip is not None
assert torch.cuda.is_available(), "ROCm GPU is unavailable"
print(f"device={torch.cuda.get_device_name(0)}")
result = (torch.ones(2, device="cuda") + 1).cpu().tolist()
assert result == [2.0, 2.0], result
print("GPU tensor check: PASS")
