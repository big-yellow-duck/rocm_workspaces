import torch

print(f"torch={torch.__version__}, HIP={torch.version.hip}")
assert torch.__version__.startswith("2.13.0+rocm10.0.0")
assert torch.version.hip is not None
assert torch.cuda.is_available(), "ROCm GPU is unavailable"
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"device={index}: {props.name}, arch={props.gcnArchName}")
    assert props.gcnArchName.split(":")[0] == "gfx1201", props.gcnArchName
    result = (torch.ones(2, device=f"cuda:{index}") + 1).cpu().tolist()
    assert result == [2.0, 2.0], result
print("gfx1201 GPU tensor checks: PASS")
