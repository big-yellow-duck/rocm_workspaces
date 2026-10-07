"""Check the tuned add, including dependent launches on a non-default stream."""

import torch
from flydsl_vectoradd import BEST_CONFIG, add_1024


def main() -> None:
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn_like(a)
        current = torch.empty_like(a)
        scratch = torch.empty_like(a)
        expected = a + b
        add_1024(a, b, current, stream=stream)
        for _ in range(8):
            add_1024(current, b, scratch, stream=stream)
            current, scratch = scratch, current
            expected = expected + b
        stream.synchronize()
        torch.testing.assert_close(current, expected, rtol=0, atol=0)
    print(f"PASS: tuned 1024x1024 FP32 add and eight dependent launches; {BEST_CONFIG}")


if __name__ == "__main__":
    main()
