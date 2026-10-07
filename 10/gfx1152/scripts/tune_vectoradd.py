"""Tune contiguous 1024x1024 FP32 vector addition on the native gfx1152 GPU.

All kernels compile before measurement. HIP events time graph batches on one
stream, comparing the same tensors and preallocated output with torch.add.
"""

import argparse
import json
import random
import statistics
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from flydsl_vectoradd import (
    BEST_CONFIG,
    vector_add,
    vector_add_2d,
    vector_add_direct,
    vector_add_persistent,
    vector_add_striped,
    vector_add_unsigned,
)
from vectoradd_asm import vector_add as vector_add_asm
from vectorAdd_flydsl import vector_add as original_vector_add

LAUNCHES = 100


def capture(fn, stream):
    for _ in range(3):
        fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(LAUNCHES):
            fn()
    return graph


def measure(graph, stream, repeats=3):
    # Warmup and graph launches use the same stream as the timing events.
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(repeats):
        graph.replay()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (LAUNCHES * repeats)


def configs():
    for layout in ("contiguous", "striped"):
        for bs in (32, 64, 128, 256, 512, 1024):
            for vw in (1, 2, 4, 8, 16):
                for vt in (1, 2, 4, 8, 16, 32, 64):
                    if vt < vw or (layout == "contiguous" and vt == 64 and vw <= 4):
                        continue
                    if layout == "striped" and vt == vw:
                        continue
                    yield {
                        "layout": layout,
                        "block_size": bs,
                        "vector_width": vw,
                        "values_per_thread": vt,
                        "grid_size": 1048576 // (bs * vt),
                    }
    for bs in (32, 64, 128, 256, 512, 1024):
        for vw in (1, 2, 4):
            for load_flags in (0, 1, 2, 3, 6):
                for store_flags in (0, 2):
                    yield {
                        "layout": "asm",
                        "block_size": bs,
                        "vector_width": vw,
                        "values_per_thread": vw,
                        "grid_size": 1048576 // (bs * vw),
                        "load_flags": load_flags,
                        "store_flags": store_flags,
                    }
    for bs in (32, 64, 128, 256, 512, 1024):
        for vw in (1, 2, 4, 8, 16):
            for vt in (4, 8, 16):
                if vt < vw:
                    continue
                for load_nt, store_nt in (
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ):
                    yield {
                        "layout": "direct",
                        "block_size": bs,
                        "vector_width": vw,
                        "values_per_thread": vt,
                        "grid_size": 1048576 // (bs * vt),
                        "load_nt": load_nt,
                        "store_nt": store_nt,
                    }
    for bs in (64, 128, 256, 512, 1024):
        for vw in (1, 2, 4):
            yield {
                "layout": "unsigned",
                "block_size": bs,
                "vector_width": vw,
                "values_per_thread": 4,
                "grid_size": 1048576 // (bs * 4),
            }
    for bs in (64, 128, 256, 512, 1024):
        for vw in (1, 2, 4):
            for vt in (vw, vw * 2):
                tiles = 1048576 // (bs * vt)
                for chunks in (2, 4, 8, 16):
                    yield {
                        "layout": "persistent",
                        "block_size": bs,
                        "vector_width": vw,
                        "values_per_thread": vt,
                        "grid_size": tiles // chunks,
                    }
    for bs in (128, 256, 512, 1024):
        for tn in (16, 32, 64, 128, 256):
            if tn > bs:
                continue
            for load_nt, store_nt in ((False, False), (True, False), (True, True)):
                yield {
                    "layout": "2d",
                    "block_size": bs,
                    "vector_width": 4,
                    "values_per_thread": 4,
                    "threads_n": tn,
                    "grid_size": 1048576 // (bs * 4),
                    "load_nt": load_nt,
                    "store_nt": store_nt,
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="logs/vectoradd_tuning.json")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--best", action="store_true", help="Benchmark the selected configuration only"
    )
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="Fail unless the median speedup exceeds one",
    )
    parser.add_argument("--wide-only", action="store_true")
    parser.add_argument(
        "--layout",
        choices=(
            "all",
            "contiguous",
            "striped",
            "persistent",
            "unsigned",
            "direct",
            "2d",
            "asm",
        ),
        default="all",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--finalists",
        type=Path,
        help="Retest top ten configurations from a prior result file",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1152"
    candidates = [
        row for row in configs() if args.layout == "all" or row["layout"] == args.layout
    ]
    if args.best:
        candidates = [BEST_CONFIG.copy()]
    if args.wide_only:
        candidates = [row for row in candidates if row["vector_width"] > 4]
    if args.quick:
        candidates = [
            row
            for row in candidates
            if row["block_size"] == 128
            and row["vector_width"] == 4
            and row["values_per_thread"] in (4, 16, 64)
        ]
    if args.finalists:
        keys = (
            "layout",
            "block_size",
            "vector_width",
            "values_per_thread",
            "grid_size",
        )
        candidates = [
            {
                key: value
                for key, value in row.items()
                if key
                in (
                    *keys,
                    "load_nt",
                    "store_nt",
                    "threads_n",
                    "load_flags",
                    "store_flags",
                )
            }
            for row in json.loads(args.finalists.read_text())["candidates"][: args.top]
        ]
    torch.manual_seed(42)
    rng = random.Random(42)
    stream = torch.cuda.Stream()
    records = []
    with torch.cuda.stream(stream):
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        expected = a + b
        baseline = capture(partial(torch.add, a, b, out=c), stream)
        orig = capture(partial(original_vector_add, a, b, c, stream=stream), stream)
        for index, config in enumerate(candidates):
            kernel = {
                "striped": vector_add_striped,
                "contiguous": vector_add,
                "persistent": vector_add_persistent,
                "unsigned": vector_add_unsigned,
                "direct": vector_add_direct,
                "2d": vector_add_2d,
                "asm": vector_add_asm,
            }[config["layout"]]
            params = [
                config["block_size"],
                config["vector_width"],
                config["values_per_thread"],
            ]
            if config["layout"] == "asm":
                params = [
                    config["block_size"],
                    config["vector_width"],
                    config["load_flags"],
                    config["store_flags"],
                ]
            if config["layout"] == "2d":
                params.extend(
                    [config["threads_n"], config["load_nt"], config["store_nt"]]
                )
            if config["layout"] == "direct":
                params.extend([config["load_nt"], config["store_nt"]])
            if config["layout"] == "persistent":
                params.append(config["grid_size"])
            fn = partial(kernel, a, b, c, *params, stream=stream)
            c.fill_(float("nan"))
            fn()
            stream.synchronize()
            torch.testing.assert_close(c, expected, rtol=0, atol=0)
            graph = capture(fn, stream)
            records.append((config | {"samples": [], "correct": True}, graph))
            print(
                f"Compiled/checked {index + 1}/{len(candidates)}: {config}", flush=True
            )
        # Give CPU compilation activity time to subside, then warm the GPU.
        time.sleep(1)
        for _ in range(20):
            baseline.replay()
        stream.synchronize()
        original_samples = []
        for round_index in range(args.rounds):
            order = list(range(len(records)))
            rng.shuffle(order)
            for index in order:
                row, graph = records[index]
                # Randomized paired order reduces clock/temperature and order bias.
                pair = [("flydsl", graph), ("torch", baseline)]
                rng.shuffle(pair)
                timings = {
                    name: measure(item, stream, args.repeats) for name, item in pair
                }
                speedup = timings["torch"] / timings["flydsl"]
                row["samples"].append(timings | {"speedup": speedup})
                print(
                    f"Round {round_index + 1}/{args.rounds}: {row['layout']} "
                    f"block={row['block_size']} vec={row['vector_width']} "
                    f"values={row['values_per_thread']} grid={row['grid_size']} "
                    f"load_nt={row.get('load_nt', False)} store_nt={row.get('store_nt', False)} "
                    f"load_flags={row.get('load_flags', 0)} store_flags={row.get('store_flags', 0)}: "
                    f"{timings['flydsl']:.3f} us "
                    f"vs Torch {timings['torch']:.3f} us ({speedup:.3f}x)",
                    flush=True,
                )
            original_samples.append(measure(orig, stream, args.repeats))
    rows = [row for row, _ in records]
    for row in rows:
        row["us"] = statistics.median(sample["flydsl"] for sample in row["samples"])
        row["torch_us"] = statistics.median(
            sample["torch"] for sample in row["samples"]
        )
        row["speedup"] = statistics.median(
            sample["speedup"] for sample in row["samples"]
        )
    rows.sort(key=lambda row: row["speedup"], reverse=True)
    result = {
        "torch": torch.__version__,
        "flydsl": version("flydsl"),
        "gpu": props.name,
        "arch": props.gcnArchName,
        "shape": [1024, 1024],
        "dtype": "float32",
        "seed": 42,
        "method": "Randomized paired HIP event timings of 100-node GPU graphs",
        "launches_per_sample": LAUNCHES * args.repeats,
        "rounds": args.rounds,
        "original_us": statistics.median(original_samples),
        "candidates": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Top configurations:", json.dumps(rows[:10], indent=2), flush=True)
    print(f"Results: {output}", flush=True)
    if args.require_speedup and rows[0]["speedup"] <= 1:
        raise SystemExit("No median speedup over torch.add in this run")


if __name__ == "__main__":
    main()
