# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Specialized 1024x1024 FP32 kernels adapted from FlyDSL's vector-add example.

The unmasked JIT functions are low-level tuning entry points. Inputs must be
contiguous FP32 tensors of exactly 1024x1024; callers must validate that contract.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.primitive import CopyOpUniversalCopyType
from vectoradd_asm import vector_add as vector_add_asm

DEFAULT_STREAM = fx.Stream(None)

COPY_OPS = {
    1: fx.UniversalCopy32b,
    2: fx.UniversalCopy64b,
    4: fx.UniversalCopy128b,
    8: lambda: CopyOpUniversalCopyType.get(256),
    16: lambda: CopyOpUniversalCopyType.get(512),
}


@flyc.kernel
def vector_add_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    tiled_copy: fx.TiledCopy,
    vector_width: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    tile = tiled_copy.tile_mn
    ga = fx.flat_divide(a, tile)[None, None, 0, bid]
    gb = fx.flat_divide(b, tile)[None, None, 0, bid]
    gc = fx.flat_divide(c, tile)[None, None, 0, bid]
    thr_copy = tiled_copy.get_slice(tid)
    ta = thr_copy.partition_S(ga)
    tb = thr_copy.partition_S(gb)
    tc = thr_copy.partition_D(gc)
    ra = fx.make_fragment_like(ta)
    rb = fx.make_fragment_like(tb)
    rc = fx.make_fragment_like(tc)
    copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
    fx.copy(copy_atom, ta, ra)
    fx.copy(copy_atom, tb, rb)
    rc.store(ra.load() + rb.load())
    fx.copy(copy_atom, rc, tc)


@flyc.jit
def vector_add(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    stream: fx.Stream = DEFAULT_STREAM,
):
    # The fixed shape and power-of-two block tiles remove edge masking.
    count = 1024 * 1024
    layout = fx.make_layout((1, count), (count, 1))
    a_flat = fx.make_view(fx.get_iter(a), layout)
    b_flat = fx.make_view(fx.get_iter(b), layout)
    c_flat = fx.make_view(fx.get_iter(c), layout)
    copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((1, block_size), order=(1, 0)),
        fx.make_ordered_layout((1, values_per_thread), order=(0, 1)),
    )
    vector_add_kernel(a_flat, b_flat, c_flat, tiled_copy, vector_width).launch(
        grid=(count // (block_size * values_per_thread), 1, 1),
        block=(block_size, 1, 1),
        stream=stream,
    )


@flyc.jit
def vector_add_striped(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    stream: fx.Stream = DEFAULT_STREAM,
):
    count = 1024 * 1024
    layout = fx.make_layout((1, count), (count, 1))
    a_flat = fx.make_view(fx.get_iter(a), layout)
    b_flat = fx.make_view(fx.get_iter(b), layout)
    c_flat = fx.make_view(fx.get_iter(c), layout)
    copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
    tiled_copy = fx.make_tiled_copy(
        copy_atom,
        fx.make_layout(
            (block_size, (vector_width, values_per_thread // vector_width)),
            (vector_width, (1, block_size * vector_width)),
        ),
        (1, block_size * values_per_thread),
    )
    vector_add_kernel(a_flat, b_flat, c_flat, tiled_copy, vector_width).launch(
        grid=(count // (block_size * values_per_thread), 1, 1),
        block=(block_size, 1, 1),
        stream=stream,
    )


@flyc.kernel
def vector_add_persistent_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    tiled_copy: fx.TiledCopy,
    vector_width: fx.Constexpr[int],
    grid_size: fx.Constexpr[int],
    tile_count: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    for chunk in fx.range_constexpr(tile_count // grid_size):
        tile_index = fx.block_idx.x + chunk * grid_size
        thr_copy = tiled_copy.get_slice(tid)
        copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
        tile = tiled_copy.tile_mn
        ga = fx.flat_divide(a, tile)[None, None, 0, tile_index]
        gb = fx.flat_divide(b, tile)[None, None, 0, tile_index]
        gc = fx.flat_divide(c, tile)[None, None, 0, tile_index]
        ta = thr_copy.partition_S(ga)
        tb = thr_copy.partition_S(gb)
        tc = thr_copy.partition_D(gc)
        ra = fx.make_fragment_like(ta)
        rb = fx.make_fragment_like(tb)
        rc = fx.make_fragment_like(tc)
        fx.copy(copy_atom, ta, ra)
        fx.copy(copy_atom, tb, rb)
        rc.store(ra.load() + rb.load())
        fx.copy(copy_atom, rc, tc)


@flyc.jit
def vector_add_persistent(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    grid_size: fx.Constexpr[int],
    stream: fx.Stream = DEFAULT_STREAM,
):
    count = 1024 * 1024
    layout = fx.make_layout((1, count), (count, 1))
    a_flat = fx.make_view(fx.get_iter(a), layout)
    b_flat = fx.make_view(fx.get_iter(b), layout)
    c_flat = fx.make_view(fx.get_iter(c), layout)
    copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
    tiled_copy = fx.make_tiled_copy(
        copy_atom,
        fx.make_layout(
            (block_size, (vector_width, values_per_thread // vector_width)),
            (vector_width, (1, block_size * vector_width)),
        ),
        (1, block_size * values_per_thread),
    )
    vector_add_persistent_kernel(
        a_flat,
        b_flat,
        c_flat,
        tiled_copy,
        vector_width,
        grid_size,
        count // (block_size * values_per_thread),
    ).launch(grid=(grid_size, 1, 1), block=(block_size, 1, 1), stream=stream)


@flyc.kernel
def vector_add_unsigned_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    tiled_copy: fx.TiledCopy,
    vector_width: fx.Constexpr[int],
):
    tid = fx.Uint64(fx.Uint32(fx.thread_idx.x))
    bid = fx.Uint64(fx.Uint32(fx.block_idx.x))
    tile = tiled_copy.tile_mn
    ga = fx.flat_divide(a, tile)[None, None, 0, bid]
    gb = fx.flat_divide(b, tile)[None, None, 0, bid]
    gc = fx.flat_divide(c, tile)[None, None, 0, bid]
    thr_copy = tiled_copy.get_slice(tid)
    ta = thr_copy.partition_S(ga)
    tb = thr_copy.partition_S(gb)
    tc = thr_copy.partition_D(gc)
    ra = fx.make_fragment_like(ta)
    rb = fx.make_fragment_like(tb)
    rc = fx.make_fragment_like(tc)
    copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
    fx.copy(copy_atom, ta, ra)
    fx.copy(copy_atom, tb, rb)
    rc.store(ra.load() + rb.load())
    fx.copy(copy_atom, rc, tc)


@flyc.jit
def vector_add_unsigned(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    stream: fx.Stream = DEFAULT_STREAM,
):
    # The fixed shape and power-of-two block tiles remove edge masking.
    count = 1024 * 1024
    layout = fx.make_layout((1, count), (count, 1))
    a_flat = fx.make_view(fx.get_iter(a), layout)
    b_flat = fx.make_view(fx.get_iter(b), layout)
    c_flat = fx.make_view(fx.get_iter(c), layout)
    copy_atom = fx.make_copy_atom(COPY_OPS[vector_width](), fx.Float32)
    tiled_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_ordered_layout((1, block_size), order=(1, 0)),
        fx.make_ordered_layout((1, values_per_thread), order=(0, 1)),
    )
    vector_add_unsigned_kernel(a_flat, b_flat, c_flat, tiled_copy, vector_width).launch(
        grid=(count // (block_size * values_per_thread), 1, 1),
        block=(block_size, 1, 1),
        stream=stream,
    )


# Updated only after a separate, randomized paired validation run.
BEST_CONFIG = {
    "layout": "asm",
    "block_size": 64,
    "vector_width": 2,
    "values_per_thread": 2,
    "grid_size": 8192,
    "load_flags": 2,
    "store_flags": 2,
}


def add_1024(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Launch the tuned forward-only gfx1152 kernel into a preallocated output."""
    for tensor in (a, b, out):
        if tensor.shape != (1024, 1024) or tensor.dtype != torch.float32:
            raise ValueError("add_1024 requires 1024x1024 FP32 tensors")
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.data_ptr() % 16:
            raise ValueError(
                "add_1024 requires contiguous, 16-byte-aligned GPU tensors"
            )
        if tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.requires_grad:
            raise ValueError("add_1024 is a forward-only kernel")
    props = torch.cuda.get_device_properties(a.device)
    if props.gcnArchName.split(":")[0] != "gfx1152":
        raise ValueError("add_1024 is tuned for gfx1152")
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    if BEST_CONFIG["layout"] == "asm":
        vector_add_asm(
            a,
            b,
            out,
            BEST_CONFIG["block_size"],
            BEST_CONFIG["vector_width"],
            BEST_CONFIG["load_flags"],
            BEST_CONFIG["store_flags"],
            stream=stream,
        )
    else:
        vector_add(
            a,
            b,
            out,
            BEST_CONFIG["block_size"],
            BEST_CONFIG["vector_width"],
            BEST_CONFIG["values_per_thread"],
            stream=stream,
        )
    return out


@flyc.kernel
def vector_add_direct_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    load_nt: fx.Constexpr[bool],
    store_nt: fx.Constexpr[bool],
):
    base = fx.Uint64(fx.Uint32(fx.block_idx.x)) * (block_size * values_per_thread)
    lane = fx.Uint64(fx.Uint32(fx.thread_idx.x)) * vector_width
    for chunk in fx.range_constexpr(values_per_thread // vector_width):
        offset = base + lane + chunk * block_size * vector_width
        va = fx.generic_load(
            fx.get_iter(a) + offset, count=vector_width, nontemporal=load_nt
        )
        vb = fx.generic_load(
            fx.get_iter(b) + offset, count=vector_width, nontemporal=load_nt
        )
        fx.generic_store(fx.get_iter(c) + offset, va + vb, nontemporal=store_nt)


@flyc.jit
def vector_add_direct(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    load_nt: fx.Constexpr[bool],
    store_nt: fx.Constexpr[bool],
    stream: fx.Stream = DEFAULT_STREAM,
):
    vector_add_direct_kernel(
        a, b, c, block_size, vector_width, values_per_thread, load_nt, store_nt
    ).launch(
        grid=(1048576 // (block_size * values_per_thread), 1, 1),
        block=(block_size, 1, 1),
        stream=stream,
    )


@flyc.kernel
def vector_add_2d_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    threads_n: fx.Constexpr[int],
    load_nt: fx.Constexpr[bool],
    store_nt: fx.Constexpr[bool],
):
    tid = fx.Uint32(fx.thread_idx.x)
    row = fx.Uint64(fx.Uint32(fx.block_idx.x)) * (block_size // threads_n) + fx.Uint64(
        tid // threads_n
    )
    col = (
        fx.Uint64(fx.Uint32(fx.block_idx.y)) * (threads_n * 4)
        + fx.Uint64(tid % threads_n) * 4
    )
    offset = row * 1024 + col
    va = fx.generic_load(fx.get_iter(a) + offset, count=4, nontemporal=load_nt)
    vb = fx.generic_load(fx.get_iter(b) + offset, count=4, nontemporal=load_nt)
    fx.generic_store(fx.get_iter(c) + offset, va + vb, nontemporal=store_nt)


@flyc.jit
def vector_add_2d(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    values_per_thread: fx.Constexpr[int],
    threads_n: fx.Constexpr[int],
    load_nt: fx.Constexpr[bool],
    store_nt: fx.Constexpr[bool],
    stream: fx.Stream = DEFAULT_STREAM,
):
    vector_add_2d_kernel(a, b, c, block_size, threads_n, load_nt, store_nt).launch(
        grid=(1024 // (block_size // threads_n), 1024 // (threads_n * 4), 1),
        block=(block_size, 1, 1),
        stream=stream,
    )
