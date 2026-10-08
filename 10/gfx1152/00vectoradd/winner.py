# SPDX-License-Identifier: Apache-2.0
"""The gfx1152 winner: 64 threads, 8192 blocks, two FP32 values/thread."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

DEFAULT_STREAM = fx.Stream(None)


def load_pair(
    a: fx.Tensor, b: fx.Tensor, offset: fx.Uint32
) -> tuple[fx.Vector, fx.Vector]:
    vector_type = ir.VectorType.get([2], ir.F32Type.get())
    bits_type = ir.IntegerType.get_signless(64)
    pair_type = llvm.StructType.get_literal([bits_type, bits_type])
    base_a = llvm.ptrtoint(bits_type, fx.get_iter(a).llvm_ptr)
    base_b = llvm.ptrtoint(bits_type, fx.get_iter(b).llvm_ptr)
    # Early-clobber keeps the offset intact until both loads have read it.
    pair = llvm.inline_asm(
        pair_type,
        [fx.as_ir_value(offset), base_a, base_b],
        "global_load_b64 $0, $2, $3 slc\n"
        "global_load_b64 $1, $2, $4 slc\n"
        "s_waitcnt vmcnt(0)",
        "=&v,=&v,v,s,s,~{memory}",
        has_side_effects=True,
    )
    return (
        fx.Vector(
            llvm.bitcast(vector_type, llvm.extractvalue(bits_type, pair, [0])),
            (2,),
            fx.Float32,
        ),
        fx.Vector(
            llvm.bitcast(vector_type, llvm.extractvalue(bits_type, pair, [1])),
            (2,),
            fx.Float32,
        ),
    )


def store_pair(c: fx.Tensor, offset: fx.Uint32, value: fx.Vector) -> None:
    bits_type = ir.IntegerType.get_signless(64)
    base_c = llvm.ptrtoint(bits_type, fx.get_iter(c).llvm_ptr)
    llvm.inline_asm(
        None,
        [
            fx.as_ir_value(offset),
            base_c,
            llvm.bitcast(bits_type, fx.as_ir_value(value)),
        ],
        "global_store_b64 $0, $2, $1 slc",
        "v,s,v,~{memory}",
        has_side_effects=True,
    )


@flyc.kernel
def winner_kernel(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor):
    # Every thread handles two floats = eight bytes.
    offset = (fx.Uint32(fx.block_idx.x) * 64 + fx.Uint32(fx.thread_idx.x)) * 8
    va, vb = load_pair(a, b, offset)
    store_pair(c, offset, va + vb)


@flyc.jit
def add(a: fx.Tensor, b: fx.Tensor, c: fx.Tensor, stream: fx.Stream = DEFAULT_STREAM):
    """Add contiguous, aligned 1024x1024 FP32 tensors on gfx1152."""
    winner_kernel(a, b, c).launch(grid=(8192, 1, 1), block=(64, 1, 1), stream=stream)
