# SPDX-License-Identifier: Apache-2.0
"""gfx1152 vector add using scalar bases and 32-bit per-lane offsets.

Specialized for contiguous 1024x1024 FP32 inputs. The checked public entry
point is flydsl_vectoradd.add_1024. Scalar-base addressing follows LLVM's
GFX11 global_load/global_store ISA syntax; vector widths are FP32 counts.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

FLAGS = {
    0: "",
    1: " glc",
    2: " slc",
    3: " glc slc",
    4: " dlc",
    5: " glc dlc",
    6: " slc dlc",
    7: " glc slc dlc",
}
DEFAULT_STREAM = fx.Stream(None)


def load_pair(
    a: fx.Tensor, b: fx.Tensor, offset: fx.Uint32, width: int, flags: int
) -> tuple[fx.Vector, fx.Vector]:
    ty = ir.VectorType.get([width], ir.F32Type.get())
    bits_ty = ir.IntegerType.get_signless(width * 32)
    pair_ty = llvm.StructType.get_literal([bits_ty, bits_ty])
    base_a = llvm.ptrtoint(ir.IntegerType.get_signless(64), fx.get_iter(a).llvm_ptr)
    base_b = llvm.ptrtoint(ir.IntegerType.get_signless(64), fx.get_iter(b).llvm_ptr)
    # Early-clobber protects the offset register until the second load reads it.
    # Wait for both asynchronous reads before the compiler consumes the values.
    pair = llvm.inline_asm(
        pair_ty,
        [fx.as_ir_value(offset), base_a, base_b],
        f"global_load_b{width * 32} $0, $2, $3{FLAGS[flags]}\n"
        f"global_load_b{width * 32} $1, $2, $4{FLAGS[flags]}\n"
        "s_waitcnt vmcnt(0)",
        "=&v,=&v,v,s,s,~{memory}",
        has_side_effects=True,
    )
    return (
        fx.Vector(
            llvm.bitcast(ty, llvm.extractvalue(bits_ty, pair, [0])),
            (width,),
            fx.Float32,
        ),
        fx.Vector(
            llvm.bitcast(ty, llvm.extractvalue(bits_ty, pair, [1])),
            (width,),
            fx.Float32,
        ),
    )


def store_vector(
    c: fx.Tensor, offset: fx.Uint32, value: fx.Vector, width: int, flags: int
) -> None:
    base_c = llvm.ptrtoint(ir.IntegerType.get_signless(64), fx.get_iter(c).llvm_ptr)
    llvm.inline_asm(
        None,
        [
            fx.as_ir_value(offset),
            base_c,
            llvm.bitcast(
                ir.IntegerType.get_signless(width * 32), fx.as_ir_value(value)
            ),
        ],
        f"global_store_b{width * 32} $0, $2, $1{FLAGS[flags]}",
        "v,s,v,~{memory}",
        has_side_effects=True,
    )


@flyc.kernel
def vector_add_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    load_flags: fx.Constexpr[int],
    store_flags: fx.Constexpr[int],
):
    offset = (fx.Uint32(fx.block_idx.x) * block_size + fx.Uint32(fx.thread_idx.x)) * (
        vector_width * 4
    )
    va, vb = load_pair(a, b, offset, vector_width, load_flags)
    store_vector(c, offset, va + vb, vector_width, store_flags)


@flyc.jit
def vector_add(
    a: fx.Tensor,
    b: fx.Tensor,
    c: fx.Tensor,
    block_size: fx.Constexpr[int],
    vector_width: fx.Constexpr[int],
    load_flags: fx.Constexpr[int],
    store_flags: fx.Constexpr[int],
    stream: fx.Stream = DEFAULT_STREAM,
):
    vector_add_kernel(
        a, b, c, block_size, vector_width, load_flags, store_flags
    ).launch(
        grid=(1048576 // (block_size * vector_width), 1, 1),
        block=(block_size, 1, 1),
        stream=stream,
    )
