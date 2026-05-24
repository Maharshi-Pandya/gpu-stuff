from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr
from cutlass._mlir.dialects import llvm, vector
from cutlass._mlir.dialects import arith as _arith
from cutlass.cutlass_dsl import T, dsl_user_op, NumericMeta


StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


@dsl_user_op
def make_vector(elem_type, *values, loc=None, ip=None):
    """Build an MLIR vector <N x elem_type> from N scalar DSL values.

    Example: make_vector(cutlass.Uint32, v0, v1) -> <2 x i32> MLIR vector
    """
    from cutlass._mlir import ir

    n = len(values)
    mlir_ty = elem_type.mlir_type
    vec_ty = ir.VectorType.get([n], mlir_ty)
    vec = llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
    for i, v in enumerate(values):
        vec = vector.insertelement(
            elem_type(v).ir_value(loc=loc, ip=ip),
            vec,
            position=_arith.constant(T.i32(), i, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    return vec


@dsl_user_op
@cute.jit
def store(
    ptr: cute.Pointer,
    val,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Store a scalar value via cute.arch.store.

    ptr:  cute.Pointer (any address space).
    val:  DSL Numeric value.
    pred: None → unconditional.  DSL Boolean → skipped when pred == 0.
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, type(val)(val), cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, type(val)(val), cop=cop, loc=loc, ip=ip)


@dsl_user_op
@cute.jit
def store_v2(
    ptr: cute.Pointer,
    v0,
    v1,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Vectorized store of 2 elements via cute.arch.store.

    Packs v0, v1 into an MLIR <2 x T> vector.
    ptr:  cute.Pointer (any address space, must be aligned for vector width).
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    vec = make_vector(type(v0), v0, v1, loc=loc, ip=ip)
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)


def _partition_fields(obj):
    """Split dataclass fields into (constexpr_dict, non_constexpr_dict) by type."""
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
    non_constexpr = {n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)}
    return constexpr, non_constexpr


def _new_from_mlir_values(self, values):
    constexpr_fields, non_constexpr_fields = _partition_fields(self)
    for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
        values = values[n_items:]
    return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values
