# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Dispatch large dense attention calls to an online-softmax GPU kernel."""

from typing import NamedTuple

from tvm import DataType, arith, relax, tirx
from tvm.ir import Expr, Op
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.relax import expr_functor

from .utils import BackendDispatcher


class AttentionDispatchInfo(NamedTuple):
    """Static kernel inputs for a supported attention call."""

    key_heads: int
    query_heads: int
    head_dim: int
    dtype: str
    scale: float
    score_buffer_bytes: int | None


def _tensor_shape(expr: relax.Expr) -> tuple[Expr, ...] | None:
    ty = expr.ty
    if not isinstance(ty, relax.TensorType) or ty.shape is None:
        return None
    return tuple(ty.shape)


def _multiplicative_factors(expr: Expr) -> list[Expr]:
    if isinstance(expr, tirx.Mul):
        return _multiplicative_factors(expr.a) + _multiplicative_factors(expr.b)
    return [expr]


def _can_prove_equal(analyzer: arith.Analyzer, lhs: Expr, rhs: Expr) -> bool:
    if analyzer.can_prove_equal(lhs, rhs):
        return True
    lhs_factors = _multiplicative_factors(lhs)
    rhs_factors = _multiplicative_factors(rhs)
    if len(lhs_factors) != len(rhs_factors):
        return False
    unmatched = list(rhs_factors)
    for lhs_factor in lhs_factors:
        for index, rhs_factor in enumerate(unmatched):
            if analyzer.can_prove_equal(lhs_factor, rhs_factor):
                unmatched.pop(index)
                break
        else:
            return False
    return True


def get_attention_dispatch_info(call: relax.Call) -> AttentionDispatchInfo | None:
    """Keep buffer accounting and dispatch eligibility on the same predicate."""

    if not isinstance(call.op, Op) or call.op.name != "relax.nn.attention":
        return None
    if call.attrs.window_size is not None or call.attrs.causal_mask is not None:
        return None

    query, key, value = call.args
    query_shape = _tensor_shape(query)
    key_shape = _tensor_shape(key)
    value_shape = _tensor_shape(value)
    if query_shape is None or key_shape is None or value_shape is None:
        return None
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return None

    batch, query_length, query_heads_expr, head_dim_expr = query_shape
    key_batch, key_length, key_heads_expr, key_dim = key_shape
    value_batch, value_length, value_heads_expr, value_dim = value_shape
    dtype = query.ty.dtype
    static_kernel_dims = (
        query_heads_expr,
        head_dim_expr,
        key_heads_expr,
        value_heads_expr,
    )
    if any(not isinstance(dim, tirx.IntImm) for dim in static_kernel_dims):
        return None
    query_heads = int(query_heads_expr)
    head_dim = int(head_dim_expr)
    key_heads = int(key_heads_expr)
    value_heads = int(value_heads_expr)
    analyzer = arith.Analyzer()
    if (
        dtype not in ("float16", "float32")
        or key.ty.dtype != dtype
        or value.ty.dtype != dtype
        or not _can_prove_equal(analyzer, batch, key_batch)
        or not _can_prove_equal(analyzer, batch, value_batch)
        or not _can_prove_equal(analyzer, key_length, value_length)
        or key_heads != value_heads
        or not _can_prove_equal(analyzer, key_dim, head_dim_expr)
        or not _can_prove_equal(analyzer, value_dim, head_dim_expr)
        or query_heads % key_heads != 0
    ):
        return None

    scale = float(call.attrs.scale) if call.attrs.scale is not None else head_dim**-0.5
    score_buffer_bytes = None
    if all(isinstance(dim, tirx.IntImm) for dim in (batch, query_length, key_length)):
        score_buffer_bytes = (
            int(batch)
            * query_heads
            * int(query_length)
            * int(key_length)
            * ((DataType(dtype).bits + 7) // 8)
        )
    return AttentionDispatchInfo(
        key_heads,
        query_heads,
        head_dim,
        dtype,
        scale,
        score_buffer_bytes,
    )


@expr_functor.mutator
class AttentionDispatcher(BackendDispatcher):
    """Replace eligible attention calls whose score buffer exceeds a byte limit."""

    def __init__(self, mod: IRModule, max_score_buffer_bytes: int):
        super().__init__(mod)
        self.max_score_buffer_bytes = max_score_buffer_bytes
        self.kernels = {}

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        if not isinstance(call.op, Op) or call.op.name != "relax.nn.attention":
            return super().visit_call_(call)
        target = self._get_target(call.ty)
        if target.kind.name != "webgpu":
            return super().visit_call_(call)
        info = get_attention_dispatch_info(call)
        if info is None:
            return super().visit_call_(call)
        if (
            info.score_buffer_bytes is not None
            and info.score_buffer_bytes <= self.max_score_buffer_bytes
        ):
            return super().visit_call_(call)

        query, key, value = call.args
        kernel_key = (
            info.key_heads,
            info.query_heads,
            info.head_dim,
            info.dtype,
            info.scale,
            str(target),
        )
        global_var = self.kernels.get(kernel_key)
        if global_var is None:
            from tvm.relax.frontend.nn.llm.kv_cache import (  # pylint: disable=import-outside-toplevel
                _attention_sequence_prefill,
            )

            kernel = _attention_sequence_prefill(
                info.key_heads,
                info.query_heads,
                info.head_dim,
                info.dtype,
                target,
                causal=0,
                sm_scale=info.scale,
            )
            global_var = self.builder_.add_func(kernel, "online_attention")
            self.kernels[kernel_key] = global_var

        lse_type = relax.TensorType(
            (query.ty.shape[0], query.ty.shape[1], query.ty.shape[2]),
            info.dtype,
            vdevice=call.ty.vdevice,
        )
        result = relax.call_tir(
            global_var,
            [query, key, value],
            out_ty=[call.ty, lse_type],
        )
        return relax.TupleGetItem(result, 0)


def DispatchAttention(max_score_buffer_bytes: int = 128 * 1024 * 1024):
    """Create a pass that bounds the materialized attention score buffer."""

    if max_score_buffer_bytes < 0:
        raise ValueError("max_score_buffer_bytes must be nonnegative")

    @module_pass(opt_level=0, name="DispatchAttention")
    def _dispatch_attention(mod: IRModule, _ctx: PassContext) -> IRModule:
        dispatcher = AttentionDispatcher(mod, max_score_buffer_bytes)
        for global_var, function in mod.functions_items():
            if isinstance(function, relax.Function):
                function = dispatcher.visit_expr(function)
                dispatcher.builder_.update_func(global_var, function)
        return dispatcher.builder_.finalize()

    return _dispatch_attention
