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

import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.ir import Op
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class AttentionModule:
    @R.function
    def main(
        q: R.Tensor((1, 2048, 12, 128), "float32"),
        k: R.Tensor((1, 2048, 12, 128), "float32"),
        v: R.Tensor((1, 2048, 12, 128), "float32"),
    ) -> R.Tensor((1, 2048, 12, 128), "float32"):
        return R.nn.attention(q, k, v)


@I.ir_module
class RepeatedAttentionModule:
    @R.function
    def main(
        q: R.Tensor((1, 2048, 12, 128), "float32"),
        k: R.Tensor((1, 2048, 12, 128), "float32"),
        v: R.Tensor((1, 2048, 12, 128), "float32"),
    ) -> R.Tensor((1, 2048, 12, 128), "float32"):
        with R.dataflow():
            first = R.nn.attention(q, k, v)
            second = R.nn.attention(first, k, v)
            R.output(second)
        return second


@I.ir_module
class Float64AttentionModule:
    @R.function
    def main(
        q: R.Tensor((1, 2048, 12, 128), "float64"),
        k: R.Tensor((1, 2048, 12, 128), "float64"),
        v: R.Tensor((1, 2048, 12, 128), "float64"),
    ) -> R.Tensor((1, 2048, 12, 128), "float64"):
        return R.nn.attention(q, k, v)


@I.ir_module
class Float16AttentionModule:
    @R.function
    def main(
        q: R.Tensor((1, 128, 12, 128), "float16"),
        k: R.Tensor((1, 128, 12, 128), "float16"),
        v: R.Tensor((1, 128, 12, 128), "float16"),
    ) -> R.Tensor((1, 128, 12, 128), "float16"):
        return R.nn.attention(q, k, v)


@I.ir_module
class DynamicAttentionModule:
    @R.function
    def main(
        q: R.Tensor((1, "query_length", 12, 128), "float32"),
        k: R.Tensor((1, "key_length", 12, 128), "float32"),
        v: R.Tensor((1, "key_length", 12, 128), "float32"),
    ) -> R.Tensor((1, "query_length", 12, 128), "float32"):
        return R.nn.attention(q, k, v)


def _webgpu_target(shared_memory_bytes=16384):
    return tvm.target.Target(
        {
            "kind": "webgpu",
            "max_shared_memory_per_block": shared_memory_bytes,
            "host": {
                "kind": "llvm",
                "mtriple": "wasm32-unknown-unknown-wasm",
            },
        }
    )


def _relax_ops(mod):
    names = []

    def collect(expr):
        if isinstance(expr, relax.Call) and isinstance(expr.op, Op):
            names.append(expr.op.name)

    relax.analysis.post_order_visit(mod["main"], collect)
    return names


def test_dispatches_attention_above_score_buffer_limit():
    with _webgpu_target():
        after = relax.backend.DispatchAttention()(AttentionModule)

    assert "relax.nn.attention" not in _relax_ops(after)
    assert "relax.call_tir" in _relax_ops(after)
    assert (
        sum(isinstance(function, tvm.tirx.PrimFunc) for function in after.functions.values()) == 1
    )


def test_keeps_attention_at_score_buffer_limit():
    score_buffer_bytes = 1 * 12 * 2048 * 2048 * 4
    with _webgpu_target():
        after = relax.backend.DispatchAttention(score_buffer_bytes)(AttentionModule)

    assert "relax.nn.attention" in _relax_ops(after)
    assert len(after.functions) == 1


def test_reuses_kernel_for_matching_attention_calls():
    with _webgpu_target():
        after = relax.backend.DispatchAttention()(RepeatedAttentionModule)

    assert "relax.nn.attention" not in _relax_ops(after)
    assert (
        sum(isinstance(function, tvm.tirx.PrimFunc) for function in after.functions.values()) == 1
    )


def test_dispatches_attention_with_symbolic_sequence_lengths():
    with _webgpu_target():
        after = relax.backend.DispatchAttention()(DynamicAttentionModule)

    assert "relax.nn.attention" not in _relax_ops(after)
    assert "relax.call_tir" in _relax_ops(after)


@pytest.mark.parametrize("limit,materialized", [(6144, True), (6143, False)])
def test_attention_dispatch_uses_symbolic_bounds(limit, materialized):
    function = DynamicAttentionModule["main"].with_attr(
        "tir_var_upper_bound", {"query_length": 8, "key_length": 16}
    )
    # 1 batch * 12 heads * 8 queries * 16 keys * 4 bytes = 6144 bytes.
    with _webgpu_target():
        after = relax.backend.DispatchAttention(limit)(tvm.IRModule({"main": function}))
    assert ("relax.nn.attention" in _relax_ops(after)) == materialized


def test_small_bounded_batch_attention_with_wide_heads_stays_materialized():
    @I.ir_module
    class Module:
        @R.function
        def main(
            q: R.Tensor(("batch", 240, 1, 1024), "float32"),
            k: R.Tensor(("batch", 240, 1, 1024), "float32"),
            v: R.Tensor(("batch", 240, 1, 1024), "float32"),
        ) -> R.Tensor(("batch", 240, 1, 1024), "float32"):
            R.func_attr({"tir_var_lower_bound": {"batch": 1}, "tir_var_upper_bound": {"batch": 9}})
            return R.nn.attention(q, k, v)

    with _webgpu_target(32768):
        after = relax.backend.DispatchAttention(9 * 240 * 240 * 4)(Module)
    assert "relax.nn.attention" in _relax_ops(after)


def test_partial_symbolic_bounds_keep_online_attention():
    function = DynamicAttentionModule["main"].with_attr("tir_var_upper_bound", {"query_length": 8})
    with _webgpu_target():
        after = relax.backend.DispatchAttention()(tvm.IRModule({"main": function}))
    assert "relax.nn.attention" not in _relax_ops(after)


def test_float16_attention_auxiliary_output_matches_kernel_signature():
    with _webgpu_target():
        after = relax.backend.DispatchAttention(0)(Float16AttentionModule)

    calls = []
    relax.analysis.post_order_visit(
        after["main"],
        lambda expr: calls.append(expr) if isinstance(expr, relax.Call) else None,
    )
    attention_call = next(call for call in calls if call.op.name == "relax.call_tir")
    assert attention_call.ty.fields[1].dtype == "float16"


def test_keeps_unsupported_dtype_and_non_webgpu_target():
    with _webgpu_target():
        unsupported_dtype = relax.backend.DispatchAttention(0)(Float64AttentionModule)
    with tvm.target.Target("llvm"):
        cpu = relax.backend.DispatchAttention(0)(AttentionModule)

    assert "relax.nn.attention" in _relax_ops(unsupported_dtype)
    assert "relax.nn.attention" in _relax_ops(cpu)


def test_rejects_negative_score_buffer_limit():
    with pytest.raises(ValueError, match="must be nonnegative"):
        relax.backend.DispatchAttention(-1)


@pytest.mark.parametrize("shared_memory_bytes", [16384, 32768])
def test_webgpu_head_dim_128_kernel_fits_shared_memory(shared_memory_bytes):
    with _webgpu_target(shared_memory_bytes) as target:
        after = relax.backend.DispatchAttention(0)(AttentionModule)
        executable = relax.build(after, target=target)

    assert executable is not None


if __name__ == "__main__":
    tvm.testing.main()
