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
"""Symbolic reductions used by materialized Relax attention."""

import pytest

import tvm
from tvm import te, tirx, topi
from tvm.relax.transform.legalize_ops.nn import _te_attention


@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_batch_matmul_proves_regrouped_symbolic_reductions(transpose_a, transpose_b):
    frames = tirx.Var("frames", "int64")
    left = 60 * frames
    right = 6 * (frames * 10)
    a = te.placeholder((1, left, 2) if transpose_a else (1, 2, left), name="a")
    b = te.placeholder((1, 3, right) if transpose_b else (1, right, 3), name="b")
    result = topi.nn.batch_matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    assert tuple(int(value) for value in result.shape) == (1, 2, 3)
    assert tvm.arith.Analyzer().can_prove_equal(result.op.reduce_axis[0].dom.extent, left)
    te.create_prim_func([a, b, result])


@pytest.mark.parametrize("mismatch", ["different_symbol", "offset", "scale", "static"])
def test_batch_matmul_rejects_unproven_reduction_equality(mismatch):
    frames = tirx.Var("frames", "int64")
    left, right = {
        "different_symbol": (frames, tirx.Var("other", "int64")),
        "offset": (60 * frames, 60 * frames + 1),
        "scale": (60 * frames, 61 * frames),
        "static": (3, 4),
    }[mismatch]
    a = te.placeholder((1, 2, left), name="a")
    b = te.placeholder((1, right, 3), name="b")
    with pytest.raises(AssertionError, match="shapes of x and y are inconsistent"):
        topi.nn.batch_matmul(a, b, transpose_b=False)


def test_materialized_attention_accepts_derived_sequence_length():
    frames = tirx.Var("frames", "int64")
    # Softmax simplifies the reduction extent to a different expression tree
    # than the reshaped value tensor used by attention's second matmul.
    shape = (1, 60 * frames, 24, 128)
    q, k, v = (te.placeholder(shape, dtype="float16", name=name) for name in ("q", "k", "v"))
    output = _te_attention(q, k, v, None, None, None)
    for actual, expected in zip(output.shape, shape):
        assert tvm.arith.Analyzer().can_prove_equal(actual, expected)
    te.create_prim_func([q, k, v, output])
