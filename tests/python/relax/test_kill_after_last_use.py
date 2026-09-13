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

import numpy as np
import tvm_ffi
from tvm_ffi.testing import object_use_count

import tvm
import tvm.relax
import tvm.testing
from tvm.relax.transform import KillAfterLastUse
from tvm.script import ir as I
from tvm.script import relax as R


def test_basic():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _dummy = R.call_packed("add_tensors", [x, y], ty_args=(R.Tuple,))
            z = R.add(x, y)
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _ = R.memory.kill_storage(storage)
            _dummy = R.call_packed("add_tensors", [x, y], ty_args=(R.Tuple,))
            z = R.add(x, y)
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_track_usage_across_trivial_rebindings():
    """To work around VM de-duplication of register usage"""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = x
            z = R.add(y, R.const(1, "float32"))
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            z = R.add(x, R.const(1, "float32"))
            _ = R.memory.kill_tensor(x)
            return z

    After = KillAfterLastUse()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_track_usage_across_trivial_rebindings_in_match_cast():
    """To work around VM de-duplication of register usage"""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = R.match_cast(x, R.Tensor([16, 32]))
            z = R.add(y, R.const(1, "float32"))
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = R.match_cast(x, R.Tensor([16, 32]))
            _ = R.memory.kill_tensor(x)
            z = R.add(y, R.const(1, "float32"))
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_release_values_after_conditional():
    """A branch must not discard the enclosing block's last-use analysis."""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor([16], "float32"), flag: R.Prim("bool")):
            captured = R.add(x, x)
            if flag:
                local = R.multiply(captured, x)
                branch = R.add(local, x)
            else:
                branch = R.add(captured, x)
            tail = R.add(branch, x)
            output = R.multiply(tail, x)
            return output

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor([16], "float32"), flag: R.Prim("bool")):
            captured = R.add(x, x)
            if flag:
                local = R.multiply(captured, x)
                result = R.add(local, x)
                _ = R.memory.kill_tensor(local)
                branch = result
            else:
                branch = R.add(captured, x)
            _ = R.memory.kill_tensor(captured)
            tail = R.add(branch, x)
            _ = R.memory.kill_tensor(branch)
            output = R.multiply(tail, x)
            _ = R.memory.kill_tensor(tail)
            return output

    tvm.ir.assert_structural_equal(Expected, KillAfterLastUse()(Before))


def test_nested_conditionals_preserve_live_aliases():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(x: R.Tensor([16], "float32"), flag: R.Prim("bool")):
            captured = R.add(x, x)
            if flag:
                if flag:
                    nested = R.multiply(captured, x)
                else:
                    nested = R.add(captured, x)
                branch = R.add(nested, captured)
            else:
                branch = captured
            tail = R.add(branch, captured)
            output = R.multiply(tail, x)
            return output

    # Reuse the same VM in both directions; an early kill may otherwise hide
    # behind unrecycled allocator contents on the first invocation.
    vm = tvm.relax.VirtualMachine(tvm.relax.build(Module, target="llvm"), tvm.cpu())
    values = np.arange(16, dtype="float32") / 16
    for flag in (True, False, True, False):
        actual = vm["main"](tvm.runtime.tensor(values), flag).numpy()
        expected = (2 * values**2 + 4 * values) * values if flag else 4 * values**2
        np.testing.assert_allclose(actual, expected)


def test_vm_branch_result_does_not_keep_a_hidden_reference():
    retained = []

    @tvm_ffi.register_global_func("test.branch_lifetime.alloc", override=True)
    def allocate():
        tensor = tvm.runtime.tensor(np.ones(16, dtype="float32"))
        retained.append(tensor)
        return tensor

    @tvm_ffi.register_global_func("test.branch_lifetime.consume", override=True)
    def consume(tensor):
        np.testing.assert_array_equal(tensor.numpy(), np.ones(16, dtype="float32"))
        return 0

    @tvm_ffi.register_global_func("test.branch_lifetime.count", override=True)
    def count():
        return object_use_count(retained[-1])

    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(flag: R.Prim("bool")):
            if flag:
                fresh = R.call_packed(
                    "test.branch_lifetime.alloc", ty_args=R.Tensor([16], "float32")
                )
                branch = (fresh,)
            else:
                fresh = R.call_packed(
                    "test.branch_lifetime.alloc", ty_args=R.Tensor([16], "float32")
                )
                branch = (fresh,)
            tensor = branch[0]
            _used = R.call_packed("test.branch_lifetime.consume", tensor, ty_args=R.Prim("int64"))
            count = R.call_packed("test.branch_lifetime.count", ty_args=R.Prim("int64"))
            return count

    vm = tvm.relax.VirtualMachine(tvm.relax.build(Module, target="llvm"), tvm.cpu())
    for flag in (True, False, True, False):
        # Only the retained Python handle should own the allocation once the
        # merged tuple and its extracted tensor pass their last use.
        assert vm["main"](flag) == 1


def test_vm_branch_returning_a_parameter_preserves_later_uses():
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor([16], "float32"), flag: R.Prim("bool")):
            if flag:
                branch = x
            else:
                branch = x
            output = R.add(branch, x)
            return output

    vm = tvm.relax.VirtualMachine(tvm.relax.build(Module, target="llvm"), tvm.cpu())
    values = np.arange(16, dtype="float32")
    for flag in (True, False):
        np.testing.assert_array_equal(
            vm["main"](tvm.runtime.tensor(values), flag).numpy(), 2 * values
        )


if __name__ == "__main__":
    tvm.testing.main()
