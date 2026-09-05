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
"""Exercise the pooled allocator itself with small CPU allocations and mock OOM."""

import subprocess
import sys
from pathlib import Path

import pytest
from tvm_ffi import libinfo as ffi_libinfo

from tvm import libinfo
from tvm.support import cc


@pytest.mark.parametrize("manager_api", [False, True])
def test_pooled_allocator_free_cache(tmp_path, manager_api):
    if sys.platform == "win32":
        pytest.skip("This standalone C++ runtime harness uses Unix linker options")
    compiler = cc.get_cc()
    if compiler is None:
        pytest.skip("A C++ compiler is required for the allocator runtime harness")
    source = Path(__file__).with_name("pooled_allocator_cache_harness.cc")
    root = Path(__file__).resolve().parents[3]
    runtime = Path(libinfo.find_libtvm_runtime())
    ffi = Path(ffi_libinfo.find_libtvm_ffi())
    options = [
        "-std=c++17",
        "-pthread",
        f"-I{root}",
        f"-I{libinfo.find_include_path()}",
        f"-I{ffi_libinfo.find_include_path()}",
        f"-I{ffi_libinfo.find_dlpack_include_path()}",
        str(runtime),
        str(ffi),
        f"-Wl,-rpath,{runtime.parent}",
        f"-Wl,-rpath,{ffi.parent}",
    ]
    if manager_api:
        options.append("-DTVM_TEST_POOL_MANAGER")
    executable = tmp_path / "pooled_allocator_cache"
    cc.create_executable(str(executable), [str(source)], options=options, cc=compiler)
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert "pooled allocator cache checks passed" in result.stdout
