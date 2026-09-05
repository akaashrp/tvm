/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ffi/function.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "src/runtime/memory/pooled_allocator.h"

using tvm::Device;
using tvm::runtime::memory::Buffer;
using tvm::runtime::memory::PooledAllocator;

static void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class MockPool : public PooledAllocator {
 public:
  MockPool() : PooledAllocator(16) {}
  ~MockPool() override { Clear(); }
  size_t frees{0};
  bool fail_next{false};
  std::unordered_map<void*, size_t> allocated;

  Buffer New(size_t size) { return Alloc({kDLCPU, 0}, size, 16, {kDLUInt, 8, 1}); }
  size_t Cached() {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    return cached_memory_;
  }

 protected:
  void* DeviceAllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType dtype) override {
    if (fail_next) {
      fail_next = false;
      TVM_FFI_THROW(InternalError) << "intentional allocation failure";
    }
    void* data = std::malloc(size ? size : 1);
    Require(data != nullptr, "mock allocation failed");
    allocated[data] = size;
    return data;
  }
  void DeviceFreeDataSpace(Device dev, void* data) override {
    Require(allocated.erase(data) == 1, "unknown or double-freed allocation");
    ++frees;
    std::free(data);
  }
};

static void TestDefaultAndReuse() {
  MockPool pool;
  auto a = pool.New(17);  // Page-rounded accounting uses 32 bytes.
  std::memset(a.data, 7, a.size);
  pool.Free(a);
  Require(pool.UsedMemory() == 32 && pool.Cached() == 32, "default cache changed");
  auto reused = pool.New(17);
  Require(reused.data == a.data && pool.frees == 0, "cached block was not reused");
  Require(pool.Cached() == 0 && pool.UsedMemory() == 32, "reuse accounting failed");
  pool.Free(reused);
  pool.Clear();
  Require(pool.UsedMemory() == 0 && pool.allocated.empty(), "clear leaked memory");
}

static void TestLimitAndShrink() {
  MockPool pool;
  auto live = pool.New(64);
  std::memset(live.data, 19, live.size);
  auto a = pool.New(16), b = pool.New(32), c = pool.New(48);
  pool.Free(a);
  pool.Free(b);
  pool.Free(c);
  pool.SetMaxCachedBytes(48);
  Require(pool.Cached() == 48 && pool.UsedMemory() == 112, "shrink limit failed");
  Require(pool.allocated.count(c.data) == 0, "largest free block was not evicted");
  pool.SetMaxCachedBytes(16);
  Require(pool.Cached() == 16 && pool.UsedMemory() == 80, "second shrink failed");
  Require(pool.allocated.count(a.data) == 1, "smaller reusable block was lost");
  auto reused = pool.New(16);
  Require(reused.data == a.data, "reuse under budget failed");
  pool.Free(reused);
  auto oversized = pool.New(128);
  pool.Free(oversized);
  Require(pool.Cached() == 16 && pool.UsedMemory() == 80, "oversized free retained");
  pool.SetMaxCachedBytes(0);
  Require(pool.Cached() == 0 && pool.UsedMemory() == 64, "zero ceiling lost live bytes");
  Require(static_cast<unsigned char*>(live.data)[63] == 19, "live allocation corrupted");
  pool.Free(live);
  auto zero = pool.New(0);
  pool.Free(zero);
  Require(pool.UsedMemory() == 0 && pool.allocated.empty(), "zero limit retained buffers");
}

static void TestClearAndAllocationFailure() {
  MockPool pool;
  auto live = pool.New(64), cached = pool.New(32);
  pool.Free(cached);
  pool.Clear();
  Require(pool.UsedMemory() == 64, "Clear reset accounting with a live buffer");
  cached = pool.New(32);
  pool.Free(cached);
  pool.fail_next = true;
  auto next = pool.New(48);
  Require(pool.Cached() == 0 && pool.UsedMemory() == 112, "allocation retry lost live accounting");
  Require(pool.allocated.count(live.data) == 1, "allocation retry freed live memory");
  pool.Free(next);
  pool.Free(live);
  pool.Clear();
  Require(pool.UsedMemory() == 0 && pool.allocated.empty(), "retry cleanup leaked");
}

static void TestConcurrentFreeCache() {
  MockPool pool;
  pool.SetMaxCachedBytes(48);
  auto work = [&pool]() {
    for (size_t i = 1; i <= 100; ++i) {
      auto buffer = pool.New((i % 7 + 1) * 16);
      std::memset(buffer.data, 23, buffer.size);
      Require(static_cast<unsigned char*>(buffer.data)[buffer.size - 1] == 23,
              "live memory was changed");
      pool.Free(buffer);
      Require(pool.Cached() <= 48, "free cache exceeded its byte ceiling");
    }
  };
  std::thread first(work), second(work);
  first.join();
  second.join();
  Require(pool.UsedMemory() == pool.Cached(), "concurrent accounting leaked live bytes");
  pool.Clear();
  Require(pool.UsedMemory() == 0 && pool.allocated.empty(), "concurrent cleanup failed");
}

#ifdef TVM_TEST_POOL_MANAGER
static void TestManagerAPI() {
  using tvm::runtime::memory::kPooled;
  using tvm::runtime::memory::MemoryManager;
  auto function =
      tvm::ffi::Function::GetGlobalRequired("vm.builtin.memory_manager.set_pool_max_cached_bytes");
  function(static_cast<int>(kDLCPU), 0, 0);
  auto* zero = MemoryManager::GetOrCreateAllocator({kDLCPU, 0}, kPooled);
  auto a = zero->Alloc({kDLCPU, 0}, 16, 64, {kDLUInt, 8, 1});
  zero->Free(a);
  Require(zero->UsedMemory() == 0, "packed API zero limit was not enforced");
  function(static_cast<int>(kDLCPU), 1, 4096);
  auto* other = MemoryManager::GetOrCreateAllocator({kDLCPU, 1}, kPooled);
  auto b = other->Alloc({kDLCPU, 1}, 16, 64, {kDLUInt, 8, 1});
  other->Free(b);
  Require(other->UsedMemory() == 4096 && zero->UsedMemory() == 0,
          "cache limits are not per device");
  function(static_cast<int>(kDLCPU), 1, 0);
  Require(other->UsedMemory() == 0, "packed API shrink did not evict cached buffers");
  bool rejected = false;
  try {
    function(static_cast<int>(kDLCPU), 0, -2);
  } catch (const tvm::ffi::Error&) {
    rejected = true;
  }
  Require(rejected, "packed API accepted a negative limit");
  function(static_cast<int>(kDLCPU), 0, -1);
  a = zero->Alloc({kDLCPU, 0}, 16, 64, {kDLUInt, 8, 1});
  zero->Free(a);
  Require(zero->UsedMemory() == 4096, "unlimited cache restoration failed");
  zero->Clear();
}
#endif

int main() {
  try {
    TestDefaultAndReuse();
    TestLimitAndShrink();
    TestClearAndAllocationFailure();
    TestConcurrentFreeCache();
#ifdef TVM_TEST_POOL_MANAGER
    TestManagerAPI();
#endif
    std::cout << "pooled allocator cache checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
