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
const { detectGPUDevice, WebGPUContext } = require("../../src/webgpu");

global.GPUBufferUsage = {
  MAP_READ: 1 << 0,
  COPY_DST: 1 << 1,
  COPY_SRC: 1 << 2,
  STORAGE: 1 << 3,
  UNIFORM: 1 << 4,
  QUERY_RESOLVE: 1 << 5,
};
global.GPUMapMode = {
  READ: 1,
};
global.GPUShaderStage = {
  COMPUTE: 1,
};

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createMockDevice({
  mapAsync = () => Promise.resolve(),
  onSubmittedWorkDone = () => Promise.resolve(),
  timestampQuery = false,
} = {}) {
  const events = [];
  const encoders = [];
  const buffers = [];
  const querySets = [];
  const invalidTimestampSubmissions = [];

  const queue = {
    submit: jest.fn((commandBuffers) => {
      for (const commandBuffer of commandBuffers) {
        for (const querySet of commandBuffer.timestampQuerySets || []) {
          if (querySet.destroyed) {
            invalidTimestampSubmissions.push(querySet);
          }
        }
      }
      events.push("submit");
    }),
    writeBuffer: jest.fn(() => events.push("writeBuffer")),
    onSubmittedWorkDone: jest.fn(onSubmittedWorkDone),
  };

  const device = {
    features: new Set(timestampQuery ? ["timestamp-query"] : []),
    limits: {
      minUniformBufferOffsetAlignment: 256,
    },
    queue,
    createCommandEncoder: jest.fn(() => {
      const commands = [];
      const timestampQuerySets = [];
      const encoderId = encoders.length;
      const encoder = {
        commands,
        beginComputePass: jest.fn((descriptor) => ({
          setPipeline: jest.fn(),
          setBindGroup: jest.fn(),
          dispatchWorkgroups: jest.fn(() => {
            commands.push("compute");
            events.push("compute");
            const timestampWrites = descriptor && descriptor.timestampWrites;
            const querySet = timestampWrites && timestampWrites.querySet;
            if (querySet !== undefined) {
              timestampQuerySets.push(querySet);
            }
          }),
          end: jest.fn(),
        })),
        resolveQuerySet: jest.fn(() => {
          commands.push("resolve");
          events.push("resolve");
        }),
        copyBufferToBuffer: jest.fn(() => {
          commands.push("copy");
          events.push("copy");
        }),
        finish: jest.fn(() => {
          const commandBuffer = { encoderId, commands: commands.slice() };
          if (timestampQuerySets.length > 0) {
            commandBuffer.timestampQuerySets = timestampQuerySets.slice();
          }
          events.push("finish");
          return commandBuffer;
        }),
      };
      encoders.push(encoder);
      return encoder;
    }),
    createBuffer: jest.fn((descriptor) => {
      const mappedData = new ArrayBuffer(descriptor.size);
      const buffer = {
        size: descriptor.size,
        destroy: jest.fn(() => events.push("destroy")),
        mapAsync: jest.fn(mapAsync),
        getMappedRange: jest.fn(() => mappedData),
        unmap: jest.fn(),
      };
      buffers.push(buffer);
      return buffer;
    }),
    createQuerySet: jest.fn((descriptor) => {
      const querySet = {
        descriptor,
        destroyed: false,
        destroy: jest.fn(() => {
          querySet.destroyed = true;
          events.push("destroyQuerySet");
        }),
      };
      querySets.push(querySet);
      return querySet;
    }),
    createBindGroupLayout: jest.fn(() => ({})),
    createPipelineLayout: jest.fn(() => ({})),
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({})),
    createBindGroup: jest.fn((descriptor) => descriptor),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn(() => Promise.resolve(null)),
    destroy: jest.fn(),
  };

  return {
    device,
    queue,
    events,
    encoders,
    buffers,
    querySets,
    invalidTimestampSubmissions,
  };
}

function createContext(deviceOptions) {
  const gpu = createMockDevice(deviceOptions);
  const memory = {
    loadRawBytes: jest.fn(),
    viewRawBytes: jest.fn(),
    storeRawBytes: jest.fn(),
  };
  const context = new WebGPUContext(memory, gpu.device);
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  return {
    ...gpu,
    context,
    memory,
    source: allocate(64),
    destination: allocate(64),
  };
}

test.each([[0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 8], [16, 16]])(
  "storage allocation of %i bytes has a valid %i-byte binding",
  (requested, expected) => {
    const { device, context } = createContext();
    const allocate = context.getDeviceAPI("deviceAllocDataSpace");
    const free = context.getDeviceAPI("deviceFreeDataSpace");
    const pointer = allocate(requested);
    expect(device.createBuffer.mock.lastCall[0].size).toBe(expected);
    const buffer = device.createBuffer.mock.results.at(-1).value;
    free(pointer);
    expect(buffer.destroy).toHaveBeenCalledTimes(1);
  }
);

test("device detection requests both workgroup invocation and X-axis limits", async () => {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const requestDevice = jest.fn(async () => ({ id: "device" }));
  const adapter = {
    features: new Set(),
    info: {},
    limits: {
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 30,
      maxComputeWorkgroupStorageSize: 32 << 10,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeInvocationsPerWorkgroup: 1024,
      maxComputeWorkgroupSizeX: 512,
    },
    requestDevice,
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        requestAdapter: jest.fn(async () => adapter),
      },
    },
  });

  await detectGPUDevice();

  expect(requestDevice).toHaveBeenCalledTimes(1);
  expect(requestDevice.mock.calls[0][0].requiredLimits).toMatchObject({
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupSizeX: 512,
  });
  if (originalNavigator === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(globalThis, "navigator", originalNavigator);
  }
});

test("compute dispatches and GPU copies share one submission", async () => {
  const { context, device, queue, encoders, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );

  shader();
  copyWithinGPU(source, 0, destination, 0, 16);
  copyWithinGPU(destination, 16, source, 32, 16);

  expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
  expect(queue.submit).not.toHaveBeenCalled();
  expect(encoders[0].commands).toEqual(["compute", "copy", "copy"]);

  await context.sync();

  expect(encoders[0].finish).toHaveBeenCalledTimes(1);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.submit.mock.calls[0][0]).toEqual([
    {
      encoderId: 0,
      commands: encoders[0].commands,
    },
  ]);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  await context.sync();
  expect(queue.submit).toHaveBeenCalledTimes(1);
});

test("pending WebGPU commands can be submitted without waiting", () => {
  const { context, queue } = createContext();
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );
  shader();
  expect(context.submitPendingCommands()).toBe(true);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();

  expect(context.submitPendingCommands()).toBe(false);
  expect(queue.submit).toHaveBeenCalledTimes(1);
});

test("timestamp capacity growth submits before replacing the query set", async () => {
  global.__WEBLLM_TRACE_RUNTIME_STATE__ = {
    enabled: true,
    level: "major",
    devtools: "off",
    ctx: "main",
    enable_gpu_timestamps: true,
  };
  const {
    context,
    queue,
    querySets,
    invalidTimestampSubmissions,
  } = createContext({ timestampQuery: true });
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );

  for (let i = 0; i < 513; ++i) {
    shader();
  }
  await context.sync();
  delete global.__WEBLLM_TRACE_RUNTIME_STATE__;

  expect(querySets.map((querySet) => querySet.descriptor.count)).toEqual([
    1024,
    2048,
  ]);
  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(invalidTimestampSubmissions).toHaveLength(0);
});

test("compute dispatches pack uniforms into a reusable aligned arena", async () => {
  const { context, device, queue, buffers } = createContext();
  const shader = context.createShader(
    {
      name: "main",
      arg_types: ["int32"],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );

  shader(11);
  shader(22);
  shader(33);

  expect(device.createBuffer).toHaveBeenCalledTimes(3);
  expect(buffers[2].size).toBe(1024 * 1024);
  expect(queue.writeBuffer.mock.calls.slice(-3).map((call) => call[1])).toEqual([
    0,
    256,
    512,
  ]);
  expect(
    device.createBindGroup.mock.calls.map(
      (call) => call[0].entries.at(-1).resource.offset,
    ),
  ).toEqual([0, 256, 512]);

  await context.sync();
  shader(44);
  shader(55);

  expect(device.createBuffer).toHaveBeenCalledTimes(3);
  expect(queue.writeBuffer.mock.calls.slice(-2).map((call) => call[1])).toEqual([
    0,
    256,
  ]);
});

test("sampled-token ring copy shares the pending compute submission", async () => {
  const { context, device, queue, encoders, source } = createContext();
  const shader = context.createShader(
    {
      name: "main",
      arg_types: [],
      launch_param_tags: [],
    },
    "@compute @workgroup_size(1) fn main() {}"
  );
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 2,
    maxTokensPerBatch: 1,
  });

  shader();
  const batchSeq = context.submitSampledTokenReadbackRing(ringId, source, 0, 1);

  expect(batchSeq).toBe(1);
  expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
  expect(encoders[0].commands).toEqual(["compute", "copy"]);
  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(queue.submit.mock.calls[0][0]).toEqual([
    {
      encoderId: 0,
      commands: encoders[0].commands,
    },
  ]);

  const batches = await context.waitSampledTokenReadbackRing(ringId);
  expect(batches).toHaveLength(1);
  expect(batches[0].batchSeq).toBe(1);
  expect(batches[0].tokenCount).toBe(1);
  context.disposeSampledTokenReadbackRing(ringId);
});

test.each([1, 2, 4, 8, 16])(
  "sampled-token ring concatenates K=%i token buffers in one submission",
  async (batchSize) => {
    const { context, device, queue, encoders, buffers } = createContext();
    const allocate = context.getDeviceAPI("deviceAllocDataSpace");
    const tokenSources = Array.from({ length: batchSize }, () => allocate(64));
    const ringId = context.createSampledTokenReadbackRing({
      slotCount: 2,
      maxTokensPerBatch: batchSize,
    });

    const batchSeq = context.submitSampledTokenReadbackRingTokens(
      ringId,
      tokenSources,
    );

    expect(batchSeq).toBe(1);
    expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
    expect(encoders[0].commands).toEqual(Array(batchSize).fill("copy"));
    expect(encoders[0].copyBufferToBuffer.mock.calls).toEqual(
      Array.from({ length: batchSize }, (_, index) => [
        buffers[2 + index],
        0,
        buffers[2 + batchSize],
        index * 4,
        4,
      ]),
    );
    expect(queue.submit).toHaveBeenCalledTimes(1);

    const batches = await context.waitSampledTokenReadbackRing(ringId);
    expect(batches).toHaveLength(1);
    expect(batches[0].tokenCount).toBe(batchSize);
    expect(Array.from(batches[0].tokens)).toEqual(Array(batchSize).fill(0));
    context.disposeSampledTokenReadbackRing(ringId);
  },
);

test("sampled-token ring rejects an empty token-buffer batch", () => {
  const { context } = createContext();
  const ringId = context.createSampledTokenReadbackRing({
    slotCount: 2,
    maxTokensPerBatch: 3,
  });

  expect(() =>
    context.submitSampledTokenReadbackRingTokens(ringId, [])
  ).toThrow("At least one sampled-token source is required.");
  context.disposeSampledTokenReadbackRing(ringId);
});

test("a host write is staged after pending GPU copies", async () => {
  const {
    context,
    queue,
    events,
    encoders,
    buffers,
    source,
    destination,
  } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const rawBytes = new Uint8Array([1, 2, 3, 4]);

  copyWithinGPU(source, 0, destination, 0, rawBytes.length);
  expect(queue.submit).not.toHaveBeenCalled();

  context.copyRawBytesToBuffer(rawBytes, destination, 4, rawBytes.length);

  expect(queue.submit).not.toHaveBeenCalled();
  expect(queue.writeBuffer).toHaveBeenCalledTimes(1);
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    buffers[2],
    0,
    rawBytes,
    0,
    rawBytes.length,
  );
  expect(encoders[0].copyBufferToBuffer).toHaveBeenLastCalledWith(
    buffers[2],
    0,
    buffers[1],
    4,
    rawBytes.length,
  );
  expect(encoders[0].commands).toEqual(["copy", "copy"]);
  expect(events).toEqual(["copy", "writeBuffer", "copy"]);

  await context.sync();

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(events).toEqual([
    "copy",
    "writeBuffer",
    "copy",
    "finish",
    "submit",
  ]);
});

test("a host write targets its destination directly without pending commands", () => {
  const { context, queue, encoders, buffers, destination } = createContext();
  const rawBytes = new Uint8Array([1, 2, 3, 4]);

  context.copyRawBytesToBuffer(rawBytes, destination, 4, rawBytes.length);

  expect(encoders).toHaveLength(0);
  expect(queue.submit).not.toHaveBeenCalled();
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    buffers[1],
    4,
    rawBytes,
    0,
    rawBytes.length,
  );
});

test("a large host write submits pending commands instead of retaining staging", () => {
  const {
    context,
    queue,
    events,
    encoders,
    buffers,
    source,
    destination,
  } = createContext();
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const rawBytes = new Uint8Array(64 * 1024 + 4);
  const largeDestination = allocate(rawBytes.length);

  copyWithinGPU(source, 0, destination, 0, 4);
  context.copyRawBytesToBuffer(
    rawBytes,
    largeDestination,
    0,
    rawBytes.length,
  );

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(encoders).toHaveLength(1);
  expect(encoders[0].commands).toEqual(["copy"]);
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    buffers[2],
    0,
    rawBytes,
    0,
    rawBytes.length,
  );
  expect(events).toEqual(["copy", "finish", "submit", "writeBuffer"]);
});

test("a Wasm upload is ordered in the pending command encoder", async () => {
  const {
    context,
    queue,
    memory,
    encoders,
    buffers,
    source,
    destination,
  } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const rawBytes = new Uint8Array([5, 6, 7, 8]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyWithinGPU(source, 0, destination, 0, rawBytes.length);
  copyToGPU(128, destination, 4, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(128, rawBytes.length);
  expect(queue.submit).not.toHaveBeenCalled();
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    buffers[2],
    0,
    rawBytes,
    0,
    rawBytes.length,
  );
  expect(encoders[0].copyBufferToBuffer).toHaveBeenLastCalledWith(
    buffers[2],
    0,
    buffers[1],
    4,
    rawBytes.length,
  );
  expect(encoders[0].commands).toEqual(["copy", "copy"]);

  await context.sync();
  expect(queue.submit).toHaveBeenCalledTimes(1);
});

test("an aligned CPU to GPU copy writes the requested bytes", () => {
  const { context, device, queue, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const wasmMemory = new WebAssembly.Memory({ initial: 1 });
  const rawBytes = new Uint8Array(wasmMemory.buffer, 128, 8);
  rawBytes.set([1, 2, 3, 4, 5, 6, 7, 8]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyToGPU(128, destination, 12, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(128, rawBytes.length);
  expect(memory.loadRawBytes).not.toHaveBeenCalled();
  expect(queue.writeBuffer.mock.calls[0][2].buffer).toBe(wasmMemory.buffer);
  expect(queue.writeBuffer).toHaveBeenCalledWith(
    device.createBuffer.mock.results[1].value,
    12,
    rawBytes,
    0,
    rawBytes.length
  );
});

test("an unaligned CPU to GPU copy pads the write to four bytes", () => {
  const { context, device, queue, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");
  const rawBytes = new Uint8Array([1, 2, 3]);
  memory.viewRawBytes.mockReturnValue(rawBytes);

  copyToGPU(256, destination, 4, rawBytes.length);

  expect(memory.viewRawBytes).toHaveBeenCalledWith(256, rawBytes.length);
  expect(memory.loadRawBytes).not.toHaveBeenCalled();
  expect(queue.writeBuffer).toHaveBeenCalledTimes(1);
  const [buffer, toOffset, data, dataOffset, nbytes] =
    queue.writeBuffer.mock.calls[0];
  expect(buffer).toBe(device.createBuffer.mock.results[1].value);
  expect(toOffset).toBe(4);
  expect(Array.from(data)).toEqual([1, 2, 3, 0]);
  expect(dataOffset).toBe(0);
  expect(nbytes).toBe(4);
});

test("a non-four-byte GPU allocation is rounded up for padded writes", () => {
  const { context, device } = createContext();
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  allocate(3);

  expect(device.createBuffer).toHaveBeenLastCalledWith({
    size: 4,
    usage: GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  expect(context.currAllocatedBytes).toBe(64 + 64 + 4);
});

test.each([
  [-1, "destination offset"],
  [0.5, "destination offset"],
  [2, "destination offset"],
])("a CPU to GPU copy rejects invalid offset %p", (offset, message) => {
  const { context, memory, destination } = createContext();
  const copyToGPU = context.getDeviceAPI("deviceCopyToGPU");

  expect(() => copyToGPU(128, destination, offset, 4)).toThrow(message);
  expect(memory.viewRawBytes).not.toHaveBeenCalled();
});

test.each([
  [-1, "source offset"],
  [0.5, "source offset"],
  [2, "source offset"],
])("a GPU readback rejects invalid offset %p", (offset, message) => {
  const { context, source } = createContext();
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  expect(() => copyFromGPU(source, offset, 128, 4)).toThrow(message);
});

test("an unaligned GPU readback copies four bytes and stores the logical bytes", async () => {
  const {
    context,
    device,
    memory,
    source,
  } = createContext();
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  device.createBuffer.mockImplementationOnce((descriptor) => {
    const mappedData = new Uint8Array([1, 2, 3, 99]).buffer;
    return {
      size: descriptor.size,
      destroy: jest.fn(),
      mapAsync: jest.fn(() => Promise.resolve()),
      getMappedRange: jest.fn(() => mappedData),
      unmap: jest.fn(),
    };
  });

  copyFromGPU(source, 0, 128, 3);
  await context.sync();

  const copyEncoder = device.createCommandEncoder.mock.results[0].value;
  expect(copyEncoder.copyBufferToBuffer).toHaveBeenCalledWith(
    device.createBuffer.mock.results[0].value,
    0,
    device.createBuffer.mock.results[2].value,
    0,
    4,
  );
  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(Array.from(memory.storeRawBytes.mock.calls[0][1])).toEqual([1, 2, 3]);
});

test("a GPU readback flushes pending copies before its own submission", async () => {
  const {
    context,
    queue,
    events,
    encoders,
    memory,
    source,
    destination,
  } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  copyWithinGPU(source, 0, destination, 0, 16);
  copyFromGPU(destination, 0, 128, 16);

  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(encoders).toHaveLength(2);
  expect(encoders[0].commands).toEqual(["copy"]);
  expect(encoders[1].commands).toEqual(["copy"]);
  expect(events).toEqual(["copy", "finish", "submit", "copy", "finish", "submit"]);

  await context.sync();

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(memory.storeRawBytes.mock.calls[0][1]).toHaveLength(16);
  expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();
});

test("buffer deallocation defers destroy until pending commands are submitted", async () => {
  const { context, queue, events, buffers, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const free = context.getDeviceAPI("deviceFreeDataSpace");
  const allocate = context.getDeviceAPI("deviceAllocDataSpace");

  copyWithinGPU(source, 0, destination, 0, 16);
  free(source);

  expect(queue.submit).not.toHaveBeenCalled();
  expect(buffers[0].destroy).not.toHaveBeenCalled();
  expect(events).toEqual(["copy"]);

  const replacement = allocate(64);
  expect(replacement).toBe(source);
  copyWithinGPU(replacement, 0, destination, 16, 16);

  await context.sync();

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(buffers[0].destroy).toHaveBeenCalledTimes(1);
  expect(buffers[2].destroy).not.toHaveBeenCalled();
  expect(events).toEqual(["copy", "copy", "finish", "submit", "destroy"]);
});

test("buffer deallocation destroys immediately without pending commands", () => {
  const { context, queue, events, buffers, source } = createContext();
  const free = context.getDeviceAPI("deviceFreeDataSpace");

  free(source);

  expect(queue.submit).not.toHaveBeenCalled();
  expect(buffers[0].destroy).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["destroy"]);
});

test("drawing flushes pending copies first", () => {
  const { context, queue, events, source, destination } = createContext();
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");
  const canvasRenderManager = {
    draw: jest.fn(() => events.push("draw")),
  };
  context.canvasRenderManager = canvasRenderManager;

  copyWithinGPU(source, 0, destination, 0, 16);
  context.drawImageFromBuffer(destination, 2, 2);

  expect(queue.submit).toHaveBeenCalledTimes(1);
  expect(canvasRenderManager.draw).toHaveBeenCalledTimes(1);
  expect(events).toEqual(["copy", "finish", "submit", "draw"]);
});

test("sync awaits a readback and a later batched GPU copy", async () => {
  const readback = createDeferred();
  const queueDone = createDeferred();
  const {
    context,
    queue,
    memory,
    source,
    destination,
  } = createContext({
    mapAsync: () => readback.promise,
    onSubmittedWorkDone: () => queueDone.promise,
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");

  copyFromGPU(source, 0, 128, 16);
  copyWithinGPU(source, 0, destination, 0, 16);

  let syncResolved = false;
  const syncPromise = context.sync().then(() => {
    syncResolved = true;
  });

  expect(queue.submit).toHaveBeenCalledTimes(2);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  queueDone.resolve();
  await Promise.resolve();
  expect(syncResolved).toBe(false);
  expect(memory.storeRawBytes).not.toHaveBeenCalled();

  readback.resolve();
  await syncPromise;

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
  expect(memory.storeRawBytes.mock.calls[0][0]).toBe(128);
  expect(memory.storeRawBytes.mock.calls[0][1]).toHaveLength(16);
});

test("a host write after a readback makes sync wait for the queue", async () => {
  const queueDone = createDeferred();
  const {
    context,
    queue,
    memory,
    source,
    destination,
  } = createContext({
    onSubmittedWorkDone: () => queueDone.promise,
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");

  copyFromGPU(source, 0, 128, 16);
  context.copyRawBytesToBuffer(
    new Uint8Array([1, 2, 3, 4]),
    destination,
    0,
    4
  );

  let syncResolved = false;
  const syncPromise = context.sync().then(() => {
    syncResolved = true;
  });
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);

  await Promise.resolve();
  expect(syncResolved).toBe(false);

  queueDone.resolve();
  await syncPromise;

  expect(memory.storeRawBytes).toHaveBeenCalledTimes(1);
});

test("sync propagates a pending readback failure", async () => {
  const readError = new Error("mapAsync failed");
  const {
    context,
    queue,
    source,
    destination,
  } = createContext({
    mapAsync: () => Promise.reject(readError),
  });
  const copyFromGPU = context.getDeviceAPI("deviceCopyFromGPU");
  const copyWithinGPU = context.getDeviceAPI("deviceCopyWithinGPU");

  copyFromGPU(source, 0, 128, 16);
  copyWithinGPU(source, 0, destination, 0, 16);

  await expect(context.sync()).rejects.toBe(readError);
  expect(queue.onSubmittedWorkDone).toHaveBeenCalledTimes(1);
});
