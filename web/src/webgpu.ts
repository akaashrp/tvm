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
import { assert } from "./support";
import { Pointer } from "./ctypes";
import { Memory } from "./memory";
import {
  Disposable,
  SampledTokenReadbackBatch,
  SampledTokenReadbackRingOptions,
  WebGPUExecutionOptions,
} from "./types";

// Keep ordered staging focused on small control uploads. Large writes are
// submitted before a direct queue.writeBuffer so the staging pool cannot
// retain model-sized buffers after initialization.
const MAX_STAGED_UPLOAD_BYTES = 64 * 1024;

// Pack per-dispatch POD arguments into aligned regions of a small number of
// buffers. A K-token command batch can contain thousands of dispatches; one
// GPUBuffer per dispatch creates avoidable driver objects and retained memory.
const UNIFORM_ARENA_BYTES = 1024 * 1024;

/** A pointer to points to the raw address space. */
export type GPUPointer = number;

type RuntimeTraceLevel = "major" | "verbose";
type RuntimeTraceLane = "cpu" | "gpu";
type RuntimeTraceValue = string | number | boolean | null;
type RuntimeTraceMeta = Record<string, RuntimeTraceValue>;

interface RuntimeTraceState {
  enabled: boolean;
  level: RuntimeTraceLevel;
  devtools: "off" | "major" | "all";
  ctx: string;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  enable_gpu_timestamps?: boolean;
}

interface RuntimeTracePayload {
  phase: string;
  level?: RuntimeTraceLevel;
  lane?: RuntimeTraceLane;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  meta?: RuntimeTraceMeta;
  abs_ts_ms?: number;
  ctx?: string;
}

interface RuntimeTimestampEntry {
  phase: string;
  startQuery: number;
  endQuery: number;
  meta: RuntimeTraceMeta;
}

type ReadbackRingSlotState = "free" | "pending" | "ready";

interface InternalSampledTokenReadbackSlot {
  buffer: GPUBuffer;
  state: ReadbackRingSlotState;
  batchSeq: number;
  submitSeq: number;
  step?: number | string;
  tokenCount: number;
  tokens?: Int32Array;
  pendingPromise?: Promise<void>;
  destroyAfterComplete: boolean;
}

interface InternalReadbackRingWaiter {
  resolve: () => void;
  reject: (reason: unknown) => void;
}

interface InternalSampledTokenReadbackRing {
  id: number;
  slotCount: number;
  maxTokensPerBatch: number;
  slotBytes: number;
  slots: InternalSampledTokenReadbackSlot[];
  nextSubmitCursor: number;
  nextBatchSeq: number;
  nextPollSeq: number;
  readyByBatchSeq: Map<number, number>;
  waiters: InternalReadbackRingWaiter[];
  disposed: boolean;
  fatalError?: Error;
}

interface SampledTokenReadbackSource {
  from: GPUPointer;
  fromTokenOffset: number;
  tokenCount: number;
}

declare global {
  var __WEBLLM_TRACE_RUNTIME_PUSH__: ((payload: RuntimeTracePayload) => void) | undefined;
  var __WEBLLM_TRACE_RUNTIME_STATE__: RuntimeTraceState | undefined;
}

function runtimeNowAbsMs(): number {
  return performance.timeOrigin + performance.now();
}

function runtimeTraceLevelEnabled(level: RuntimeTraceLevel): boolean {
  const state = globalThis.__WEBLLM_TRACE_RUNTIME_STATE__;
  if (!state || !state.enabled) {
    return false;
  }
  if (state.level === "major" && level === "verbose") {
    return false;
  }
  return true;
}

function runtimeTraceGPUTimeEnabled(): boolean {
  const state = globalThis.__WEBLLM_TRACE_RUNTIME_STATE__;
  return !!(state?.enabled && state.enable_gpu_timestamps);
}

function runtimeTraceCurrentStep(): number | string | undefined {
  return globalThis.__WEBLLM_TRACE_RUNTIME_STATE__?.step;
}

function runtimeTraceEmit(
  phase: string,
  meta: RuntimeTraceMeta = {},
  options: {
    level?: RuntimeTraceLevel;
    lane?: RuntimeTraceLane;
    step?: number | string;
    abs_ts_ms?: number;
  } = {},
): void {
  const level = options.level ?? "verbose";
  if (!runtimeTraceLevelEnabled(level)) {
    return;
  }
  const tracePush = globalThis.__WEBLLM_TRACE_RUNTIME_PUSH__;
  if (tracePush === undefined) {
    return;
  }
  tracePush({
    phase,
    level,
    lane: options.lane ?? "cpu",
    step: options.step ?? runtimeTraceCurrentStep(),
    meta,
    abs_ts_ms: options.abs_ts_ms ?? runtimeNowAbsMs(),
  });
}

function parseWGSLWorkgroupSize(code: string, entryPoint: string): [number, number, number] {
  const escapedName = entryPoint.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  let match = code.match(
    new RegExp(
      `@compute\\s+@workgroup_size\\(([^)]*)\\)\\s*fn\\s+${escapedName}\\b`,
    ),
  );
  if (match === null) {
    match = code.match(/@compute\s+@workgroup_size\(([^)]*)\)/);
  }
  if (match === null) {
    return [1, 1, 1];
  }
  const dims = match[1].split(",").map((value) => {
    const parsed = Number.parseInt(value.trim(), 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
  });
  return [dims[0] ?? 1, dims[1] ?? 1, dims[2] ?? 1];
}

export interface GPUDeviceDetectOutput {
  adapter: GPUAdapter;
  adapterInfo: GPUAdapterInfo;
  device: GPUDevice;
}

function roundUpToFourBytes(nbytes: number): number {
  if (!Number.isSafeInteger(nbytes) || nbytes < 0) {
    throw new Error(`Invalid WebGPU buffer size: ${nbytes}`);
  }
  const aligned = Math.ceil(nbytes / 4) * 4;
  if (!Number.isSafeInteger(aligned)) {
    throw new Error(`WebGPU buffer size is too large to align: ${nbytes}`);
  }
  return aligned;
}

function validateWebGPUCopyOffset(offset: number, name: string): void {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset % 4 != 0) {
    throw new Error(`${name} must be a nonnegative multiple of four: ${offset}`);
  }
}

/**
 * DetectGPU device in the environment.
 */
export async function detectGPUDevice(powerPreference: "low-power" | "high-performance" = "high-performance"): Promise<GPUDeviceDetectOutput | undefined> {
  if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (adapter == null) {
      throw Error(
        "Unable to find a compatible GPU. This issue might be because your computer doesn't have a GPU, or your system settings are not configured properly. " +
          "Please check if your device has a GPU properly set up and if your your browser supports WebGPU. " +
          "You can also consult your browser's compatibility chart to see if it supports WebGPU. " +
          "For more information about WebGPU support in your browser, visit https://webgpureport.org/"
      );
    }
    const computeMB = (value: number) => {
      return Math.ceil(value / (1 << 20)) + "MB";
    }

    // more detailed error message
    let requiredMaxBufferSize = 1 << 30;  // 1GB
    if (requiredMaxBufferSize > adapter.limits.maxBufferSize) {
      // If 1GB is too large, try 256MB (default size stated in WebGPU doc)
      const backupRequiredMaxBufferSize = 1 << 28;  // 256MB
      console.log(
        `Requested maxBufferSize exceeds limit. \n` +
        `requested=${computeMB(requiredMaxBufferSize)}, \n` +
        `limit=${computeMB(adapter.limits.maxBufferSize)}. \n` +
        `WARNING: Falling back to ${computeMB(backupRequiredMaxBufferSize)}...`
      );
      requiredMaxBufferSize = backupRequiredMaxBufferSize;
      if (backupRequiredMaxBufferSize > adapter.limits.maxBufferSize) {
        // Fail if 256MB is still too big
        throw Error(
          `Cannot initialize runtime because of requested maxBufferSize ` +
          `exceeds limit. requested=${computeMB(backupRequiredMaxBufferSize)}, ` +
          `limit=${computeMB(adapter.limits.maxBufferSize)}. ` +
          `Consider upgrading your browser.`
        );
      }
    }

    let requiredMaxStorageBufferBindingSize = 1 << 30;  // 1GB
    if (requiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
      // If 1GB is too large, try 128MB (default size stated in WebGPU doc)
      const backupRequiredMaxStorageBufferBindingSize = 1 << 27;  // 128MB
      console.log(
        `Requested maxStorageBufferBindingSize exceeds limit. \n` +
        `requested=${computeMB(requiredMaxStorageBufferBindingSize)}, \n` +
        `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. \n` +
        `WARNING: Falling back to ${computeMB(backupRequiredMaxStorageBufferBindingSize)}...`
      );
      requiredMaxStorageBufferBindingSize = backupRequiredMaxStorageBufferBindingSize;
      if (backupRequiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
        // Fail if 128MB is still too big
        throw Error(
          `Cannot initialize runtime because of requested maxStorageBufferBindingSize ` +
          `exceeds limit. requested=${computeMB(backupRequiredMaxStorageBufferBindingSize)}, ` +
          `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. `
        );
      }
    }

    const requiredMaxComputeWorkgroupStorageSize = 32 << 10;
    if (requiredMaxComputeWorkgroupStorageSize > adapter.limits.maxComputeWorkgroupStorageSize) {
      throw Error(
        `Cannot initialize runtime because of requested maxComputeWorkgroupStorageSize ` +
        `exceeds limit. requested=${requiredMaxComputeWorkgroupStorageSize}, ` +
        `limit=${adapter.limits.maxComputeWorkgroupStorageSize}. `
      );
    }

    const requiredMaxStorageBuffersPerShaderStage = 10;  // default is 8
    if (requiredMaxStorageBuffersPerShaderStage > adapter.limits.maxStorageBuffersPerShaderStage) {
      throw Error(
        `Cannot initialize runtime because of requested maxStorageBuffersPerShaderStage ` +
        `exceeds limit. requested=${requiredMaxStorageBuffersPerShaderStage}, ` +
        `limit=${adapter.limits.maxStorageBuffersPerShaderStage}. `
      );
    }

    const candidates = [1024, 512, 256];
    const invocationLimit = adapter.limits.maxComputeInvocationsPerWorkgroup;
    const requiredMaxComputeInvocationsPerWorkgroup =
      candidates.find(x => x <= invocationLimit) || undefined;
    if (requiredMaxComputeInvocationsPerWorkgroup === undefined) {
      console.log(`No candidate fits invocation limit=${invocationLimit}; will rely on defaults`);
    } else if (requiredMaxComputeInvocationsPerWorkgroup !== 1024) {
      console.log(
        `Falling back to maxComputeInvocationsPerWorkgroup=${requiredMaxComputeInvocationsPerWorkgroup} ` +
        `due to device limit=${invocationLimit}`
      )
    }
    const workgroupSizeXLimit = adapter.limits.maxComputeWorkgroupSizeX;
    const requiredMaxComputeWorkgroupSizeX =
      candidates.find(x => x <= workgroupSizeXLimit) || undefined;
    if (requiredMaxComputeWorkgroupSizeX === undefined) {
      console.log(`No candidate fits workgroup X limit=${workgroupSizeXLimit}; will rely on defaults`);
    } else if (requiredMaxComputeWorkgroupSizeX !== 1024) {
      console.log(
        `Falling back to maxComputeWorkgroupSizeX=${requiredMaxComputeWorkgroupSizeX} ` +
        `due to device limit=${workgroupSizeXLimit}`
      )
    }

    const requiredFeatures: GPUFeatureName[] = [];
    // Always require f16 if available
    if (adapter.features.has("shader-f16")) {
      requiredFeatures.push("shader-f16");
    }
    if (adapter.features.has("subgroups")) {
      requiredFeatures.push("subgroups");
    }
    // Enable optional timestamp queries when available to support profiling traces.
    if (adapter.features.has("timestamp-query")) {
      requiredFeatures.push("timestamp-query");
    }
    // requestAdapterInfo() is deprecated, causing requestAdapterInfo to raise
    // issue when building. However, it is still needed for older browsers, hence `as any`.
    const adapterInfo = adapter.info || await (adapter as any).requestAdapterInfo();
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: requiredMaxBufferSize,
        maxStorageBufferBindingSize: requiredMaxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: requiredMaxComputeWorkgroupStorageSize,
        maxStorageBuffersPerShaderStage: requiredMaxStorageBuffersPerShaderStage,
        maxComputeInvocationsPerWorkgroup: requiredMaxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: requiredMaxComputeWorkgroupSizeX,
      },
      requiredFeatures
    });
    return {
      adapter: adapter,
      adapterInfo: adapterInfo,
      device: device
    };
  } else {
    return undefined;
  }
}

/**
 * Create GPU buffer with `createBuffer()` but with error catching; destroy if error caught.
 * @param device The GPUDevice used to create a buffer.
 * @param descriptor The GPUBufferDescriptor passed to `createBuffer()`.
 * @returns The buffer created by `createBuffer()`.
 *
 * Note: We treat any error occurred at `createBuffer()` fatal and expect the user to handle
 *   `device.destroy()` with `device.lost.then()`.
 */
function tryCreateBuffer(device: GPUDevice, descriptor: GPUBufferDescriptor) {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");

  const buffer = device.createBuffer(descriptor);

  // Destroy at most once even if multiple error types fire.
  Promise.all([
    device.popErrorScope(),
    device.popErrorScope(),
    device.popErrorScope(),
  ]).then((errors) => {
    const captured = errors.filter((error): error is GPUError => error !== null);
    if (captured.length > 0) {
      device.destroy();
      captured.forEach((error) => console.error(error));
    }
  }).catch((err) => {
    console.error("Failed to pop error scopes:", err);
  });

  return buffer;
}

const canvasRenderWGSL = `
@group(0) @binding(0) var my_sampler : sampler;
@group(0) @binding(1) var my_texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vidx : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.position = vec4(pos[vidx], 0.0, 1.0);
  output.uv = uv[vidx];
  return output;
}

@fragment
fn fragment_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(my_texture, my_sampler, uv);
}

@fragment
fn fragment_clear(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return vec4(1.0, 1.0, 1.0, 1.0);
}
`
class CanvasRenderManager implements Disposable {
  private device: GPUDevice;
  private canvasContext: GPUCanvasContext;
  private stagingTexture: GPUTexture;
  private renderSampler: GPUSampler;
  private renderPipeline: GPURenderPipeline;
  private clearPipeline: GPURenderPipeline;
  private canvasTextureFormat: GPUTextureFormat;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    const ctx = canvas.getContext("webgpu");
    if (ctx == null) {
      throw Error("Cannot bind WebGPU context");
    }
    // avoid possible ts complain
    this.canvasContext = ctx as any;
    this.canvasTextureFormat = navigator.gpu.getPreferredCanvasFormat();
    this.canvasContext.configure({
      device: this.device,
      format: this.canvasTextureFormat,
      alphaMode: "opaque",
    });

    this.renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "vertex_main",
      },
      fragment: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "fragment_main",
        targets: [{
          format: this.canvasTextureFormat,
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    this.clearPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "vertex_main",
      },
      fragment: {
        module: device.createShaderModule({
          code: canvasRenderWGSL,
        }),
        entryPoint: "fragment_clear",
        targets: [{
          format: this.canvasTextureFormat,
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    this.renderSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    // staging texture always be in RGBA
    this.stagingTexture = device.createTexture({
      size: [canvas.height, canvas.width, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  clear() {
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.canvasContext.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    passEncoder.setPipeline(this.clearPipeline);
    const renderBindingGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.renderSampler },
        { binding: 1, resource: this.stagingTexture.createView() },
      ],
    });
    passEncoder.setBindGroup(0, renderBindingGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
    runtimeTraceEmit(
      "webgpu.queue.submit",
      { reason: "canvas_clear" },
      { level: "verbose" },
    );
  }

  draw(buffer: GPUBuffer, height: number, width: number) {
    // resize the staging texture
    if (height != this.stagingTexture.height || width != this.stagingTexture.width) {
      this.stagingTexture.destroy();
      this.stagingTexture = this.device.createTexture({
        size: [height, width, 1],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToTexture({
      buffer: buffer,
      offset: 0,
      bytesPerRow: this.stagingTexture.width * 4
    }, {
      texture: this.stagingTexture
    }, {
      width: this.stagingTexture.width,
      height: this.stagingTexture.height
    });

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.canvasContext.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    passEncoder.setPipeline(this.renderPipeline);
    const renderBindingGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.renderSampler },
        { binding: 1, resource: this.stagingTexture.createView() },
      ],
    });
    passEncoder.setBindGroup(0, renderBindingGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
    runtimeTraceEmit(
      "webgpu.queue.submit",
      {
        reason: "canvas_draw",
        width,
        height,
      },
      { level: "verbose" },
    );
  }

  dispose(): void {
    this.stagingTexture.destroy();
  }
}

/**
 * Function info from the API
 */
export interface FunctionInfo {
  name: string;
  arg_types: Array<string>;
  launch_param_tags: Array<string>;
}

/**
 * WebGPU context
 * Manages all the webgpu resources here.
 */
export class WebGPUContext {
  device: GPUDevice;
  memory: Memory;
  // internal data
  private bufferTable: Array<GPUBuffer | undefined> = [undefined];
  private bufferTableFreeId: Array<number> = [];
  private canvasRenderManager?: CanvasRenderManager = undefined;
  // Pool of MAP_READ staging buffers to avoid per-copy create/destroy overhead
  private readStagingBufferPool: Array<{ buffer: GPUBuffer; size: number }> = [];
  private maxReadStagingBuffers = 4;
  // Pending GPU→CPU copies, including storing the mapped data in WASM memory.
  private pendingGPUToCPUCopy: Promise<void> | null = null;
  // Whether a pending GPU→CPU copy is still the last queue operation.
  private pendingGPUToCPUCopyIsQueueTail = false;
  // Pending read promises from sampled-token readback rings.
  // These are tracked separately from pendingGPUToCPUCopy because ring
  // submissions can overlap and should be awaited collectively in sync().
  private pendingSampledTokenReadbackReads: Set<Promise<void>> = new Set();
  private sampledTokenReadbackRings: Map<number, InternalSampledTokenReadbackRing> = new Map();
  private nextSampledTokenReadbackRingId = 1;
  // Batched command encoding: accumulate compute passes and GPU copies in a
  // single encoder, and submit only on flush to reduce JS-native transition overhead.
  private pendingEncoder: GPUCommandEncoder | null = null;
  private pendingComputePass: GPUComputePassEncoder | null = null;
  private pendingComputePassCount = 0;
  private readonly batchComputePasses: boolean;
  // Uniform arenas reused across flushes. Dispatch arguments occupy distinct,
  // aligned regions while their command encoder is pending.
  private uniformArenaPool: Array<GPUBuffer> = [];
  private uniformArenaPoolSizes: Array<number> = [];
  private pendingUniformArenaIndex = 0;
  private pendingUniformArenaOffset = 0;
  private pendingUniformArgumentBytes = 0;
  private pendingUniformReservedBytes = 0;
  // Pool of COPY_SRC buffers used to order CPU writes after commands already
  // recorded in pendingEncoder. Each write in a batch uses a distinct slot
  // because queue.writeBuffer executes before the command buffer is submitted.
  private uploadBufferPool: Array<GPUBuffer> = [];
  private uploadBufferPoolSizes: Array<number> = [];
  private pendingDispatchCount = 0;
  private readonly maxDispatchesPerSubmit: number;
  private readonly maxDeferredDestroyBytes: number;
  private pendingGPUToGPUCopyCount = 0;
  private pendingGPUToGPUCopyBytes = 0;
  private pendingStagedUploadCount = 0;
  private pendingStagedUploadBytes = 0;
  private pendingRingStagingCopyCount = 0;
  private pendingRingStagingCopyBytes = 0;
  // Buffers removed from the pointer table while commands still reference
  // them. The GPUBuffer objects remain alive until that encoder is submitted.
  private pendingBufferDestroys: Array<GPUBuffer> = [];
  private pendingBufferDestroyBytes = 0;
  // Optional GPU timestamp query resources (enabled only for tracing).
  private timestampQuerySet: GPUQuerySet | undefined = undefined;
  private timestampResolveBuffer: GPUBuffer | undefined = undefined;
  private timestampQueryCapacity = 0;
  private nextTimestampQuery = 0;
  private pendingTimestampEntries: Array<RuntimeTimestampEntry> = [];
  private pendingTimestampRead: Promise<void> = Promise.resolve();
  private traceSubmitCounter = 0;
  // flags for debugging
  // stats of the runtime.
  // peak allocation
  private peakAllocatedBytes = 0;
  // current allocation
  private currAllocatedBytes = 0;
  // all allocation(ignoring free)
  private allAllocatedBytes = 0;
  // shader submit counter
  private shaderSubmitCounter = 0;
  // limite number of shaders to be submitted, useful for debugging, default to -1
  protected debugShaderSubmitLimit = -1;
  // log and sync each step
  protected debugLogFinish = false;

  constructor(memory: Memory, device: GPUDevice, options: WebGPUExecutionOptions = {}) {
    if (options.batchComputePasses !== undefined && typeof options.batchComputePasses !== "boolean") {
      throw new Error("batchComputePasses must be a boolean.");
    }
    this.batchComputePasses = options.batchComputePasses ?? true;
    const limit = options.maxDispatchesPerSubmit ?? 128;
    if (!Number.isSafeInteger(limit) || limit < 0) {
      throw new Error("maxDispatchesPerSubmit must be a non-negative safe integer.");
    }
    this.maxDispatchesPerSubmit = limit;
    const destroyLimit = options.maxDeferredDestroyBytes ?? 512 * 1024 * 1024;
    if (!Number.isSafeInteger(destroyLimit) || destroyLimit < 0) {
      throw new Error("maxDeferredDestroyBytes must be a non-negative safe integer.");
    }
    this.maxDeferredDestroyBytes = destroyLimit;
    this.memory = memory;
    this.device = device;
    runtimeTraceEmit(
      "webgpu.context.create",
      {
        timestamp_query_supported: this.device.features.has("timestamp-query"),
      },
      { level: "major" },
    );
  }

  private shouldRecordGPUTimestamps(): boolean {
    return (
      runtimeTraceGPUTimeEnabled() && this.device.features.has("timestamp-query")
    );
  }

  private ensureTimestampResources(minQueryCount: number): void {
    if (!this.shouldRecordGPUTimestamps()) {
      return;
    }
    if (minQueryCount <= this.timestampQueryCapacity) {
      return;
    }
    // A pending encoder may already contain passes that reference the current
    // query set. Submit it before replacing timestamp resources; destroying a
    // query set referenced by an unsubmitted command buffer invalidates that
    // entire submission.
    if (this.nextTimestampQuery !== 0) {
      this.flushCommands();
    }
    assert(this.nextTimestampQuery === 0);
    const nextCapacity = Math.max(
      1024,
      1 << Math.ceil(Math.log2(Math.max(minQueryCount, 2))),
    );
    this.timestampQuerySet?.destroy();
    this.timestampResolveBuffer?.destroy();
    this.timestampQuerySet = this.device.createQuerySet({
      type: "timestamp",
      count: nextCapacity,
    });
    this.timestampResolveBuffer = tryCreateBuffer(this.device, {
      size: nextCapacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.timestampQueryCapacity = nextCapacity;
    runtimeTraceEmit(
      "webgpu.timestamp.resources",
      {
        query_capacity: nextCapacity,
      },
      { level: "major", lane: "gpu" },
    );
  }

  private scheduleTimestampRead(
    queryCount: number,
    submitAbsTsMs: number,
    submitSeq: number,
    entries: RuntimeTimestampEntry[],
    readBuffer: GPUBuffer,
    submitStep?: number | string,
  ): void {
    if (queryCount === 0 || entries.length === 0) {
      readBuffer.destroy();
      return;
    }
    const nbytes = queryCount * 8;
    this.pendingTimestampRead = this.pendingTimestampRead.then(async () => {
      let isMapped = false;
      try {
        runtimeTraceEmit(
          "webgpu.timestamp.map_async.start",
          {
            query_count: queryCount,
            submit_seq: submitSeq,
          },
          { lane: "gpu", level: "verbose", step: submitStep },
        );
        await readBuffer.mapAsync(GPUMapMode.READ, 0, nbytes);
        isMapped = true;
        const mapped = readBuffer.getMappedRange(0, nbytes);
        const timestamps = new BigUint64Array(mapped.slice(0));
        readBuffer.unmap();
        isMapped = false;

        let baseTs = Number.MAX_SAFE_INTEGER;
        for (const entry of entries) {
          baseTs = Math.min(baseTs, Number(timestamps[entry.startQuery]));
        }
        if (!Number.isFinite(baseTs) || baseTs === Number.MAX_SAFE_INTEGER) {
          baseTs = 0;
        }

        for (const entry of entries) {
          const gpuStart = Number(timestamps[entry.startQuery]);
          const gpuEnd = Number(timestamps[entry.endQuery]);
          const durationMs = Math.max(0, (gpuEnd - gpuStart) / 1e6);
          const absTsMs = submitAbsTsMs + Math.max(0, (gpuStart - baseTs) / 1e6);
          runtimeTraceEmit(
            entry.phase,
            {
              ...entry.meta,
              submit_seq: submitSeq,
              gpu_start_tick: gpuStart,
              gpu_end_tick: gpuEnd,
              gpu_duration_ms: durationMs,
            },
            {
              lane: "gpu",
              level: "major",
              abs_ts_ms: absTsMs,
              step: submitStep,
            },
          );
        }
        runtimeTraceEmit(
          "webgpu.timestamp.map_async.end",
          {
            query_count: queryCount,
            submit_seq: submitSeq,
          },
          { lane: "gpu", level: "verbose", step: submitStep },
        );
      } catch (err) {
        if (isMapped) {
          try {
            readBuffer.unmap();
          } catch {
            // Best-effort cleanup on map/read errors.
          }
        }
        runtimeTraceEmit(
          "webgpu.timestamp.map_async.error",
          {
            submit_seq: submitSeq,
            message: String(err),
          },
          { lane: "gpu", level: "major", step: submitStep },
        );
      } finally {
        readBuffer.destroy();
      }
    });
  }

  /**
   * Flush all pending GPU commands by finishing and submitting the
   * accumulated command encoder.
   *
   * Must be called before:
   * - GPU→CPU readback (deviceCopyFromGPU)
   * - Buffer deallocation (deviceFreeDataSpace)
   * - Canvas drawing (drawImageFromBuffer)
   * - Queue sync (sync)
   *
   * @returns The submit sequence, or undefined when there were no pending commands.
   */
  flushCommands(): number | undefined {
    if (this.pendingEncoder) {
      this.endComputePass();
      let queryCount = 0;
      let timestampEntries: RuntimeTimestampEntry[] = [];
      let timestampReadBuffer: GPUBuffer | undefined;
      if (
        this.pendingTimestampEntries.length > 0 &&
        this.shouldRecordGPUTimestamps() &&
        this.timestampQuerySet !== undefined &&
        this.timestampResolveBuffer !== undefined
      ) {
        queryCount = this.nextTimestampQuery;
        timestampEntries = this.pendingTimestampEntries.slice();
        timestampReadBuffer = tryCreateBuffer(this.device, {
          size: queryCount * 8,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.pendingEncoder.resolveQuerySet(
          this.timestampQuerySet,
          0,
          queryCount,
          this.timestampResolveBuffer,
          0,
        );
        this.pendingEncoder.copyBufferToBuffer(
          this.timestampResolveBuffer,
          0,
          timestampReadBuffer,
          0,
          queryCount * 8,
        );
      }
      const submittedDispatches = this.pendingDispatchCount;
      const submittedUniformArenaCount =
        submittedDispatches === 0 ? 0 : this.pendingUniformArenaIndex + 1;
      const submittedUniformArgumentBytes = this.pendingUniformArgumentBytes;
      const submittedUniformReservedBytes = this.pendingUniformReservedBytes;
      const submittedGPUToGPUCopies = this.pendingGPUToGPUCopyCount;
      const submittedGPUToGPUCopyBytes = this.pendingGPUToGPUCopyBytes;
      const submittedStagedUploads = this.pendingStagedUploadCount;
      const submittedStagedUploadBytes = this.pendingStagedUploadBytes;
      const submittedRingStagingCopies = this.pendingRingStagingCopyCount;
      const submittedRingStagingCopyBytes = this.pendingRingStagingCopyBytes;
      const submittedBufferDestroys = this.pendingBufferDestroys;
      const submittedBufferDestroyBytes = this.pendingBufferDestroyBytes;
      const commandBuffer = this.pendingEncoder.finish();
      const submitAbsTsMs = runtimeNowAbsMs();
      const submitStep = runtimeTraceCurrentStep();
      this.device.queue.submit([commandBuffer]);
      this.traceSubmitCounter += 1;
      const submitSeq = this.traceSubmitCounter;
      runtimeTraceEmit(
        "webgpu.queue.submit",
        {
          submit_seq: submitSeq,
          dispatches: submittedDispatches,
          compute_passes: this.pendingComputePassCount,
          max_dispatches_per_submit: this.maxDispatchesPerSubmit,
          max_deferred_destroy_bytes: this.maxDeferredDestroyBytes,
          uniform_arena_count: submittedUniformArenaCount,
          uniform_argument_bytes: submittedUniformArgumentBytes,
          uniform_reserved_bytes: submittedUniformReservedBytes,
          gpu_to_gpu_copies: submittedGPUToGPUCopies,
          staged_upload_copies: submittedStagedUploads,
          ring_staging_copies: submittedRingStagingCopies,
          gpu_to_gpu_copy_bytes: submittedGPUToGPUCopyBytes,
          staged_upload_bytes: submittedStagedUploadBytes,
          ring_staging_copy_bytes: submittedRingStagingCopyBytes,
          copy_bytes:
            submittedGPUToGPUCopyBytes +
            submittedStagedUploadBytes +
            submittedRingStagingCopyBytes,
          deferred_destroys: submittedBufferDestroys.length,
          deferred_destroy_bytes: submittedBufferDestroyBytes,
          query_count: queryCount,
        },
        { level: "major", step: submitStep },
      );
      if (queryCount > 0 && timestampReadBuffer !== undefined) {
        this.scheduleTimestampRead(
          queryCount,
          submitAbsTsMs,
          submitSeq,
          timestampEntries,
          timestampReadBuffer,
          submitStep,
        );
      }
      this.pendingEncoder = null;
      this.pendingDispatchCount = 0;
      this.pendingComputePassCount = 0;
      this.pendingUniformArenaIndex = 0;
      this.pendingUniformArenaOffset = 0;
      this.pendingUniformArgumentBytes = 0;
      this.pendingUniformReservedBytes = 0;
      this.pendingGPUToGPUCopyCount = 0;
      this.pendingGPUToGPUCopyBytes = 0;
      this.pendingStagedUploadCount = 0;
      this.pendingStagedUploadBytes = 0;
      this.pendingRingStagingCopyCount = 0;
      this.pendingRingStagingCopyBytes = 0;
      this.pendingBufferDestroys = [];
      this.pendingBufferDestroyBytes = 0;
      this.pendingTimestampEntries = [];
      this.nextTimestampQuery = 0;
      for (const buffer of submittedBufferDestroys) {
        buffer.destroy();
      }
      if (submittedBufferDestroys.length > 0) {
        runtimeTraceEmit(
          "webgpu.deferred_destroy",
          {
            submit_seq: submitSeq,
            count: submittedBufferDestroys.length,
            bytes: submittedBufferDestroyBytes,
          },
          { level: "verbose", step: submitStep },
        );
      }
      // This submission is now the last queue operation, so the
      // GPU→CPU copy fast path in sync() is no longer valid.
      this.pendingGPUToCPUCopyIsQueueTail = false;
      return submitSeq;
    }
    return undefined;
  }

  /**
   * Submit pending commands without waiting for GPU completion.
   *
   * @returns Whether a command buffer was submitted.
   */
  submitPendingCommands(): boolean {
    return this.flushCommands() !== undefined;
  }

  /**
   * Dispose context.
   */
  dispose() {
    this.flushCommands();
    for (const ringId of Array.from(this.sampledTokenReadbackRings.keys())) {
      this.disposeSampledTokenReadbackRing(ringId);
    }
    this.canvasRenderManager?.dispose();
    this.bufferTableFreeId = [];
    while (this.bufferTable.length != 0) {
      this.bufferTable.pop()?.destroy();
    }
    for (const buf of this.uniformArenaPool) {
      buf.destroy();
    }
    this.uniformArenaPool.length = 0;
    this.uniformArenaPoolSizes.length = 0;
    for (const buf of this.uploadBufferPool) {
      buf.destroy();
    }
    this.uploadBufferPool.length = 0;
    this.uploadBufferPoolSizes.length = 0;
    while (this.readStagingBufferPool.length != 0) {
      this.readStagingBufferPool.pop()?.buffer.destroy();
    }
    this.timestampQuerySet?.destroy();
    this.timestampResolveBuffer?.destroy();
    this.device.destroy();
  }


  /**
   * Wait for all pending GPU tasks to complete
   */
  async sync(): Promise<void> {
    const syncStart = runtimeNowAbsMs();
    runtimeTraceEmit("webgpu.sync.start", {}, { level: "verbose" });
    this.flushCommands();

    const pendingRead = this.pendingGPUToCPUCopy;
    const pendingReadIsQueueTail = this.pendingGPUToCPUCopyIsQueueTail;
    this.pendingGPUToCPUCopy = null;
    this.pendingGPUToCPUCopyIsQueueTail = false;

    if (pendingRead && pendingReadIsQueueTail) {
      await pendingRead;
    } else {
      const queueDone = this.device.queue.onSubmittedWorkDone();
      if (pendingRead) {
        await Promise.all([pendingRead, queueDone]);
      } else {
        await queueDone;
      }
    }
    if (this.pendingSampledTokenReadbackReads.size > 0) {
      await Promise.all(Array.from(this.pendingSampledTokenReadbackReads));
    }
    // Ensure async timestamp readbacks are finished before returning so
    // callers can drain GPU-lane events deterministically.
    await this.pendingTimestampRead;
    runtimeTraceEmit(
      "webgpu.sync.end",
      {
        duration_ms: runtimeNowAbsMs() - syncStart,
      },
      { level: "verbose" },
    );
  }

  /**
   * Obtain the runtime information in readable format.
   */
  runtimeStatsText(): string {
    let info = "peak-memory=" + Math.ceil(this.peakAllocatedBytes / (1 << 20)) + " MB";
    info += ", all-memory=" + Math.ceil(this.allAllocatedBytes / (1 << 20)) + " MB";
    info += ", shader-submissions=" + this.shaderSubmitCounter;
    return info;
  }

  /**
   * Create a sampled-token readback ring.
   *
   * Each submission copies a contiguous int32 token vector from a GPU buffer to a
   * per-slot MAP_READ staging buffer and starts slot-local mapAsync.
   */
  createSampledTokenReadbackRing(options: SampledTokenReadbackRingOptions): number {
    const slotCount = options.slotCount ?? 3;
    const maxTokensPerBatch = options.maxTokensPerBatch;
    if (!Number.isInteger(slotCount) || slotCount <= 0) {
      throw new Error(`slotCount must be a positive integer. Got ${slotCount}.`);
    }
    if (!Number.isInteger(maxTokensPerBatch) || maxTokensPerBatch <= 0) {
      throw new Error(
        `maxTokensPerBatch must be a positive integer. Got ${maxTokensPerBatch}.`,
      );
    }
    const slotBytes = maxTokensPerBatch * 4; // assume int32 tokens
    const slots: InternalSampledTokenReadbackSlot[] = [];
    for (let i = 0; i < slotCount; ++i) {
      slots.push({
        buffer: tryCreateBuffer(this.device, {
          size: slotBytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        state: "free",
        batchSeq: 0,
        submitSeq: 0,
        tokenCount: 0,
        destroyAfterComplete: false,
      });
    }
    const ringId = this.nextSampledTokenReadbackRingId++;
    this.sampledTokenReadbackRings.set(ringId, {
      id: ringId,
      slotCount,
      maxTokensPerBatch,
      slotBytes,
      slots,
      nextSubmitCursor: 0,
      nextBatchSeq: 1,
      nextPollSeq: 1,
      readyByBatchSeq: new Map<number, number>(),
      waiters: [],
      disposed: false,
    });
    runtimeTraceEmit(
      "webgpu.readback_ring.create",
      {
        ring_id: ringId,
        slot_count: slotCount,
        max_tokens_per_batch: maxTokensPerBatch,
        slot_bytes: slotBytes,
      },
      { level: "major" },
    );
    return ringId;
  }

  /**
   * Dispose a previously created sampled-token readback ring.
   */
  disposeSampledTokenReadbackRing(ringId: number): void {
    const ring = this.sampledTokenReadbackRings.get(ringId);
    if (ring === undefined) {
      return;
    }
    ring.disposed = true;
    this.sampledTokenReadbackRings.delete(ringId);
    for (const slot of ring.slots) {
      slot.tokens = undefined;
      ring.readyByBatchSeq.delete(slot.batchSeq);
      if (slot.state === "pending") {
        slot.destroyAfterComplete = true;
      } else {
        slot.buffer.destroy();
        this.resetSampledTokenReadbackSlot(slot);
      }
    }
    this.notifySampledTokenReadbackRingWaiters(ring);
    runtimeTraceEmit(
      "webgpu.readback_ring.dispose",
      { ring_id: ringId },
      { level: "major" },
    );
  }

  /**
   * Submit a sampled-token readback batch from a GPU buffer.
   *
   * @param ringId Ring handle returned by createSampledTokenReadbackRing.
   * @param from Source GPU pointer containing int32 tokens.
   * @param fromTokenOffset Source token offset (in token units, not bytes).
   * @param tokenCount Number of int32 token ids to copy.
   * @returns Monotonic batch sequence id.
   */
  submitSampledTokenReadbackRing(
    ringId: number,
    from: GPUPointer,
    fromTokenOffset: number,
    tokenCount: number,
  ): number {
    return this.submitSampledTokenReadbackRingSources(ringId, [
      { from, fromTokenOffset, tokenCount },
    ]);
  }

  /**
   * Submit one sampled token from each GPU pointer as one contiguous batch.
   */
  submitSampledTokenReadbackRingTokens(
    ringId: number,
    from: Array<GPUPointer>,
  ): number {
    if (from.length === 0) {
      throw new Error("At least one sampled-token source is required.");
    }
    return this.submitSampledTokenReadbackRingSources(
      ringId,
      from.map((ptr) => ({ from: ptr, fromTokenOffset: 0, tokenCount: 1 })),
    );
  }

  private submitSampledTokenReadbackRingSources(
    ringId: number,
    sources: Array<SampledTokenReadbackSource>,
  ): number {
    const ring = this.requireSampledTokenReadbackRing(ringId);
    this.throwIfSampledTokenReadbackRingFailed(ring);
    let tokenCount = 0;
    for (const source of sources) {
      if (!Number.isInteger(source.fromTokenOffset) || source.fromTokenOffset < 0) {
        throw new Error(
          `fromTokenOffset must be a non-negative integer. Got ${source.fromTokenOffset}.`,
        );
      }
      if (!Number.isInteger(source.tokenCount) || source.tokenCount <= 0) {
        throw new Error(
          `tokenCount must be a positive integer. Got ${source.tokenCount}.`,
        );
      }
      tokenCount += source.tokenCount;
    }
    if (tokenCount > ring.maxTokensPerBatch) {
      throw new Error(
        `tokenCount ${tokenCount} exceeds maxTokensPerBatch ${ring.maxTokensPerBatch}.`,
      );
    }
    const slotIdx = this.findFreeSampledTokenReadbackSlot(ring);
    if (slotIdx === -1) {
      throw new Error(
        `Sampled-token readback ring ${ring.id} is full. Drain or wait before submitting more batches.`,
      );
    }
    const slot = ring.slots[slotIdx];
    const batchSeq = ring.nextBatchSeq++;
    const nbytes = tokenCount * 4;

    const submitStep = runtimeTraceCurrentStep();
    this.endComputePass();
    if (!this.pendingEncoder) {
      this.pendingEncoder = this.device.createCommandEncoder();
    }
    let toByteOffset = 0;
    for (const source of sources) {
      const sourceBytes = source.tokenCount * 4;
      this.pendingEncoder.copyBufferToBuffer(
        this.gpuBufferFromPtr(source.from),
        source.fromTokenOffset * 4,
        slot.buffer,
        toByteOffset,
        sourceBytes,
      );
      toByteOffset += sourceBytes;
    }
    this.pendingRingStagingCopyCount += sources.length;
    this.pendingRingStagingCopyBytes += nbytes;
    const submitSeq = this.flushCommands();
    assert(submitSeq !== undefined);

    slot.state = "pending";
    slot.batchSeq = batchSeq;
    slot.submitSeq = submitSeq;
    slot.step = submitStep;
    slot.tokenCount = tokenCount;
    slot.tokens = undefined;

    runtimeTraceEmit(
      "webgpu.readback_ring.submit",
      {
        ring_id: ring.id,
        slot_idx: slotIdx,
        batch_seq: batchSeq,
        submit_seq: submitSeq,
        token_count: tokenCount,
        copy_count: sources.length,
        copy_bytes: nbytes,
      },
      { level: "major", step: submitStep },
    );
    runtimeTraceEmit(
      "webgpu.readback_ring.map_async.start",
      {
        ring_id: ring.id,
        slot_idx: slotIdx,
        batch_seq: batchSeq,
        submit_seq: submitSeq,
        bytes: nbytes,
      },
      { level: "major", step: submitStep },
    );

    const readPromise = slot.buffer
      .mapAsync(GPUMapMode.READ, 0, nbytes)
      .then(() => {
        const mapped = slot.buffer.getMappedRange(0, nbytes);
        const tokens = new Int32Array(mapped.slice(0, nbytes));
        slot.buffer.unmap();

        if (ring.disposed) {
          if (slot.destroyAfterComplete) {
            slot.buffer.destroy();
            slot.destroyAfterComplete = false;
          }
          this.resetSampledTokenReadbackSlot(slot);
          return;
        }

        slot.tokens = tokens;
        slot.state = "ready";
        ring.readyByBatchSeq.set(batchSeq, slotIdx);
        runtimeTraceEmit(
          "webgpu.readback_ring.map_async.end",
          {
            ring_id: ring.id,
            slot_idx: slotIdx,
            batch_seq: batchSeq,
            submit_seq: submitSeq,
            token_count: tokenCount,
            bytes: nbytes,
          },
          { level: "major", step: slot.step },
        );
        this.notifySampledTokenReadbackRingWaiters(ring);
      })
      .catch((err) => {
        try {
          slot.buffer.unmap();
        } catch {
          // Best-effort cleanup.
        }
        if (ring.disposed) {
          if (slot.destroyAfterComplete) {
            slot.buffer.destroy();
            slot.destroyAfterComplete = false;
          }
          this.resetSampledTokenReadbackSlot(slot);
          return;
        }
        const error = err instanceof Error ? err : new Error(String(err));
        this.resetSampledTokenReadbackSlot(slot);
        ring.fatalError = error;
        runtimeTraceEmit(
          "webgpu.readback_ring.map_async.error",
          {
            ring_id: ring.id,
            slot_idx: slotIdx,
            batch_seq: batchSeq,
            submit_seq: submitSeq,
            message: String(error),
          },
          { level: "major", step: slot.step },
        );
        this.notifySampledTokenReadbackRingWaiters(ring, error);
        throw error;
      })
      .finally(() => {
        slot.pendingPromise = undefined;
      });
    slot.pendingPromise = readPromise;
    this.trackPendingSampledTokenReadbackRead(readPromise);
    return batchSeq;
  }

  /**
   * Poll ready batches in sequence order.
   */
  pollSampledTokenReadbackRing(ringId: number): Array<SampledTokenReadbackBatch> {
    const ring = this.requireSampledTokenReadbackRing(ringId);
    this.throwIfSampledTokenReadbackRingFailed(ring);
    return this.drainReadySampledTokenReadbackBatches(ring);
  }

  /**
   * Wait until at least one batch becomes ready, then return all currently
   * contiguous ready batches.
   */
  async waitSampledTokenReadbackRing(
    ringId: number,
  ): Promise<Array<SampledTokenReadbackBatch>> {
    const ring = this.requireSampledTokenReadbackRing(ringId);
    this.throwIfSampledTokenReadbackRingFailed(ring);
    const readyNow = this.drainReadySampledTokenReadbackBatches(ring);
    if (readyNow.length > 0) {
      return readyNow;
    }
    if (!this.sampledTokenReadbackRingHasPending(ring)) {
      return [];
    }
    await new Promise<void>((resolve, reject) => {
      ring.waiters.push({ resolve, reject });
    });
    this.throwIfSampledTokenReadbackRingFailed(ring);
    return this.drainReadySampledTokenReadbackBatches(ring);
  }

  /**
   * Draw image from data in storage buffer.
   * @param ptr The GPU ptr
   * @param height The height of the image.
   * @param width The width of the image.
   */
  drawImageFromBuffer(ptr: GPUPointer, height: number, width: number) {
    if (this.canvasRenderManager == undefined) {
      throw Error("Do not have a canvas context, call bindCanvas first");
    }
    this.flushCommands();
    this.canvasRenderManager.draw(this.gpuBufferFromPtr(ptr), height, width);
    this.pendingGPUToCPUCopyIsQueueTail = false;
  }

  /**
   * Copy raw bytes into buffer ptr.
   *
   * @param rawBytes The raw bytes
   * @param toPtr The target gpu buffer ptr
   * @param toOffset The beginning offset
   * @param nbytes Number of bytes
   */
  copyRawBytesToBuffer(
    rawBytes: Uint8Array,
    toPtr: GPUPointer,
    toOffset: number,
    nbytes: number
  ): void {
    this.writeRawBytesToBuffer(
      rawBytes,
      toPtr,
      toOffset,
      nbytes,
      "copy_raw_bytes_to_buffer",
    );
  }
  /**
   * Clear canvas
   */
  clearCanvas() {
    if (this.canvasRenderManager) {
      this.canvasRenderManager.clear();
      this.pendingGPUToCPUCopyIsQueueTail = false;
    }
  }

  /**
   * Bind a canvas element to the runtime.
   * @param canvas The HTML canvas/
   */
  bindCanvas(canvas: HTMLCanvasElement) {
    this.canvasRenderManager = new CanvasRenderManager(this.device, canvas);
  }

  /**
   * Create a PackedFunc that runs the given shader
   * via createComputePipeline
   *
    * @param finfo The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @returns The shader
   */
  createShader(finfo: FunctionInfo, code: string): Function {
    return this.createShadeInternal(finfo, code, false) as Function;
  }

  /**
   * Create a PackedFunc that runs the given shader asynchronously
   * via createComputePipelineAsync
   *
    * @param finfo The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @returns The shader
   */
  async createShaderAsync(finfo: FunctionInfo, code: string): Promise<Function> {
    return await (this.createShadeInternal(finfo, code, true) as Promise<Function>);
  }

  /**
   * Reserve an aligned region for one dispatch's POD arguments.
   *
   * queue.writeBuffer() executes before a pending encoder is submitted, so
   * dispatches in that encoder cannot overwrite the same region. Arenas are
   * reused after flush; queue ordering keeps writes for the next submission
   * behind commands that consume the previous contents.
   */
  private allocateUniformRegion(nbytes: number): {
    buffer: GPUBuffer;
    offset: number;
  } {
    const alignment = Math.max(
      4,
      this.device.limits.minUniformBufferOffsetAlignment,
    );
    const reservedBytes = Math.ceil(nbytes / alignment) * alignment;
    let arenaIndex = this.pendingUniformArenaIndex;
    let offset = this.pendingUniformArenaOffset;
    let arenaSize = this.uniformArenaPoolSizes[arenaIndex] ?? 0;

    if (arenaSize !== 0 && offset + reservedBytes > arenaSize) {
      arenaIndex += 1;
      offset = 0;
      arenaSize = this.uniformArenaPoolSizes[arenaIndex] ?? 0;
    }

    const requiredArenaSize =
      Math.ceil(Math.max(UNIFORM_ARENA_BYTES, reservedBytes) / alignment) *
      alignment;
    if (arenaSize < requiredArenaSize) {
      this.uniformArenaPool[arenaIndex]?.destroy();
      this.uniformArenaPool[arenaIndex] = this.device.createBuffer({
        size: requiredArenaSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.uniformArenaPoolSizes[arenaIndex] = requiredArenaSize;
      arenaSize = requiredArenaSize;
    }

    const buffer = this.uniformArenaPool[arenaIndex];
    assert(buffer !== undefined);
    this.pendingUniformArenaIndex = arenaIndex;
    this.pendingUniformArenaOffset = offset + reservedBytes;
    this.pendingUniformArgumentBytes += nbytes;
    this.pendingUniformReservedBytes += reservedBytes;
    this.pendingDispatchCount += 1;
    return { buffer, offset };
  }

  /**
   * Get a staging buffer for a CPU→GPU write recorded in pendingEncoder.
   *
   * A batch cannot reuse a staging slot: queue.writeBuffer executes before
   * submit, so reuse would overwrite bytes needed by an earlier encoded copy.
   * Slots are reused after flush, where WebGPU queue ordering protects the
   * previous submission from the next queue write.
   */
  private getUploadBufferFromPool(nbytes: number): GPUBuffer {
    const uploadIdx = this.pendingStagedUploadCount;
    if (
      uploadIdx < this.uploadBufferPool.length &&
      this.uploadBufferPoolSizes[uploadIdx] >= nbytes
    ) {
      return this.uploadBufferPool[uploadIdx];
    }
    if (uploadIdx < this.uploadBufferPool.length) {
      this.uploadBufferPool[uploadIdx].destroy();
    }
    const buffer = this.device.createBuffer({
      size: nbytes,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.uploadBufferPool[uploadIdx] = buffer;
    this.uploadBufferPoolSizes[uploadIdx] = nbytes;
    return buffer;
  }

  /**
   * Preserve CPU→GPU write ordering without forcing a command submission.
   *
   * queue.writeBuffer targets a temporary buffer immediately. The pending
   * encoder then copies those bytes to the real destination after all commands
   * already recorded in it and before commands recorded by the caller next.
   */
  private writeRawBytesToBuffer(
    rawBytes: Uint8Array,
    toPtr: GPUPointer,
    toOffset: number,
    nbytes: number,
    reason: string,
  ): void {
    const destination = this.gpuBufferFromPtr(toPtr);
    const canStage =
      this.pendingEncoder !== null &&
      nbytes <= MAX_STAGED_UPLOAD_BYTES &&
      nbytes % 4 === 0 &&
      toOffset % 4 === 0;
    if (canStage) {
      this.endComputePass();
      const uploadBuffer = this.getUploadBufferFromPool(nbytes);
      this.device.queue.writeBuffer(
        uploadBuffer,
        0,
        rawBytes as GPUAllowSharedBufferSource,
        0,
        nbytes,
      );
      this.pendingEncoder.copyBufferToBuffer(
        uploadBuffer,
        0,
        destination,
        toOffset,
        nbytes,
      );
      this.pendingStagedUploadCount += 1;
      this.pendingStagedUploadBytes += nbytes;
      runtimeTraceEmit(
        "webgpu.queue.write_buffer",
        {
          bytes: nbytes,
          reason,
          ptr: toPtr,
          staged: true,
        },
        { level: "verbose" },
      );
    } else {
      const submittedPendingCommands = this.flushCommands() !== undefined;
      this.device.queue.writeBuffer(
        destination,
        toOffset,
        rawBytes as GPUAllowSharedBufferSource,
        0,
        nbytes,
      );
      runtimeTraceEmit(
        "webgpu.queue.write_buffer",
        {
          bytes: nbytes,
          reason,
          ptr: toPtr,
          staged: false,
          submitted_pending_commands: submittedPendingCommands,
        },
        { level: "verbose" },
      );
    }
    this.pendingGPUToCPUCopyIsQueueTail = false;
  }

  /**
   * Internal impl of createShader for both async and sync mode.
   *
    * @param finfo The function information already parsed as a record.
   * @param code The shader data(in WGSL)
   * @param asyncMode Whether use async mode.
   * @returns The shader function or promise of shader func.
   */
  private createShadeInternal(
    finfo: FunctionInfo,
    code: string,
    asyncMode: boolean
  ): Function | Promise<Function> {
    const dispatchToDim: Array<number> = [];
    let paramWriteAccess: Array<number> = [];
    const workgroupSize = parseWGSLWorkgroupSize(code, finfo.name);

    for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
      const tag: string = finfo.launch_param_tags[i];
      if (tag.startsWith("blockIdx.")) {
        const target: number = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
        assert(target >= 0 && target < 3);
        dispatchToDim.push(target);
      } else if (tag.startsWith("threadIdx.")) {
        const target: number = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
        assert(target >= 0 && target < 3);
        dispatchToDim.push(target + 3);
      } else if (tag.startsWith("paramWriteAccess:")) {
        paramWriteAccess = JSON.parse(tag.substring(17));
      } else {
        throw new Error("Cannot handle thread_axis " + tag);
      }
    }


    const layoutEntries: Array<GPUBindGroupLayoutEntry> = [];
    const bufferArgIndices: Array<number> = [];
    const podArgIndices: Array<number> = [];

    for (let i = 0; i < finfo.arg_types.length; ++i) {
      const dtype = finfo.arg_types[i];
      if (dtype == "handle") {
        layoutEntries.push({
          binding: bufferArgIndices.length,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: paramWriteAccess[bufferArgIndices.length] ? "storage" : "read-only-storage"
          }
        });
        bufferArgIndices.push(i);
      } else if (dtype.startsWith("int") || dtype.startsWith("uint") || dtype.startsWith("float")) {
        podArgIndices.push(i);
      } else {
        throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
      }
    }

    assert(paramWriteAccess.length == bufferArgIndices.length);
    // POD arguments are pass in the end
    layoutEntries.push({
      binding: bufferArgIndices.length,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "uniform"
      }
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: layoutEntries
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    runtimeTraceEmit(
      "webgpu.descriptor.setup",
      {
        function_name: finfo.name,
        bindings: layoutEntries.length,
      },
      { level: "verbose" },
    );

    // Function to create the pipeline.
    const createShaderFunc = (pipeline: GPUComputePipeline): Function => {
      const submitShader = (...args: Array<GPUPointer | number>): void => {
        if (this.debugShaderSubmitLimit != -1 &&
          this.shaderSubmitCounter >= this.debugShaderSubmitLimit) {
          this.shaderSubmitCounter += 1;
          return;
        }
        const encodeStart = performance.now();
        runtimeTraceEmit(
          "webgpu.command_encode.start",
          {
            function_name: finfo.name,
            shader_submit_counter: this.shaderSubmitCounter,
          },
          { level: "verbose" },
        );

        const bindGroupEntries: Array<GPUBindGroupEntry> = [];
        const numBufferOrPodArgs = bufferArgIndices.length + podArgIndices.length;
        let timestampEntry: RuntimeTimestampEntry | undefined = undefined;

        assert(args.length == numBufferOrPodArgs + dispatchToDim.length);

        const workDim: Array<number> = [1, 1, 1, 1, 1, 1];
        for (let i = 0; i < dispatchToDim.length; ++i) {
          workDim[dispatchToDim[i]] = args[numBufferOrPodArgs + i];
        }

        // get around 65535 restriction of blockIdx.x
        if (workDim[2] != 1) {
          throw Error("WebGPU: blockIdx.z is reserved for internal use");
        }
        const packDimX = workDim[0];
        // spread thinsg out into blockIdx.z
        if (workDim[0] >= (1 << 16)) {
          let wl_x = workDim[0];
          let wl_z = workDim[2];

          while (wl_x >= (1 << 16)) {
            if (wl_x % 2 == 0) {
              wl_x = wl_x / 2;
            } else {
              // pad up
              wl_x = (wl_x + 1) / 2;
            }
            wl_z *= 2;
          }
          workDim[0] = wl_x;
          workDim[2] = wl_z;
          assert(wl_x * wl_z >= packDimX);
        }

        const totalWorkgroups = workDim[0] * workDim[1] * workDim[2];
        const threadInvocations =
          totalWorkgroups * workgroupSize[0] * workgroupSize[1] * workgroupSize[2];

        if (this.shouldRecordGPUTimestamps()) {
          this.ensureTimestampResources(this.nextTimestampQuery + 2);
          if (this.timestampQuerySet !== undefined) {
            const startQuery = this.nextTimestampQuery;
            const endQuery = startQuery + 1;
            this.nextTimestampQuery += 2;
            timestampEntry = {
              phase: "gpu.compute.dispatch",
              startQuery,
              endQuery,
              meta: {
                function_name: finfo.name,
                packed_workgroups_x: packDimX,
                workgroups_x: workDim[0],
                workgroups_y: workDim[1],
                workgroups_z: workDim[2],
                workgroup_size_x: workgroupSize[0],
                workgroup_size_y: workgroupSize[1],
                workgroup_size_z: workgroupSize[2],
                total_workgroups: totalWorkgroups,
                thread_invocations: threadInvocations,
              },
            };
          }
        }
        // Timestamp-capacity growth can flush the prior encoder. Create the
        // encoder only after timestamp resources for this dispatch are ready.
        if (!this.pendingEncoder) {
          this.pendingEncoder = this.device.createCommandEncoder();
        }
        const computePassDescriptor =
          timestampEntry !== undefined && this.timestampQuerySet !== undefined
            ? {
                timestampWrites: {
                  querySet: this.timestampQuerySet,
                  beginningOfPassWriteIndex: timestampEntry.startQuery,
                  endOfPassWriteIndex: timestampEntry.endQuery,
                },
              }
            : undefined;
        // A compute usage scope is one dispatch. Adjacent dispatches may share
        // a pass even when a later dispatch reads an earlier dispatch's output.
        // Timestamp writes belong to a pass, so traced dispatches stay separate.
        if (computePassDescriptor !== undefined) this.endComputePass();
        if (this.pendingComputePass === null) {
          this.pendingComputePass = this.pendingEncoder.beginComputePass(computePassDescriptor);
          this.pendingComputePassCount += 1;
        }
        const compute = this.pendingComputePass;
        compute.setPipeline(pipeline);

        for (let i = 0; i < bufferArgIndices.length; ++i) {
          bindGroupEntries.push({
            binding: i,
            resource: {
              buffer: this.gpuBufferFromPtr(args[bufferArgIndices[i]])
            }
          });
        }

        const sizeOfI32 = 4;
        const bufBytes = (podArgIndices.length + 1) * sizeOfI32;
        const podArgRegion = this.allocateUniformRegion(bufBytes);
        const i32View = new Int32Array(podArgIndices.length + 1);
        const u32View = new Uint32Array(i32View.buffer);
        const f32View = new Float32Array(i32View.buffer);

        for (let i = 0; i < podArgIndices.length; ++i) {
          const value = args[podArgIndices[i]];
          const dtype = finfo.arg_types[podArgIndices[i]];
          if (dtype.startsWith("int")) {
            i32View[i] = value;
          } else if (dtype.startsWith("uint")) {
            u32View[i] = value;
          } else if (dtype.startsWith("float")) {
            f32View[i] = value;
          } else {
            throw Error("Unknown pod dtype " + dtype);
          }
        }
        // always pass in dim z launching grid size in
        u32View[podArgIndices.length] = packDimX;
        runtimeTraceEmit(
          "webgpu.queue.write_buffer",
          {
            bytes: i32View.buffer.byteLength,
            reason: "uniform_pod",
            function_name: finfo.name,
          },
          { level: "verbose" },
        );
        this.device.queue.writeBuffer(
          podArgRegion.buffer,
          podArgRegion.offset,
          i32View.buffer,
        );

        bindGroupEntries.push({
          binding: bufferArgIndices.length,
          resource: {
            buffer: podArgRegion.buffer,
            offset: podArgRegion.offset,
            size: i32View.buffer.byteLength
          }
        });

        const bindSetupStart = performance.now();
        const bindGroup = this.device.createBindGroup({
          layout: bindGroupLayout,
          entries: bindGroupEntries
        });
        runtimeTraceEmit(
          "webgpu.bind_group.setup",
          {
            function_name: finfo.name,
            bindings: bindGroupEntries.length,
            duration_ms: performance.now() - bindSetupStart,
          },
          { level: "verbose" },
        );
        compute.setBindGroup(0, bindGroup);

        compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);
        if (timestampEntry !== undefined) {
          this.pendingTimestampEntries.push(timestampEntry);
        }
        if (!this.batchComputePasses || computePassDescriptor !== undefined) {
          this.endComputePass();
        }
        runtimeTraceEmit(
          "webgpu.command_encode.end",
          {
            function_name: finfo.name,
            duration_ms: performance.now() - encodeStart,
            packed_workgroups_x: packDimX,
            workgroups_x: workDim[0],
            workgroups_y: workDim[1],
            workgroups_z: workDim[2],
            workgroup_size_x: workgroupSize[0],
            workgroup_size_y: workgroupSize[1],
            workgroup_size_z: workgroupSize[2],
            total_workgroups: totalWorkgroups,
            thread_invocations: threadInvocations,
          },
          { level: "verbose" },
        );

        // In debug mode, flush immediately so we can observe each submission.
        if (this.debugLogFinish) {
          this.flushCommands();
          const currCounter = this.shaderSubmitCounter;
          this.device.queue.onSubmittedWorkDone().then(() => {
            console.log("[" + currCounter + "][Debug] finish shader" + finfo.name);
          });
        }
        this.shaderSubmitCounter += 1;
        // Bound command-buffer growth for long device-resident loops. Submit
        // only after ending the compute pass; queue ordering preserves state
        // dependencies and permits uniform arenas to be reused safely.
        if (this.maxDispatchesPerSubmit > 0 &&
            this.pendingDispatchCount >= this.maxDispatchesPerSubmit) {
          this.flushCommands();
        }
      };
      return submitShader;
    };

    const shaderModule = this.device.createShaderModule({
      code: code,
      compilationHints: [
        {
          entryPoint: "main",
          layout: pipelineLayout
        }
      ]
    });

    if (asyncMode) {
      return this.device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: finfo.name
        }
      }).then((pipeline: GPUComputePipeline) => {
        return createShaderFunc(pipeline);
      });
    } else {
      const pipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: finfo.name
        }
      });
      return createShaderFunc(pipeline);
    }
  }

  /**
   * Get the device API according to its name
    * @param name The name of the API.
   * @returns The corresponding device api.
   */
  getDeviceAPI(name: string): Function {
    if (name == "deviceAllocDataSpace") {
      return (nbytes: number): GPUPointer => {
        return this.deviceAllocDataSpace(nbytes);
      };
    } else if (name == "deviceFreeDataSpace") {
      return (ptr: GPUPointer): void => {
        return this.deviceFreeDataSpace(ptr);
      };
    } else if (name == "deviceCopyToGPU") {
      return (
        from: Pointer,
        to: GPUPointer,
        toOffset: number,
        nbytes: number
      ): void => {
        this.deviceCopyToGPU(from, to, toOffset, nbytes);
      };
    } else if (name == "deviceCopyFromGPU") {
      return (
        from: GPUPointer,
        fromOffset: number,
        to: Pointer,
        nbytes: number
      ): void => {
        this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
      };
    } else if (name == "deviceCopyWithinGPU") {
      return (
        from: GPUPointer,
        fromOffset: number,
        to: Pointer,
        toOffset: number,
        nbytes: number
      ): void => {
        this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
      };
    } else {
      throw new Error("Unknown DeviceAPI function " + name);
    }
  }

  private requireSampledTokenReadbackRing(
    ringId: number,
  ): InternalSampledTokenReadbackRing {
    const ring = this.sampledTokenReadbackRings.get(ringId);
    if (ring === undefined || ring.disposed) {
      throw new Error(`Cannot find sampled-token readback ring ${ringId}.`);
    }
    return ring;
  }

  private throwIfSampledTokenReadbackRingFailed(
    ring: InternalSampledTokenReadbackRing,
  ): void {
    if (ring.fatalError !== undefined) {
      throw ring.fatalError;
    }
  }

  private findFreeSampledTokenReadbackSlot(
    ring: InternalSampledTokenReadbackRing,
  ): number {
    for (let i = 0; i < ring.slotCount; ++i) {
      const idx = (ring.nextSubmitCursor + i) % ring.slotCount;
      if (ring.slots[idx].state === "free") {
        ring.nextSubmitCursor = (idx + 1) % ring.slotCount;
        return idx;
      }
    }
    return -1;
  }

  private resetSampledTokenReadbackSlot(
    slot: InternalSampledTokenReadbackSlot,
  ): void {
    slot.state = "free";
    slot.batchSeq = 0;
    slot.submitSeq = 0;
    slot.step = undefined;
    slot.tokenCount = 0;
    slot.tokens = undefined;
    slot.pendingPromise = undefined;
  }

  private sampledTokenReadbackRingHasPending(
    ring: InternalSampledTokenReadbackRing,
  ): boolean {
    for (const slot of ring.slots) {
      if (slot.state === "pending") {
        return true;
      }
    }
    return false;
  }

  private drainReadySampledTokenReadbackBatches(
    ring: InternalSampledTokenReadbackRing,
  ): Array<SampledTokenReadbackBatch> {
    const ret: Array<SampledTokenReadbackBatch> = [];
    while (true) {
      const slotIdx = ring.readyByBatchSeq.get(ring.nextPollSeq);
      if (slotIdx === undefined) {
        break;
      }
      const slot = ring.slots[slotIdx];
      if (slot.state !== "ready" || slot.tokens === undefined) {
        break;
      }
      ret.push({
        batchSeq: slot.batchSeq,
        submitSeq: slot.submitSeq,
        tokenCount: slot.tokenCount,
        tokens: slot.tokens,
      });
      ring.readyByBatchSeq.delete(ring.nextPollSeq);
      ring.nextPollSeq += 1;
      this.resetSampledTokenReadbackSlot(slot);
    }
    return ret;
  }

  private notifySampledTokenReadbackRingWaiters(
    ring: InternalSampledTokenReadbackRing,
    error?: Error,
  ): void {
    if (ring.waiters.length === 0) {
      return;
    }
    const waiters = ring.waiters.splice(0, ring.waiters.length);
    for (const waiter of waiters) {
      if (error !== undefined) {
        waiter.reject(error);
      } else {
        waiter.resolve();
      }
    }
  }

  private trackPendingSampledTokenReadbackRead(readPromise: Promise<void>): void {
    let trackedPromise: Promise<void>;
    trackedPromise = readPromise.finally(() => {
      this.pendingSampledTokenReadbackReads.delete(trackedPromise);
    });
    this.pendingSampledTokenReadbackReads.add(trackedPromise);
  }

  // DeviceAPI
  private deviceAllocDataSpace(nbytes: number): GPUPointer {
    // WebGPU buffer copies and queue writes operate in four-byte units.
    const allocationBytes = Math.max(4, roundUpToFourBytes(nbytes));
    const buffer = tryCreateBuffer(this.device, {
      size: allocationBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.currAllocatedBytes += buffer.size;
    this.allAllocatedBytes += buffer.size;
    if (this.currAllocatedBytes > this.peakAllocatedBytes) {
      this.peakAllocatedBytes = this.currAllocatedBytes;
    }
    const ptr = this.attachToBufferTable(buffer);
    runtimeTraceEmit(
      "webgpu.temp_alloc",
      {
        ptr,
        bytes: nbytes,
        curr_allocated_bytes: this.currAllocatedBytes,
      },
      { level: "verbose" },
    );
    return ptr;
  }

  private deviceFreeDataSpace(ptr: GPUPointer): void {
    const idx = ptr;
    const buffer = this.bufferTable[idx];
    const deferDestroy = this.pendingEncoder !== null;
    this.bufferTable[idx] = undefined;
    assert(buffer !== undefined);
    this.bufferTableFreeId.push(idx);
    this.currAllocatedBytes -= buffer.size;
    runtimeTraceEmit(
      "webgpu.temp_free",
      {
        ptr,
        bytes: buffer.size,
        curr_allocated_bytes: this.currAllocatedBytes,
        deferred_destroy: deferDestroy,
      },
      { level: "verbose" },
    );
    if (deferDestroy) {
      this.pendingBufferDestroys.push(buffer);
      this.pendingBufferDestroyBytes += buffer.size;
      // A small dispatch count can still retain gigabytes of dead scratch
      // buffers. Submit their uses before destroying them, without waiting for
      // the GPU. Queue ordering and the normal flush path preserve dependencies.
      if (this.maxDeferredDestroyBytes > 0 &&
          this.pendingBufferDestroyBytes >= this.maxDeferredDestroyBytes) {
        this.flushCommands();
      }
    } else {
      buffer.destroy();
    }
  }

  private deviceCopyToGPU(
    from: Pointer,
    to: GPUPointer,
    toOffset: number,
    nbytes: number
  ): void {
    validateWebGPUCopyOffset(toOffset, "WebGPU destination offset");
    let rawBytes = this.memory.viewRawBytes(from, nbytes);
    if (rawBytes.length % 4 !== 0) {
      // writeBuffer requires length to be multiples of 4, so we pad here
      const toPad = 4 - rawBytes.length % 4;
      const padded = new Uint8Array(rawBytes.length + toPad);
      padded.set(rawBytes);
      rawBytes = padded;
      nbytes = nbytes + toPad;
    }
    this.writeRawBytesToBuffer(
      rawBytes,
      to,
      toOffset,
      nbytes,
      "device_copy_to_gpu",
    );
  }

  /**
   * Get a MAP_READ staging buffer from the pool, or create one if none fits.
   * Uses first-fit-by-size: returns the first pooled buffer >= nbytes.
   */
  private getOrCreateReadStagingBuffer(nbytes: number): GPUBuffer {
    for (let i = 0; i < this.readStagingBufferPool.length; i++) {
      if (this.readStagingBufferPool[i].size >= nbytes) {
        const entry = this.readStagingBufferPool.splice(i, 1)[0];
        return entry.buffer;
      }
    }
    return tryCreateBuffer(this.device, {
      size: nbytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Return a MAP_READ staging buffer to the pool for reuse.
   * Evicts the smallest buffer if the pool is full.
   */
  private recycleReadStagingBuffer(buf: GPUBuffer): void {
    buf.unmap();
    if (this.readStagingBufferPool.length >= this.maxReadStagingBuffers) {
      // Evict smallest buffer to make room
      let minIdx = 0;
      for (let i = 1; i < this.readStagingBufferPool.length; i++) {
        if (this.readStagingBufferPool[i].size < this.readStagingBufferPool[minIdx].size) {
          minIdx = i;
        }
      }
      this.readStagingBufferPool.splice(minIdx, 1)[0].buffer.destroy();
    }
    this.readStagingBufferPool.push({ buffer: buf, size: buf.size });
  }

  private deviceCopyFromGPU(
    from: GPUPointer,
    fromOffset: number,
    to: Pointer,
    nbytes: number
  ): void {
    validateWebGPUCopyOffset(fromOffset, "WebGPU source offset");
    // Flush batched compute passes before the readback copy.
    this.flushCommands();
    if (nbytes == 0) {
      this.memory.storeRawBytes(to, new Uint8Array(0));
      return;
    }
    const copyBytes = roundUpToFourBytes(nbytes);
    const gpuTemp = this.getOrCreateReadStagingBuffer(copyBytes);
    const copyStep = runtimeTraceCurrentStep();
    runtimeTraceEmit(
      "webgpu.staging.copy.start",
      {
        bytes: nbytes,
        from_ptr: from,
      },
      { level: "major", step: copyStep },
    );

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(
      this.gpuBufferFromPtr(from),
      fromOffset,
      gpuTemp,
      0,
      copyBytes
    );
    const copyCommands = copyEncoder.finish();
    const submitSeq = this.traceSubmitCounter + 1;
    this.device.queue.submit([copyCommands]);
    this.traceSubmitCounter = submitSeq;
    runtimeTraceEmit(
      "webgpu.queue.submit",
      {
        submit_seq: submitSeq,
        dispatches: 0,
        copy_bytes: nbytes,
        reason: "gpu_to_cpu_copy",
      },
      { level: "major", step: copyStep },
    );

    runtimeTraceEmit(
      "webgpu.map_async.start",
      {
        bytes: nbytes,
        submit_seq: submitSeq,
      },
      { level: "major", step: copyStep },
    );
    const readPromise = gpuTemp.mapAsync(GPUMapMode.READ)
      .then(() => {
        const data = gpuTemp.getMappedRange(0, copyBytes);
        this.memory.storeRawBytes(to, new Uint8Array(data).subarray(0, nbytes));
        this.recycleReadStagingBuffer(gpuTemp);
        runtimeTraceEmit(
          "webgpu.map_async.end",
          {
            bytes: nbytes,
            submit_seq: submitSeq,
          },
          { level: "major", step: copyStep },
        );
        runtimeTraceEmit(
          "webgpu.staging.copy.end",
          {
            bytes: nbytes,
            from_ptr: from,
          },
          { level: "major", step: copyStep },
        );
      })
      .catch((err) => {
        runtimeTraceEmit(
          "webgpu.map_async.error",
          {
            bytes: nbytes,
            submit_seq: submitSeq,
            message: String(err),
          },
          { level: "major", step: copyStep },
        );
        throw err;
      });
    // Chain with any existing pending read so sync() awaits all of them.
    this.pendingGPUToCPUCopy = this.pendingGPUToCPUCopy
      ? this.pendingGPUToCPUCopy.then(() => readPromise)
      : readPromise;
    this.pendingGPUToCPUCopyIsQueueTail = true;
  }

  private deviceCopyWithinGPU(
    from: GPUPointer,
    fromOffset: number,
    to: Pointer,
    toOffset: number,
    nbytes: number
  ): void {
    // Keep copies in the same command encoder as compute dispatches. Command
    // ordering within the encoder preserves dependencies, while a later
    // readback, CPU write, deallocation, or sync provides the flush point.
    this.endComputePass();
    if (!this.pendingEncoder) {
      this.pendingEncoder = this.device.createCommandEncoder();
    }
    this.pendingEncoder.copyBufferToBuffer(
      this.gpuBufferFromPtr(from),
      fromOffset,
      this.gpuBufferFromPtr(to),
      toOffset,
      nbytes
    );
    this.pendingGPUToGPUCopyCount += 1;
    this.pendingGPUToGPUCopyBytes += nbytes;
  }

  private gpuBufferFromPtr(ptr: GPUPointer): GPUBuffer {
    const buffer = this.bufferTable[ptr];
    assert(buffer !== undefined);
    return buffer;
  }

  private endComputePass(): void {
    if (this.pendingComputePass !== null) {
      this.pendingComputePass.end();
      this.pendingComputePass = null;
    }
  }

  private attachToBufferTable(buffer: GPUBuffer): GPUPointer {
    if (this.bufferTableFreeId.length != 0) {
      const idx = this.bufferTableFreeId.pop() as number;
      this.bufferTable[idx] = buffer;
      return idx;
    } else {
      const idx = this.bufferTable.length;
      this.bufferTable.push(buffer);
      return idx;
    }
  }
}
