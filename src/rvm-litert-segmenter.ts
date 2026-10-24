import {
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  supportsFeature,
  type CompiledModel
} from "@litertjs/core";
import type { RvmMatte } from "./rvm-segmenter";
import { RvmGpuInputPipeline } from "./rvm-gpu-input-pipeline";

const WASM_URL = `${import.meta.env.BASE_URL}litert-wasm/`;
const PROFILES = {
  wide: {
    width: 512,
    height: 288,
    modelUrl: `${import.meta.env.BASE_URL}models/rvm_mobilenetv3_512x288_fp16.tflite`
  },
  square: {
    width: 256,
    height: 256,
    modelUrl: `${import.meta.env.BASE_URL}models/rvm_mobilenetv3_256x256_fp16.tflite`
  }
};
export type RvmLiteRtProfile = keyof typeof PROFILES;

let runtimePromise: Promise<void> | undefined;
const modelPromises = new Map<RvmLiteRtProfile, Promise<CompiledModel>>();

async function modelResource(profileName: RvmLiteRtProfile) {
  if (!isWebGPUSupported()) throw new Error("LiteRT WebGPU is unavailable in this browser");
  runtimePromise ??= (async () => {
    const jspi = await supportsFeature("jspi");
    await loadLiteRt(WASM_URL, { jspi, threads: false });
  })();
  await runtimePromise;
  let modelPromise = modelPromises.get(profileName);
  if (!modelPromise) {
    modelPromise = loadAndCompile(PROFILES[profileName].modelUrl, {
      accelerator: "webgpu",
      gpuOptions: { precision: "fp16" }
    });
    modelPromises.set(profileName, modelPromise);
  }
  return modelPromise;
}

function deleteTensors(tensors: Record<string, Tensor> | undefined) {
  if (!tensors) return;
  for (const tensor of Object.values(tensors)) tensor.delete();
}

function recurrentShapes(width: number, height: number): Record<string, number[]> {
  return {
    r1i: [1, height / 2, width / 2, 16],
    r2i: [1, height / 4, width / 4, 20],
    r3i: [1, height / 8, width / 8, 40],
    r4i: [1, height / 16, width / 16, 64]
  };
}

function zeroState(width: number, height: number) {
  return Object.fromEntries(Object.entries(recurrentShapes(width, height)).map(([name, shape]) => [
    name,
    new Tensor(new Float32Array(shape.reduce((size, value) => size * value)), shape)
  ]));
}

export class RvmLiteRtSegmenter {
  private inputCanvas = document.createElement("canvas");
  private inputContext = this.inputCanvas.getContext("2d", { alpha: false, willReadFrequently: true })!;
  private recurrent = new WeakMap<object, Record<string, Tensor>>();
  private gpuInputs = new WeakMap<object, RvmGpuInputPipeline>();
  private compiled?: CompiledModel;
  private profile: (typeof PROFILES)[RvmLiteRtProfile];
  private gpuInputAvailable = true;

  constructor(private profileName: RvmLiteRtProfile) {
    this.profile = PROFILES[profileName];
    this.inputCanvas.width = this.profile.width;
    this.inputCanvas.height = this.profile.height;
  }

  get isFullyAccelerated() {
    return this.compiled?.isFullyAccelerated;
  }

  async load() {
    this.compiled = await modelResource(this.profileName);
  }

  reset(source: object) {
    deleteTensors(this.recurrent.get(source));
    this.recurrent.delete(source);
    this.gpuInputs.get(source)?.destroy();
    this.gpuInputs.delete(source);
  }

  private createCpuInput(frame: CanvasImageSource, width: number, height: number, matteSize: number, squareOffsetX: number) {
    this.inputContext.fillStyle = "black";
    this.inputContext.fillRect(0, 0, width, height);
    this.inputContext.drawImage(frame, squareOffsetX, 0, matteSize, matteSize);
    const rgba = this.inputContext.getImageData(0, 0, width, height).data;
    const rgb = new Float32Array(width * height * 3);
    for (let sourceIndex = 0, targetIndex = 0; sourceIndex < rgba.length; sourceIndex += 4) {
      rgb[targetIndex++] = rgba[sourceIndex] / 255;
      rgb[targetIndex++] = rgba[sourceIndex + 1] / 255;
      rgb[targetIndex++] = rgba[sourceIndex + 2] / 255;
    }
    return new Tensor(rgb, [1, height, width, 3]);
  }

  private createGpuInput(source: object, frame: CanvasImageSource, width: number, height: number) {
    const dimensions = frame instanceof HTMLVideoElement
      ? [frame.videoWidth, frame.videoHeight]
      : [Number((frame as { width?: number }).width), Number((frame as { height?: number }).height)];
    const [sourceWidth, sourceHeight] = dimensions;
    if (!sourceWidth || !sourceHeight) throw new Error("RVM source dimensions are unavailable");
    let pipeline = this.gpuInputs.get(source);
    if (!pipeline?.matches(sourceWidth, sourceHeight)) {
      pipeline?.destroy();
      pipeline = RvmGpuInputPipeline.create(sourceWidth, sourceHeight, width, height);
      this.gpuInputs.set(source, pipeline);
    }
    return pipeline.createInputTensor(frame as GPUCopyExternalImageSource);
  }

  async segment(source: object, frame: CanvasImageSource): Promise<RvmMatte> {
    const model = this.compiled ?? await modelResource(this.profileName);
    this.compiled = model;
    const { width, height } = this.profile;
    const matteSize = Math.min(width, height);
    const squareOffsetX = (width - matteSize) / 2;

    // TypeGPU writes normalized RGB directly into a persistent GPUBuffer that
    // LiteRT consumes on the same GPUDevice. Keep the CPU conversion as a
    // compatibility fallback if the shared-device path is unavailable.
    let input: Tensor;
    if (this.gpuInputAvailable) {
      try {
        input = this.createGpuInput(source, frame, width, height);
      } catch (error) {
        this.gpuInputAvailable = false;
        this.gpuInputs.get(source)?.destroy();
        this.gpuInputs.delete(source);
        console.warn("RVM GPU preprocessing unavailable; using CPU pixels", error);
        input = this.createCpuInput(frame, width, height, matteSize, squareOffsetX);
      }
    } else {
      input = this.createCpuInput(frame, width, height, matteSize, squareOffsetX);
    }

    const previous = this.recurrent.get(source) ?? zeroState(width, height);
    let outputs: Record<string, Tensor>;
    try {
      outputs = await model.run({ src: input, ...previous });
    } finally {
      input.delete();
    }

    this.recurrent.set(source, {
      r1i: outputs.r1o,
      r2i: outputs.r2o,
      r3i: outputs.r3o,
      r4i: outputs.r4o
    });
    deleteTensors(previous);
    outputs.fgr.delete();

    const alphaOnCpu = await outputs.pha.moveTo("wasm");
    const confidence = alphaOnCpu.toTypedArray() as Float32Array;
    const alpha = new Uint8Array(matteSize * matteSize);
    for (let y = 0; y < matteSize; y++) {
      const sourceRow = y * width + squareOffsetX;
      const targetRow = y * matteSize;
      for (let x = 0; x < matteSize; x++) {
        alpha[targetRow + x] = Math.round(Math.max(0, Math.min(1, confidence[sourceRow + x])) * 255);
      }
    }
    alphaOnCpu.delete();
    return { alpha, width: matteSize, height: matteSize };
  }
}
