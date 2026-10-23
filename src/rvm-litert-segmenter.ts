import {
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  supportsFeature,
  type CompiledModel
} from "@litertjs/core";
import type { RvmMatte } from "./rvm-segmenter";

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
  private compiled?: CompiledModel;
  private profile: (typeof PROFILES)[RvmLiteRtProfile];

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
  }

  async segment(source: object, frame: CanvasImageSource): Promise<RvmMatte> {
    const model = this.compiled ?? await modelResource(this.profileName);
    this.compiled = model;
    const { width, height } = this.profile;
    const matteSize = Math.min(width, height);
    const squareOffsetX = (width - matteSize) / 2;

    // Our renderer consumes a square matte. Preserve that geometry inside the
    // wide LiteRT graph, then crop the square alpha region back out. The 256
    // profile is already square, so this draw and crop are both one-to-one.
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

    const input = new Tensor(rgb, [1, height, width, 3]);
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
