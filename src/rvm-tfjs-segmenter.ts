import type { RvmMatte } from "./rvm-segmenter";

type TfModule = typeof import("@tensorflow/tfjs");
type TfTensor = import("@tensorflow/tfjs").Tensor;
type TfGraphModel = import("@tensorflow/tfjs").GraphModel;

const INPUT_SIZE = 256;
const MODEL_URL = `${import.meta.env.BASE_URL}models/rvm-tfjs/model.json`;
const OUTPUTS = ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"];

let tfPromise: Promise<TfModule> | undefined;
let modelPromise: Promise<TfGraphModel> | undefined;

async function resources() {
  tfPromise ??= (async () => {
    const tf = await import("@tensorflow/tfjs");
    await import("@tensorflow/tfjs-backend-webgpu");
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
    if (!(await tf.setBackend("webgpu"))) throw new Error("TensorFlow.js could not initialize WebGPU");
    await tf.ready();
    return tf;
  })();
  const tf = await tfPromise;
  modelPromise ??= tf.loadGraphModel(MODEL_URL);
  return { tf, model: await modelPromise };
}

function dispose(tensors: TfTensor[] | undefined) {
  tensors?.forEach((tensor) => tensor.dispose());
}

export class RvmTfjsSegmenter {
  private inputCanvas = document.createElement("canvas");
  private inputContext = this.inputCanvas.getContext("2d", { alpha: false })!;
  private recurrent = new WeakMap<object, TfTensor[]>();

  constructor() {
    this.inputCanvas.width = this.inputCanvas.height = INPUT_SIZE;
  }

  async load() {
    await resources();
  }

  reset(source: object) {
    dispose(this.recurrent.get(source));
    this.recurrent.delete(source);
  }

  async segment(source: object, frame: CanvasImageSource): Promise<RvmMatte> {
    const { tf, model } = await resources();
    this.inputContext.drawImage(frame, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const previous = this.recurrent.get(source);
    const states = previous ?? Array.from({ length: 4 }, () => tf.scalar(0));
    const sourceTensor = tf.tidy(() => tf.browser.fromPixels(this.inputCanvas, 3).expandDims(0).div(255));
    const ratioTensor = tf.scalar(1);
    let outputs: TfTensor[];
    try {
      outputs = await model.executeAsync({
        src: sourceTensor,
        r1i: states[0],
        r2i: states[1],
        r3i: states[2],
        r4i: states[3],
        downsample_ratio: ratioTensor
      }, OUTPUTS) as TfTensor[];
    } finally {
      sourceTensor.dispose();
      ratioTensor.dispose();
    }

    const [foreground, matte, ...next] = outputs;
    this.recurrent.set(source, next);
    dispose(states);
    const confidence = await matte.data();
    const height = matte.shape[1] ?? INPUT_SIZE;
    const width = matte.shape[2] ?? INPUT_SIZE;
    const alpha = new Uint8Array(confidence.length);
    for (let i = 0; i < confidence.length; i++) alpha[i] = Math.round(Math.max(0, Math.min(1, Number(confidence[i]))) * 255);
    foreground.dispose();
    matte.dispose();
    return { alpha, width, height };
  }
}
