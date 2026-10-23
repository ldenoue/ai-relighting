import ortWasmUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";

type OrtModule = typeof import("onnxruntime-web/webgpu");
type OrtTensor = import("onnxruntime-web").Tensor;

export type RvmMatte = {
  alpha: Uint8Array;
  width: number;
  height: number;
};

const INPUT_SIZE = 256;
const MODEL_URL = `${import.meta.env.BASE_URL}models/rvm-mobilenetv3-fp32.onnx`;

let ortPromise: Promise<OrtModule> | undefined;
let sessionPromise: Promise<import("onnxruntime-web").InferenceSession> | undefined;

async function resources() {
  ortPromise ??= import("onnxruntime-web/webgpu");
  const ort = await ortPromise;
  ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, location.href).href };
  ort.env.wasm.numThreads = 1;
  sessionPromise ??= ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    executionMode: "sequential",
    enableMemPattern: false
  });
  return { ort, session: await sessionPromise };
}

function dispose(tensors: OrtTensor[] | undefined) {
  tensors?.forEach((tensor) => tensor.dispose());
}

export class RvmSegmenter {
  private inputCanvas = document.createElement("canvas");
  private inputContext = this.inputCanvas.getContext("2d", { alpha: false, willReadFrequently: true })!;
  private recurrent = new WeakMap<object, OrtTensor[]>();

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
    const { ort, session } = await resources();
    this.inputContext.drawImage(frame, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = this.inputContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const planeSize = INPUT_SIZE * INPUT_SIZE;
    const rgb = new Float32Array(planeSize * 3);
    for (let i = 0, pixel = 0; i < planeSize; i++, pixel += 4) {
      rgb[i] = pixels[pixel] / 255;
      rgb[planeSize + i] = pixels[pixel + 1] / 255;
      rgb[planeSize * 2 + i] = pixels[pixel + 2] / 255;
    }

    const previous = this.recurrent.get(source);
    const states = previous ?? Array.from({ length: 4 }, () => new ort.Tensor("float32", new Float32Array(1), [1, 1, 1, 1]));
    const sourceTensor = new ort.Tensor("float32", rgb, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    // RVM's commonly documented 0.25 ratio targets HD frames (roughly a
    // 270px recurrent pass). At our 256px input it would collapse inference
    // to only 64px and can erase the subject entirely, so retain full detail.
    const ratioTensor = new ort.Tensor("float32", new Float32Array([1]), [1]);
    let outputs: Awaited<ReturnType<typeof session.run>>;
    try {
      outputs = await session.run({
        src: sourceTensor,
        r1i: states[0],
        r2i: states[1],
        r3i: states[2],
        r4i: states[3],
        downsample_ratio: ratioTensor
      });
    } finally {
      sourceTensor.dispose();
      ratioTensor.dispose();
    }
    const next = [outputs.r1o, outputs.r2o, outputs.r3o, outputs.r4o];
    this.recurrent.set(source, next);
    dispose(states);

    const confidence = outputs.pha.data as Float32Array;
    const width = outputs.pha.dims[3] ?? INPUT_SIZE;
    const height = outputs.pha.dims[2] ?? INPUT_SIZE;
    const alpha = new Uint8Array(confidence.length);
    for (let i = 0; i < confidence.length; i++) alpha[i] = Math.round(Math.max(0, Math.min(1, confidence[i])) * 255);
    outputs.fgr?.dispose();
    outputs.pha.dispose();
    return { alpha, width, height };
  }
}
