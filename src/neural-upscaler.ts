import ortWasmUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url";

export type NeuralUpscaleMethod = "websr" | "fsrcnn";

type WebSrModule = typeof import("@websr/websr");
type OrtModule = typeof import("onnxruntime-web/webgpu");

const INPUT_SIZE = 320;
const MODEL_OUTPUT_SIZE = 640;
const baseUrl = import.meta.env.BASE_URL;

let inferenceQueue = Promise.resolve();
let webGpuDevicePromise: Promise<GPUDevice> | undefined;
let webSrModulePromise: Promise<WebSrModule> | undefined;
let webSrWeightsPromise: Promise<unknown> | undefined;
let ortModulePromise: Promise<OrtModule> | undefined;
let fsrcnnSessionPromise: Promise<import("onnxruntime-web").InferenceSession> | undefined;

async function getWebGpuDevice() {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
  webGpuDevicePromise ??= (async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found");
    return adapter.requestDevice();
  })();
  return webGpuDevicePromise;
}

async function getWebSrResources() {
  webSrModulePromise ??= import("@websr/websr");
  webSrWeightsPromise ??= fetch(`${baseUrl}models/websr-cnn-2x-s-rl.json`).then((response) => {
    if (!response.ok) throw new Error(`WebSR weights failed to load (${response.status})`);
    return response.json() as Promise<unknown>;
  });
  const [module, weights, device] = await Promise.all([webSrModulePromise, webSrWeightsPromise, getWebGpuDevice()]);
  return { WebSR: module.default, weights, device };
}

async function getFsrcnnResources() {
  ortModulePromise ??= import("onnxruntime-web/webgpu");
  const ort = await ortModulePromise;
  ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, location.href).href };
  ort.env.wasm.numThreads = 1;
  fsrcnnSessionPromise ??= ort.InferenceSession.create(`${baseUrl}models/fsrcnn-x2-320.onnx`, {
    executionProviders: ["webgpu"],
    // ORT's WebGPU graph/lifetime optimizer currently destroys an intermediate
    // buffer too early for this model's DepthToSpace tail. Keep the small graph
    // sequential; inference remains entirely on WebGPU.
    graphOptimizationLevel: "disabled",
    executionMode: "sequential",
    enableMemPattern: false
  });
  return { ort, session: await fsrcnnSessionPromise };
}

export class NeuralUpscaler {
  private inputCanvas = document.createElement("canvas");
  private inputContext = this.inputCanvas.getContext("2d", { alpha: false, willReadFrequently: true })!;
  private webSrCanvas = document.createElement("canvas");
  private fsrcnnCanvas = document.createElement("canvas");
  private fsrcnnContext = this.fsrcnnCanvas.getContext("2d", { alpha: false, willReadFrequently: true })!;
  private results = new Map<NeuralUpscaleMethod, HTMLCanvasElement>();
  private failed = new Set<NeuralUpscaleMethod>();
  private webSr?: InstanceType<WebSrModule["default"]>;
  private busy = false;
  private lastCompletionAt = new Map<NeuralUpscaleMethod, number>();
  private smoothedFps = new Map<NeuralUpscaleMethod, number>();
  private requestedMethod?: NeuralUpscaleMethod;

  constructor(
    private reportStatus: (message: string, error?: boolean) => void,
    private reportFps: (fps: number, method: NeuralUpscaleMethod) => void
  ) {
    this.inputCanvas.width = this.inputCanvas.height = INPUT_SIZE;
    this.fsrcnnCanvas.width = this.fsrcnnCanvas.height = MODEL_OUTPUT_SIZE;
    this.inputContext.imageSmoothingEnabled = true;
    this.inputContext.imageSmoothingQuality = "high";
    this.fsrcnnContext.imageSmoothingEnabled = true;
    this.fsrcnnContext.imageSmoothingQuality = "high";
  }

  getResult(method: NeuralUpscaleMethod) {
    return this.results.get(method);
  }

  request(method: NeuralUpscaleMethod, captureSource: () => CanvasImageSource, flipY = false) {
    if (this.busy || this.failed.has(method)) return;
    this.busy = true;
    this.requestedMethod = method;
    const source = captureSource();
    this.inputContext.save();
    this.inputContext.setTransform(1, 0, 0, flipY ? -1 : 1, 0, flipY ? INPUT_SIZE : 0);
    this.inputContext.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
    this.inputContext.restore();
    if (!this.results.has(method)) this.reportStatus(method === "websr" ? "Loading WebSR…" : "Loading FSRCNN…");

    const work = async () => {
      const output = method === "websr" ? await this.runWebSr() : await this.runFsrcnn();
      const stable = this.results.get(method) ?? document.createElement("canvas");
      stable.width = MODEL_OUTPUT_SIZE;
      stable.height = MODEL_OUTPUT_SIZE;
      const context = stable.getContext("2d", { alpha: false })!;
      context.drawImage(output, 0, 0, MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE);
      this.results.set(method, stable);
      const completedAt = performance.now();
      const previousCompletion = this.lastCompletionAt.get(method);
      if (previousCompletion) {
        const instantaneousFps = 1000 / Math.max(1, completedAt - previousCompletion);
        const previousFps = this.smoothedFps.get(method) ?? instantaneousFps;
        const fps = previousFps * .75 + instantaneousFps * .25;
        this.smoothedFps.set(method, fps);
        this.reportFps(fps, method);
      }
      this.lastCompletionAt.set(method, completedAt);
      if (this.requestedMethod === method) this.reportStatus(method === "websr" ? "WebSR · 320→640" : "FSRCNN · 320→640");
    };

    inferenceQueue = inferenceQueue.then(work).catch((error: unknown) => {
      this.failed.add(method);
      console.error(`${method} upscaling failed`, error);
      this.reportStatus(`${method === "websr" ? "WebSR" : "FSRCNN"} unavailable · using bilinear`, true);
    }).finally(() => {
      this.busy = false;
    });
  }

  private async runWebSr() {
    if (!this.webSr) {
      const { WebSR, weights, device } = await getWebSrResources();
      this.webSr = new WebSR({
        canvas: this.webSrCanvas,
        weights,
        network_name: "anime4k/cnn-2x-s",
        gpu: device,
        resolution: { width: INPUT_SIZE, height: INPUT_SIZE }
      });
    }
    const bitmap = await createImageBitmap(this.inputCanvas);
    try {
      await this.webSr.render(bitmap);
    } finally {
      bitmap.close();
    }
    return this.webSrCanvas;
  }

  private async runFsrcnn() {
    const { ort, session } = await getFsrcnnResources();
    const pixels = this.inputContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const luminance = new Float32Array(INPUT_SIZE * INPUT_SIZE);
    for (let i = 0, pixel = 0; i < luminance.length; i++, pixel += 4) {
      luminance[i] = (pixels[pixel] * .299 + pixels[pixel + 1] * .587 + pixels[pixel + 2] * .114) / 255;
    }
    const input = new ort.Tensor("float32", luminance, [1, INPUT_SIZE, INPUT_SIZE, 1]);
    const outputs = await session.run({ [session.inputNames[0]]: input });
    const enhancedLuminance = outputs[session.outputNames[0]].data as Float32Array;

    this.fsrcnnContext.drawImage(this.inputCanvas, 0, 0, MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE);
    const output = this.fsrcnnContext.getImageData(0, 0, MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE);
    for (let i = 0, pixel = 0; i < enhancedLuminance.length; i++, pixel += 4) {
      const oldY = output.data[pixel] * .299 + output.data[pixel + 1] * .587 + output.data[pixel + 2] * .114;
      const newY = Math.max(0, Math.min(255, enhancedLuminance[i] * 255));
      const difference = newY - oldY;
      output.data[pixel] += difference;
      output.data[pixel + 1] += difference;
      output.data[pixel + 2] += difference;
    }
    this.fsrcnnContext.putImageData(output, 0, 0);
    return this.fsrcnnCanvas;
  }
}

