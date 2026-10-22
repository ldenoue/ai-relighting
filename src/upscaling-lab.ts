import { ImageSpatialUpscaler, type SpatialImageMethod } from "./image-spatial-upscaler";
import { NeuralUpscaler, type NeuralUpscaleMethod } from "./neural-upscaler";
import "./upscaling-style.css";

type LabMethod = SpatialImageMethod | NeuralUpscaleMethod;

const methods: Array<{ id: LabMethod; name: string; detail: string }> = [
  { id: "bilinear", name: "Bilinear", detail: "Browser sampling baseline" },
  { id: "lanczos2", name: "Lanczos2", detail: "16-tap reconstruction" },
  { id: "fsr1", name: "FSR 1.0", detail: "EASU edge-directed reconstruction" },
  { id: "anime4k", name: "Anime4K Original", detail: "Sobel-guided line refinement" },
  { id: "websr", name: "WebSR CNN-S", detail: "WebGPU neural 2× model" },
  { id: "fsrcnn", name: "FSRCNN ×2", detail: "ONNX/WebGPU luminance model" }
];

const app = document.querySelector<HTMLDivElement>("#upscaling-app")!;
app.innerHTML = `
  <main class="lab-shell">
    <header class="lab-header">
      <div>
        <a class="back-link" href="./">← Relight Lab</a>
        <h1>Upscaling Lab</h1>
        <p>Downsample one image, then reconstruct it with every browser upscaler.</p>
      </div>
      <label class="rcas-toggle"><input id="lab-rcas" type="checkbox" checked><span></span>RCAS final pass</label>
    </header>

    <input id="image-input" type="file" accept="image/*" hidden>
    <button class="image-drop" id="image-drop" type="button">
      <span class="drop-icon">＋</span>
      <strong>Choose an image</strong>
      <small>Drop an image here or click to browse</small>
    </button>

    <section class="source-panel" id="source-panel" hidden>
      <div class="source-preview"><canvas id="source-canvas"></canvas></div>
      <div class="source-info">
        <span class="eyebrow">Test source</span>
        <strong id="source-name"></strong>
        <div class="dimension-list">
          <span>Original <b id="original-size"></b></span>
          <span>Reduced input <b id="input-size"></b></span>
          <span>Comparison output <b id="output-size"></b></span>
        </div>
      </div>
      <div class="reduction-controls">
        <span>Input size</span>
        <div class="reduction-buttons" role="group" aria-label="Input image size">
          <button type="button" data-scale="5">5%</button>
          <button type="button" data-scale="10">10%</button>
          <button type="button" data-scale="25">25%</button>
          <button class="active" type="button" data-scale="50">50%</button>
          <button type="button" data-scale="75">75%</button>
          <button type="button" data-scale="100">100%</button>
        </div>
        <button class="run-all" id="run-all" type="button">Run all algorithms</button>
      </div>
    </section>

    <section class="results" id="results" hidden>
      ${methods.map((method) => `
        <article class="result-card" data-method="${method.id}">
          <header><div><strong>${method.name}</strong><small>${method.detail}</small></div><output>Waiting</output></header>
          <div class="result-frame"><canvas></canvas></div>
          <footer><span class="result-dimensions">—</span><button type="button" class="download-result">Download PNG</button></footer>
        </article>`).join("")}
    </section>
  </main>`;

const fileInput = document.querySelector<HTMLInputElement>("#image-input")!;
const dropTarget = document.querySelector<HTMLButtonElement>("#image-drop")!;
const sourcePanel = document.querySelector<HTMLElement>("#source-panel")!;
const results = document.querySelector<HTMLElement>("#results")!;
const sourceCanvas = document.querySelector<HTMLCanvasElement>("#source-canvas")!;
const sourceContext = sourceCanvas.getContext("2d", { alpha: false })!;
const rcasToggle = document.querySelector<HTMLInputElement>("#lab-rcas")!;

let sourceBitmap: ImageBitmap | undefined;
let sourceFileName = "image";
let inputScale = 50;
let runGeneration = 0;
const OUTPUT_EDGE = 512;

const outputCanvases = new Map<LabMethod, HTMLCanvasElement>();
const spatialUpscalers = new Map<SpatialImageMethod, ImageSpatialUpscaler>();

for (const method of methods) {
  const card = document.querySelector<HTMLElement>(`[data-method="${method.id}"]`)!;
  const canvas = card.querySelector<HTMLCanvasElement>("canvas")!;
  outputCanvases.set(method.id, canvas);
  if (method.id !== "websr" && method.id !== "fsrcnn") {
    spatialUpscalers.set(method.id, new ImageSpatialUpscaler(canvas));
  }
  card.querySelector<HTMLButtonElement>(".download-result")!.addEventListener("click", () => {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${sourceFileName.replace(/\.[^.]+$/, "")}-${method.id}.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }, "image/png");
  });
}

function setCardStatus(method: LabMethod, message: string, error = false) {
  const card = document.querySelector<HTMLElement>(`[data-method="${method}"]`)!;
  const output = card.querySelector<HTMLOutputElement>("output")!;
  output.value = message;
  output.classList.toggle("error", error);
  card.classList.toggle("working", message.includes("…"));
}

function getOutputDimensions() {
  if (!sourceBitmap) return { width: OUTPUT_EDGE, height: OUTPUT_EDGE };
  const scale = OUTPUT_EDGE / Math.max(sourceBitmap.width, sourceBitmap.height);
  return {
    width: Math.max(1, Math.round(sourceBitmap.width * scale)),
    height: Math.max(1, Math.round(sourceBitmap.height * scale))
  };
}

function updateSourceCanvas() {
  if (!sourceBitmap) return;
  const output = getOutputDimensions();
  const scale = inputScale / 100;
  sourceCanvas.width = Math.max(1, Math.round(output.width * scale));
  sourceCanvas.height = Math.max(1, Math.round(output.height * scale));
  sourceContext.imageSmoothingEnabled = true;
  sourceContext.imageSmoothingQuality = "high";
  sourceContext.drawImage(sourceBitmap, 0, 0, sourceCanvas.width, sourceCanvas.height);
  document.querySelector("#input-size")!.textContent = `${sourceCanvas.width} × ${sourceCanvas.height}`;
  document.querySelector("#output-size")!.textContent = `${output.width} × ${output.height}`;
}

function makeNeuralInput() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 320;
  const context = canvas.getContext("2d", { alpha: false })!;
  context.fillStyle = "#000";
  context.fillRect(0, 0, 320, 320);
  const fit = Math.min(320 / sourceCanvas.width, 320 / sourceCanvas.height);
  const width = sourceCanvas.width * fit;
  const height = sourceCanvas.height * fit;
  const x = (320 - width) / 2;
  const y = (320 - height) / 2;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(sourceCanvas, x, y, width, height);
  return { canvas, x, y, width, height };
}

async function runNeural(method: NeuralUpscaleMethod, generation: number) {
  if (!sourceBitmap) return;
  const cardCanvas = outputCanvases.get(method)!;
  const status = { message: "Preparing…", error: false };
  const upscaler = new NeuralUpscaler(
    (message, error = false) => {
      status.message = message;
      status.error = error;
      if (generation === runGeneration) setCardStatus(method, error ? message : `${message}…`, error);
    },
    () => {}
  );
  const neuralInput = makeNeuralInput();
  const startedAt = performance.now();
  upscaler.request(method, () => neuralInput.canvas);
  while (generation === runGeneration) {
    const enhanced = upscaler.getResult(method);
    if (enhanced) {
      const output = getOutputDimensions();
      const fitted = document.createElement("canvas");
      fitted.width = output.width;
      fitted.height = output.height;
      const context = fitted.getContext("2d", { alpha: false })!;
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(
        enhanced,
        neuralInput.x * 2,
        neuralInput.y * 2,
        neuralInput.width * 2,
        neuralInput.height * 2,
        0,
        0,
        fitted.width,
        fitted.height
      );

      const postCanvas = document.createElement("canvas");
      const post = new ImageSpatialUpscaler(postCanvas);
      post.run("bilinear", fitted, fitted.width, fitted.height, fitted.width, fitted.height, rcasToggle.checked);
      cardCanvas.width = fitted.width;
      cardCanvas.height = fitted.height;
      cardCanvas.getContext("2d", { alpha: false })!.drawImage(postCanvas, 0, 0);
      const elapsed = performance.now() - startedAt;
      setCardStatus(method, `${elapsed.toFixed(0)} ms`);
      document.querySelector<HTMLElement>(`[data-method="${method}"] .result-dimensions`)!.textContent = `${fitted.width} × ${fitted.height}`;
      return;
    }
    if (status.error) return;
    if (performance.now() - startedAt > 120000) {
      setCardStatus(method, "Timed out", true);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

async function runAll() {
  if (!sourceBitmap) return;
  const generation = ++runGeneration;
  updateSourceCanvas();
  const { width: outputWidth, height: outputHeight } = getOutputDimensions();
  const applyRcas = rcasToggle.checked;

  for (const method of methods) {
    setCardStatus(method.id, "Running…");
    document.querySelector<HTMLElement>(`[data-method="${method.id}"] .result-dimensions`)!.textContent = `${outputWidth} × ${outputHeight}`;
  }
  await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));

  for (const method of ["bilinear", "lanczos2", "fsr1", "anime4k"] as SpatialImageMethod[]) {
    if (generation !== runGeneration) return;
    try {
      const elapsed = spatialUpscalers.get(method)!.run(
        method,
        sourceCanvas,
        sourceCanvas.width,
        sourceCanvas.height,
        outputWidth,
        outputHeight,
        applyRcas
      );
      setCardStatus(method, `${elapsed.toFixed(1)} ms`);
    } catch (error) {
      setCardStatus(method, error instanceof Error ? error.message : "Failed", true);
    }
  }
  void runNeural("websr", generation);
  void runNeural("fsrcnn", generation);
}

async function loadFile(file: File) {
  if (!file.type.startsWith("image/")) return;
  const bitmap = await createImageBitmap(file);
  sourceBitmap?.close();
  sourceBitmap = bitmap;
  sourceFileName = file.name;
  document.querySelector("#source-name")!.textContent = file.name;
  document.querySelector("#original-size")!.textContent = `${bitmap.width} × ${bitmap.height}`;
  sourcePanel.hidden = false;
  results.hidden = false;
  dropTarget.classList.add("has-image");
  dropTarget.querySelector("strong")!.textContent = "Choose another image";
  dropTarget.querySelector("small")!.textContent = "Drop a replacement here or click to browse";
  updateSourceCanvas();
  await runAll();
}

dropTarget.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) void loadFile(file);
  fileInput.value = "";
});
for (const eventName of ["dragenter", "dragover"] as const) {
  dropTarget.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropTarget.classList.add("drag-active");
  });
}
dropTarget.addEventListener("dragleave", () => dropTarget.classList.remove("drag-active"));
dropTarget.addEventListener("drop", (event) => {
  event.preventDefault();
  dropTarget.classList.remove("drag-active");
  const file = [...event.dataTransfer!.files].find((candidate) => candidate.type.startsWith("image/"));
  if (file) void loadFile(file);
});

document.querySelectorAll<HTMLButtonElement>("[data-scale]").forEach((button) => {
  button.addEventListener("click", () => {
    inputScale = Number(button.dataset.scale);
    document.querySelectorAll<HTMLButtonElement>("[data-scale]").forEach((candidate) => {
      candidate.classList.toggle("active", candidate === button);
    });
    void runAll();
  });
});
document.querySelector<HTMLButtonElement>("#run-all")!.addEventListener("click", () => void runAll());
rcasToggle.addEventListener("change", () => void runAll());
