import { FilesetResolver, ImageSegmenter, type ImageSegmenterResult } from "@mediapipe/tasks-vision";
import { RelightRenderer, type Look } from "./renderer";
import "./style.css";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm";

const state: Look = {
  autoAdjust: true,
  targetLuma: .58,
  manualEv: 0,
  backgroundEv: 0,
  warmth: 0,
  contrast: 1,
  isolate: true,
  labTransfer: true,
  referenceStrength: 0,
  referencePersonMean: [.5, .5, .5],
  referencePersonStd: [.22, .22, .22],
  referenceBackgroundMean: [.5, .5, .5],
  referenceBackgroundStd: [.22, .22, .22],
  referencePersonLabMean: [.5, 0, 0],
  referencePersonLabStd: [.22, .22, .22],
  referenceBackgroundLabMean: [.5, 0, 0],
  referenceBackgroundLabStd: [.22, .22, .22],
  referenceBackgroundColor: [.06, .06, .28],
  replaceBackground: true,
  rimIntensity: .4,
  audioReactive: false,
  audioLevel: 0,
  audioBeat: 0,
  rimColor: [.56, .86, 1]
};
let showMask = false;

type MusicTrack = {
  trackId: number;
  trackName: string;
  artistName: string;
  previewUrl?: string;
  artworkUrl100?: string;
  trackViewUrl?: string;
};

type StoredVideo = {
  id: string;
  name: string;
  type: string;
  blob: Blob;
  createdAt: number;
};

class VideoSessionStore {
  private readonly manifestKey = "relight-lab.video-ids";
  private readonly dbPromise: Promise<IDBDatabase>;

  constructor() {
    this.dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open("relight-lab-session", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("videos", { keyPath: "id" });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private ids() {
    try {
      const value = JSON.parse(sessionStorage.getItem(this.manifestKey) ?? "[]");
      return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [];
    } catch {
      return [];
    }
  }

  private setIds(ids: string[]) {
    sessionStorage.setItem(this.manifestKey, JSON.stringify(ids));
  }

  async save(file: File) {
    const id = crypto.randomUUID();
    const record: StoredVideo = { id, name: file.name, type: file.type, blob: file, createdAt: Date.now() };
    const db = await this.dbPromise;
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("videos", "readwrite");
      transaction.objectStore("videos").put(record);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    this.setIds([...this.ids(), id]);
    return id;
  }

  async load() {
    const ids = this.ids();
    const db = await this.dbPromise;
    const records: StoredVideo[] = [];
    for (const id of ids) {
      const record = await new Promise<StoredVideo | undefined>((resolve, reject) => {
        const request = db.transaction("videos", "readonly").objectStore("videos").get(id);
        request.onsuccess = () => resolve(request.result as StoredVideo | undefined);
        request.onerror = () => reject(request.error);
      });
      if (record) records.push(record);
    }
    if (records.length !== ids.length) this.setIds(records.map((record) => record.id));
    return records;
  }

  async remove(id: string) {
    this.setIds(this.ids().filter((storedId) => storedId !== id));
    const db = await this.dbPromise;
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction("videos", "readwrite");
      transaction.objectStore("videos").delete(id);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  }
}

const videoSessionStore = new VideoSessionStore();

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <main class="app-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">◒</span> Relight Lab</div>
      <div class="status"><span class="status-dot" id="model-dot"></span><span id="model-status">Model loads when isolation is enabled</span></div>
    </header>
    <section class="controls">
      <div class="actions">
        <button class="action" id="webcam">Use webcam</button>
        <button class="action drop-target" id="reference" title="Choose or drop a reference image">Ref image</button>
        <input id="file-input" type="file" accept="video/*" multiple hidden />
        <input id="reference-input" type="file" accept="image/*" hidden />
      </div>
    </section>
    <div class="toggle-row">
      <label class="toggle"><input id="auto-adjust" type="checkbox" checked><span class="switch"></span>Auto adjustments</label>
      <label class="toggle"><input id="isolate" type="checkbox" checked><span class="switch"></span>Isolate people</label>
      <label class="toggle"><input id="show-mask" type="checkbox"><span class="switch"></span>Show masks</label>
      <label class="toggle"><input id="reference-background" type="checkbox" checked><span class="switch"></span>Reference background</label>
      <label class="toggle" title="Reinhard mean and standard-deviation transfer in CIELAB"><input id="lab-transfer" type="checkbox" checked><span class="switch"></span>Reinhard LAB</label>
      <div class="shape-switch" role="group" aria-label="Video tile shape">
        <span>Tiles</span>
        <button type="button" data-shape="square" aria-pressed="false">Square</button>
        <button class="active" type="button" data-shape="portrait" aria-pressed="true">Portrait</button>
      </div>
      <div class="backlight-compact" title="Music drives this slider while playing; drag it for temporary manual control">
        <label for="rim-intensity">Light</label>
        <input id="rim-intensity" type="range" min="0" max="100" value="40" step="1">
        <input id="rim-color" type="color" value="#8edcff" aria-label="Backlight color">
        <output id="rim-color-output" for="rim-color" hidden>#8edcff</output>
      </div>
      <span class="hint">MediaPipe multiclass · 256 × 256 · videos survive reload</span>
    </div>
    <section class="music-panel">
      <div class="music-copy">
        <strong>Music preview</strong>
        <small>Search and stream 30-second previews</small>
      </div>
      <div class="music-controls">
        <form class="music-search" id="music-search-form">
          <input id="music-search-input" type="search" placeholder="Search Apple Music" aria-label="Search Apple Music" autocomplete="off">
          <button type="submit">Search</button>
        </form>
        <div class="music-results" id="music-results" hidden></div>
        <div class="now-playing" id="now-playing" hidden>
          <img id="music-artwork" alt="">
          <button class="preview-toggle" id="preview-toggle" type="button" aria-label="Pause preview">Ⅱ</button>
          <div class="track-copy"><strong id="track-name"></strong><span id="track-artist"></span></div>
          <a id="track-link" target="_blank" rel="noopener">Open in Apple Music ↗</a>
        </div>
        <div class="music-footnote"><span id="music-status">Enter a song or artist.</span><span>Preview provided courtesy of iTunes</span></div>
      </div>
    </section>
    <section class="reference-strip drop-target" id="reference-strip" title="Drop a reference image here" hidden>
      <img class="reference-thumb" id="reference-thumb" alt="Lighting reference">
      <div class="reference-copy"><span class="reference-label">Reference look</span><strong id="reference-name"></strong><span id="reference-stats">Analyzing reference…</span></div>
      <button class="remove" id="reset-reference" title="Restore default reference" aria-label="Restore default reference">✕</button>
    </section>
    <section class="grid" id="grid" data-tile-shape="portrait">
      <button class="add-video-tile drop-target" id="add-video-tile" type="button" aria-label="Add videos">
        <span class="add-video-icon">＋</span>
        <strong>Add videos</strong>
        <small>Drop files here or click to browse</small>
      </button>
    </section>
  </main>`;

class SegmenterPool {
  private model?: ImageSegmenter;
  private loading?: Promise<ImageSegmenter>;
  private busy = false;
  private lastRun = 0;
  private queue: VideoSource[] = [];
  private queued = new Set<VideoSource>();

  async load() {
    if (this.model) return this.model;
    if (this.loading) return this.loading;
    setModelStatus("Loading person model…");
    this.loading = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      this.model = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        outputCategoryMask: false,
        outputConfidenceMasks: true
      });
      setModelStatus("Person model ready", true);
      return this.model;
    })().catch((error) => {
      setModelStatus("Person model unavailable");
      this.loading = undefined;
      throw error;
    });
    return this.loading;
  }

  request(source: VideoSource) {
    if (!state.isolate || this.queued.has(source)) return;
    this.queue.push(source);
    this.queued.add(source);
    void this.drain();
  }

  remove(source: VideoSource) {
    this.queued.delete(source);
    this.queue = this.queue.filter((item) => item !== source);
  }

  private async drain() {
    if (this.busy || !state.isolate || !this.queue.length) return;
    // Cap the shared model near 30 fps. With several sources this budget is
    // distributed round-robin, while every rendered mask remains frame-locked.
    const wait = Math.max(0, 33 - (performance.now() - this.lastRun));
    if (wait > 0) {
      window.setTimeout(() => void this.drain(), wait);
      return;
    }
    const source = this.queue.shift()!;
    this.queued.delete(source);
    if (!source.active || source.video.readyState < 2) {
      void this.drain();
      return;
    }
    this.busy = true;
    try {
      const model = await this.load();
      const frame = source.captureSegmentationFrame();
      if (!frame) return;
      model.segmentForVideo(frame, performance.now(), (result) => source.acceptSegmentation(result));
    } catch (error) {
      console.error(error);
    } finally {
      this.lastRun = performance.now();
      this.busy = false;
      void this.drain();
    }
  }
}

type ColorStats = {
  mean: [number, number, number];
  std: [number, number, number];
  luma: number;
  count: number;
};

function extractPersonMatte(result: ImageSegmenterResult) {
  const background = result.confidenceMasks?.[0];
  if (background) {
    const confidence = background.getAsFloat32Array();
    const alpha = new Uint8Array(confidence.length);
    for (let i = 0; i < confidence.length; i++) alpha[i] = Math.round(Math.max(0, Math.min(1, 1 - confidence[i])) * 255);
    return { alpha, width: background.width, height: background.height };
  }
  // Retain compatibility with model variants that only provide categories.
  const category = result.categoryMask;
  if (!category) return undefined;
  const raw = category.getAsUint8Array();
  const alpha = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) alpha[i] = raw[i] > 0 ? 255 : 0;
  return { alpha, width: category.width, height: category.height };
}

function colorStats(pixels: Uint8ClampedArray, mask: Uint8Array, person: boolean): ColorStats {
  const sum = [0, 0, 0];
  const square = [0, 0, 0];
  let count = 0;
  const pixelCount = Math.min(mask.length, pixels.length / 4);
  for (let i = 0; i < pixelCount; i += 2) {
    if ((mask[i] >= 128) !== person) continue;
    const p = i * 4;
    for (let c = 0; c < 3; c++) {
      const value = pixels[p + c] / 255;
      sum[c] += value;
      square[c] += value * value;
    }
    count++;
  }
  if (!count) return { mean: [.5, .5, .5], std: [.22, .22, .22], luma: .5, count: 0 };
  const mean = sum.map((value) => value / count) as [number, number, number];
  const std = square.map((value, c) => Math.sqrt(Math.max(.0036, value / count - mean[c] * mean[c]))) as [number, number, number];
  return { mean, std, luma: mean[0] * .2126 + mean[1] * .7152 + mean[2] * .0722, count };
}

function rgbPixelToLab(red: number, green: number, blue: number): [number, number, number] {
  const toLinear = (value: number) => {
    const channel = value / 255;
    return channel <= .04045 ? channel / 12.92 : Math.pow((channel + .055) / 1.055, 2.4);
  };
  const r = toLinear(red);
  const g = toLinear(green);
  const b = toLinear(blue);
  const x = (r * .4124564 + g * .3575761 + b * .1804375) / .95047;
  const y = r * .2126729 + g * .7151522 + b * .072175;
  const z = (r * .0193339 + g * .119192 + b * .9503041) / 1.08883;
  const curve = (value: number) => value > .008856 ? Math.cbrt(value) : 7.787 * value + 16 / 116;
  const fx = curve(x);
  const fy = curve(y);
  const fz = curve(z);
  return [(116 * fy - 16) / 100, 500 * (fx - fy) / 128, 200 * (fy - fz) / 128];
}

function labColorStats(pixels: Uint8ClampedArray, mask: Uint8Array, person: boolean): ColorStats {
  const sum = [0, 0, 0];
  const square = [0, 0, 0];
  let count = 0;
  const pixelCount = Math.min(mask.length, pixels.length / 4);
  for (let i = 0; i < pixelCount; i += 2) {
    if ((mask[i] >= 128) !== person) continue;
    const p = i * 4;
    const lab = rgbPixelToLab(pixels[p], pixels[p + 1], pixels[p + 2]);
    for (let c = 0; c < 3; c++) {
      sum[c] += lab[c];
      square[c] += lab[c] * lab[c];
    }
    count++;
  }
  if (!count) return { mean: [.5, 0, 0], std: [.22, .22, .22], luma: .5, count: 0 };
  const mean = sum.map((value) => value / count) as [number, number, number];
  const std = square.map((value, c) => Math.sqrt(Math.max(.000625, value / count - mean[c] * mean[c]))) as [number, number, number];
  return { mean, std, luma: mean[0], count };
}

function mixStats(current: [number, number, number], next: [number, number, number], amount: number) {
  for (let i = 0; i < 3; i++) current[i] += (next[i] - current[i]) * amount;
}

function dominantBackgroundColor(pixels: Uint8ClampedArray, mask: Uint8Array): [number, number, number] {
  // A quantized mode is more representative of a backdrop than an RGB mean,
  // which can be skewed by floor lines, shadows, or small bright objects.
  const bins = new Uint32Array(16 * 16 * 16);
  const pixelCount = Math.min(mask.length, pixels.length / 4);
  for (let i = 0; i < pixelCount; i += 2) {
    if (mask[i] >= 128) continue;
    const p = i * 4;
    const index = ((pixels[p] >> 4) << 8) | ((pixels[p + 1] >> 4) << 4) | (pixels[p + 2] >> 4);
    bins[index]++;
  }
  let dominant = 0;
  for (let i = 1; i < bins.length; i++) if (bins[i] > bins[dominant]) dominant = i;
  const targetR = dominant >> 8;
  const targetG = (dominant >> 4) & 15;
  const targetB = dominant & 15;
  const sum = [0, 0, 0];
  let count = 0;
  for (let i = 0; i < pixelCount; i++) {
    if (mask[i] >= 128) continue;
    const p = i * 4;
    if ((pixels[p] >> 4) !== targetR || (pixels[p + 1] >> 4) !== targetG || (pixels[p + 2] >> 4) !== targetB) continue;
    sum[0] += pixels[p]; sum[1] += pixels[p + 1]; sum[2] += pixels[p + 2];
    count++;
  }
  return count ? [sum[0] / count / 255, sum[1] / count / 255, sum[2] / count / 255] : [.06, .06, .28];
}

function colorToHex(color: [number, number, number]) {
  return `#${color.map((channel) => Math.round(channel * 255).toString(16).padStart(2, "0")).join("")}`;
}

function hexToColor(value: string): [number, number, number] {
  return [
    Number.parseInt(value.slice(1, 3), 16) / 255,
    Number.parseInt(value.slice(3, 5), 16) / 255,
    Number.parseInt(value.slice(5, 7), 16) / 255
  ];
}

class ReferenceAnalyzer {
  private model?: ImageSegmenter;
  private loading?: Promise<ImageSegmenter>;
  private objectUrl?: string;

  private async load() {
    if (this.model) return this.model;
    if (!this.loading) {
      this.loading = (async () => {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        this.model = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "IMAGE",
          outputCategoryMask: false,
          outputConfidenceMasks: true
        });
        return this.model;
      })();
    }
    return this.loading;
  }

  async analyze(file: File) {
    const strip = document.querySelector<HTMLElement>("#reference-strip")!;
    const thumb = document.querySelector<HTMLImageElement>("#reference-thumb")!;
    const statsLabel = document.querySelector<HTMLElement>("#reference-stats")!;
    document.querySelector("#reference-name")!.textContent = file.name;
    statsLabel.textContent = "Analyzing person and environment…";
    strip.hidden = false;
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = URL.createObjectURL(file);
    thumb.src = this.objectUrl;

    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const model = await this.load();
    model.segment(canvas, (result) => {
      const matte = extractPersonMatte(result);
      if (!matte) throw new Error("The reference person could not be segmented");

      const sample = document.createElement("canvas");
      sample.width = matte.width;
      sample.height = matte.height;
      const sampleCtx = sample.getContext("2d", { willReadFrequently: true })!;
      sampleCtx.drawImage(canvas, 0, 0, sample.width, sample.height);
      const pixels = sampleCtx.getImageData(0, 0, sample.width, sample.height).data;
      const person = colorStats(pixels, matte.alpha, true);
      const background = colorStats(pixels, matte.alpha, false);
      const personLab = labColorStats(pixels, matte.alpha, true);
      const backgroundLab = labColorStats(pixels, matte.alpha, false);
      const backgroundColor = dominantBackgroundColor(pixels, matte.alpha);
      if (person.count < 100) throw new Error("No clear person was found in the reference");

      state.referencePersonMean = person.mean;
      state.referencePersonStd = person.std;
      state.referenceBackgroundMean = background.mean;
      state.referenceBackgroundStd = background.std;
      state.referencePersonLabMean = personLab.mean;
      state.referencePersonLabStd = personLab.std;
      state.referenceBackgroundLabMean = backgroundLab.mean;
      state.referenceBackgroundLabStd = backgroundLab.std;
      state.referenceBackgroundColor = backgroundColor;
      state.targetLuma = Math.max(.3, Math.min(.75, person.luma));
      state.referenceStrength = .45;
      const colorHex = colorToHex(backgroundColor);
      statsLabel.textContent = `person ${Math.round(person.luma * 100)}% · background ${colorHex}`;
    });
  }

}

const segmenter = new SegmenterPool();
const referenceAnalyzer = new ReferenceAnalyzer();
const sources: VideoSource[] = [];

class VideoSource {
  active = true;
  readonly video = document.createElement("video");
  private canvas = document.createElement("canvas");
  private maskCanvas = document.createElement("canvas");
  private renderer: RelightRenderer;
  private card = document.createElement("article");
  private lastAnalysis = 0;
  private objectUrl?: string;
  private stream?: MediaStream;
  private analysisCanvas = document.createElement("canvas");
  private analysisCtx = this.analysisCanvas.getContext("2d", { willReadFrequently: true })!;
  private segmentationFrame = document.createElement("canvas");
  private segmentationCtx = this.segmentationFrame.getContext("2d", { alpha: false })!;
  private pendingFrame = document.createElement("canvas");
  private pendingCtx = this.pendingFrame.getContext("2d", { alpha: false })!;
  private pairedFrame = document.createElement("canvas");
  private pairedCtx = this.pairedFrame.getContext("2d", { alpha: false })!;
  private hasPairedFrame = false;
  private sourceTextureDirty = true;

  constructor(file: File | MediaStream, name: string, private storedId?: string) {
    this.video.className = "source-video";
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.loop = true;
    this.video.autoplay = true;
    if (file instanceof MediaStream) {
      this.stream = file;
      this.video.srcObject = file;
    } else {
      this.objectUrl = URL.createObjectURL(file);
      this.video.src = this.objectUrl;
    }
    this.maskCanvas.className = "mask-preview";
    this.maskCanvas.hidden = !showMask;
    this.card.className = "video-card";
    this.card.innerHTML = `<div class="stage"><button class="remove" title="Remove">✕</button></div>`;
    this.card.querySelector(".stage")!.append(this.video, this.canvas, this.maskCanvas);
    const removeButton = this.card.querySelector<HTMLButtonElement>(".remove")!;
    removeButton.setAttribute("aria-label", `Remove ${name}`);
    removeButton.addEventListener("click", () => this.remove());
    const grid = document.querySelector("#grid")!;
    grid.insertBefore(this.card, document.querySelector("#add-video-tile"));
    this.renderer = new RelightRenderer(this.canvas);
    this.video.addEventListener("loadeddata", () => this.video.play().catch(() => {}));
    this.tick();
  }

  private tick = () => {
    if (!this.active) return;
    if (state.isolate) {
      if (!this.hasPairedFrame) this.captureSegmentationFrame();
      const displayFrame = this.hasPairedFrame ? this.pairedFrame : this.pendingFrame;
      this.renderer.render(displayFrame, state, this.sourceTextureDirty);
      this.sourceTextureDirty = false;
    } else {
      this.captureSegmentationFrame();
      this.renderer.render(this.pendingFrame, state, true);
    }
    if (performance.now() - this.lastAnalysis > 120) {
      this.lastAnalysis = performance.now();
      if (state.isolate) segmenter.request(this);
      else this.updateGlobalExposure();
    }
    requestAnimationFrame(this.tick);
  };

  captureSegmentationFrame() {
    if (this.video.readyState < 2 || !this.video.videoWidth) return undefined;
    const renderSize = Math.min(1280, Math.max(512, Math.max(this.video.videoWidth, this.video.videoHeight)));
    if (this.pendingFrame.width !== renderSize || this.pendingFrame.height !== renderSize) {
      this.pendingFrame.width = renderSize;
      this.pendingFrame.height = renderSize;
      this.pendingCtx.imageSmoothingEnabled = true;
      this.pendingCtx.imageSmoothingQuality = "high";
    }
    if (this.segmentationFrame.width !== 256 || this.segmentationFrame.height !== 256) {
      this.segmentationFrame.width = 256;
      this.segmentationFrame.height = 256;
      this.segmentationCtx.imageSmoothingEnabled = true;
      this.segmentationCtx.imageSmoothingQuality = "high";
    }
    const [red, green, blue] = state.referenceBackgroundColor.map((channel) => Math.round(channel * 255));
    this.pendingCtx.fillStyle = `rgb(${red} ${green} ${blue})`;
    this.pendingCtx.fillRect(0, 0, renderSize, renderSize);
    const scale = Math.min(renderSize / this.video.videoWidth, renderSize / this.video.videoHeight);
    const width = Math.round(this.video.videoWidth * scale);
    const height = Math.round(this.video.videoHeight * scale);
    const x = Math.floor((renderSize - width) / 2);
    const y = Math.floor((renderSize - height) / 2);
    // Draw the decoded video once, then derive the model input from this exact
    // high-resolution snapshot so the display frame and matte stay paired.
    this.pendingCtx.drawImage(this.video, x, y, width, height);
    this.segmentationCtx.drawImage(this.pendingFrame, 0, 0, 256, 256);
    if (!this.hasPairedFrame) this.sourceTextureDirty = true;
    return this.segmentationFrame;
  }

  acceptSegmentation(result: ImageSegmenterResult) {
    const matte = extractPersonMatte(result);
    if (!matte) return;
    if (this.pairedFrame.width !== this.pendingFrame.width || this.pairedFrame.height !== this.pendingFrame.height) {
      this.pairedFrame.width = this.pendingFrame.width;
      this.pairedFrame.height = this.pendingFrame.height;
      this.pairedCtx.imageSmoothingEnabled = true;
      this.pairedCtx.imageSmoothingQuality = "high";
    }
    this.pairedCtx.drawImage(this.pendingFrame, 0, 0);
    const alpha = this.renderer.updateMask(matte.alpha, matte.width, matte.height);
    this.renderer.updateFraming(alpha, matte.width, matte.height);
    this.hasPairedFrame = true;
    this.sourceTextureDirty = true;
    this.updateMaskPreview(alpha, matte.width, matte.height);
    this.updateExposure(alpha, matte.width, matte.height);
    this.renderer.render(this.pairedFrame, state, true);
    this.sourceTextureDirty = false;
  }

  private updateExposure(mask: Uint8Array, width: number, height: number) {
    this.analysisCanvas.width = width;
    this.analysisCanvas.height = height;
    this.analysisCtx.drawImage(this.pairedFrame, 0, 0, width, height);
    const pixels = this.analysisCtx.getImageData(0, 0, width, height).data;
    const personColor = colorStats(pixels, mask, true);
    const backgroundColor = colorStats(pixels, mask, false);
    const personLab = labColorStats(pixels, mask, true);
    const backgroundLab = labColorStats(pixels, mask, false);
    if (personColor.count) {
      mixStats(this.renderer.sourcePersonMean, personColor.mean, .12);
      mixStats(this.renderer.sourcePersonStd, personColor.std, .12);
      mixStats(this.renderer.sourcePersonLabMean, personLab.mean, .12);
      mixStats(this.renderer.sourcePersonLabStd, personLab.std, .12);
    }
    if (backgroundColor.count) {
      mixStats(this.renderer.sourceBackgroundMean, backgroundColor.mean, .12);
      mixStats(this.renderer.sourceBackgroundStd, backgroundColor.std, .12);
      mixStats(this.renderer.sourceBackgroundLabMean, backgroundLab.mean, .12);
      mixStats(this.renderer.sourceBackgroundLabStd, backgroundLab.std, .12);
    }
    let sum = 0;
    let count = 0;
    for (let i = 0; i < mask.length; i += 2) {
      if (mask[i] < 128) continue;
      const p = i * 4;
      const luma = (pixels[p] * .2126 + pixels[p + 1] * .7152 + pixels[p + 2] * .0722) / 255;
      if (luma > .025 && luma < .98) { sum += luma; count++; }
    }
    if (!count) return;
    const measured = sum / count;
    const desiredEv = Math.max(-1.5, Math.min(1.5, Math.log2(state.targetLuma / Math.max(.02, measured))));
    this.renderer.autoEv += (desiredEv - this.renderer.autoEv) * .07;
  }

  private updateGlobalExposure() {
    if (this.video.readyState < 2) return;
    const width = 256;
    const height = Math.max(1, Math.round(width * this.video.videoHeight / this.video.videoWidth));
    this.analysisCanvas.width = width;
    this.analysisCanvas.height = height;
    this.analysisCtx.drawImage(this.video, 0, 0, width, height);
    const pixels = this.analysisCtx.getImageData(0, 0, width, height).data;
    const wholeFrameMask = new Uint8Array(width * height).fill(255);
    const frameColor = colorStats(pixels, wholeFrameMask, true);
    const frameLab = labColorStats(pixels, wholeFrameMask, true);
    mixStats(this.renderer.sourcePersonMean, frameColor.mean, .12);
    mixStats(this.renderer.sourcePersonStd, frameColor.std, .12);
    mixStats(this.renderer.sourcePersonLabMean, frameLab.mean, .12);
    mixStats(this.renderer.sourcePersonLabStd, frameLab.std, .12);
    let sum = 0;
    let count = 0;
    for (let p = 0; p < pixels.length; p += 8) {
      const luma = (pixels[p] * .2126 + pixels[p + 1] * .7152 + pixels[p + 2] * .0722) / 255;
      if (luma > .025 && luma < .98) { sum += luma; count++; }
    }
    if (!count) return;
    const measured = sum / count;
    const desiredEv = Math.max(-1.5, Math.min(1.5, Math.log2(state.targetLuma / Math.max(.02, measured))));
    this.renderer.autoEv += (desiredEv - this.renderer.autoEv) * .07;
  }

  private updateMaskPreview(mask: Uint8Array, width: number, height: number) {
    this.maskCanvas.width = width;
    this.maskCanvas.height = height;
    const ctx = this.maskCanvas.getContext("2d")!;
    const image = ctx.createImageData(width, height);
    for (let i = 0; i < mask.length; i++) {
      const p = i * 4;
      image.data[p] = mask[i] ? 214 : 20;
      image.data[p + 1] = mask[i] ? 255 : 21;
      image.data[p + 2] = mask[i] ? 87 : 24;
      image.data[p + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  setMaskVisible(visible: boolean) { this.maskCanvas.hidden = !visible; }

  remove() {
    this.active = false;
    segmenter.remove(this);
    this.card.remove();
    this.video.pause();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    if (this.storedId) void videoSessionStore.remove(this.storedId).catch((error) => console.error("Could not remove stored video", error));
    const index = sources.indexOf(this);
    if (index >= 0) sources.splice(index, 1);
  }
}

function setModelStatus(message: string, ready = false) {
  document.querySelector("#model-status")!.textContent = message;
  document.querySelector("#model-dot")!.classList.toggle("ready", ready);
}

function addSource(source: File | MediaStream, name: string, storedId?: string) {
  sources.push(new VideoSource(source, name, storedId));
}

async function addVideoFile(file: File) {
  try {
    const storedId = await videoSessionStore.save(file);
    addSource(file, file.name, storedId);
  } catch (error) {
    console.error("Could not persist video for this session", error);
    setModelStatus("Video added, but browser storage was unavailable");
    addSource(file, file.name);
  }
}

async function restoreSessionVideos() {
  try {
    for (const record of await videoSessionStore.load()) {
      const file = new File([record.blob], record.name, { type: record.type, lastModified: record.createdAt });
      addSource(file, record.name, record.id);
    }
  } catch (error) {
    console.error("Could not restore session videos", error);
  }
}

const previewAudio = new Audio();
previewAudio.crossOrigin = "anonymous";
previewAudio.preload = "none";
previewAudio.loop = true;
let currentTrack: MusicTrack | undefined;
let audioContext: AudioContext | undefined;
let audioAnalyser: AnalyserNode | undefined;
let audioSamples: Uint8Array<ArrayBuffer> | undefined;
let frequencySamples: Uint8Array<ArrayBuffer> | undefined;
let previousSpectrum: Float32Array<ArrayBuffer> | undefined;
const onsetHistory: number[] = [];
let lastBeatAt = -Infinity;
let meterAnimation = 0;
let rimSliderHeld = false;

function searchMusic(term: string) {
  return new Promise<MusicTrack[]>((resolve, reject) => {
    const callbackName = `__relightMusic${Date.now()}${Math.floor(Math.random() * 1000)}`;
    const script = document.createElement("script");
    const globals = window as unknown as Record<string, unknown>;
    const localeRegion = navigator.language.match(/-([A-Za-z]{2})$/)?.[1]?.toUpperCase() ?? "US";
    let settled = false;
    const cleanup = () => {
      delete globals[callbackName];
      script.remove();
    };
    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Music search timed out"));
    }, 10000);
    globals[callbackName] = (payload: { results?: MusicTrack[] }) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      cleanup();
      resolve((payload.results ?? []).filter((track) => Boolean(track.previewUrl)));
    };
    script.onerror = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      cleanup();
      reject(new Error("Music search could not be reached"));
    };
    const params = new URLSearchParams({
      term,
      media: "music",
      entity: "song",
      limit: "8",
      country: localeRegion,
      callback: callbackName
    });
    script.src = `https://itunes.apple.com/search?${params}`;
    document.head.append(script);
  });
}

function artworkUrl(track: MusicTrack) {
  return (track.artworkUrl100 ?? "").replace("100x100bb", "160x160bb").replace(/^http:/, "https:");
}

function renderMusicResults(tracks: MusicTrack[]) {
  const results = document.querySelector<HTMLElement>("#music-results")!;
  results.replaceChildren();
  if (!tracks.length) {
    const empty = document.createElement("div");
    empty.className = "music-result";
    empty.textContent = "No playable previews found.";
    results.append(empty);
  }
  for (const track of tracks) {
    const button = document.createElement("button");
    button.className = "music-result";
    button.type = "button";
    const art = document.createElement("img");
    art.src = artworkUrl(track);
    art.alt = "";
    const copy = document.createElement("span");
    copy.className = "track-copy";
    const title = document.createElement("strong");
    title.textContent = track.trackName;
    const artist = document.createElement("span");
    artist.textContent = track.artistName;
    copy.append(title, artist);
    const play = document.createElement("span");
    play.textContent = "▶";
    button.append(art, copy, play);
    button.addEventListener("click", () => void playTrack(track));
    results.append(button);
  }
  results.hidden = false;
}

function ensureAudioGraph() {
  if (audioAnalyser) return;
  audioContext = new AudioContext();
  const source = audioContext.createMediaElementSource(previewAudio);
  audioAnalyser = audioContext.createAnalyser();
  audioAnalyser.fftSize = 2048;
  audioAnalyser.smoothingTimeConstant = .45;
  audioSamples = new Uint8Array(audioAnalyser.fftSize);
  frequencySamples = new Uint8Array(audioAnalyser.frequencyBinCount);
  previousSpectrum = new Float32Array(audioAnalyser.frequencyBinCount);
  source.connect(audioAnalyser);
  audioAnalyser.connect(audioContext.destination);
}

function detectBeat(now: number) {
  if (!audioAnalyser || !audioContext || !frequencySamples || !previousSpectrum) return false;
  audioAnalyser.getByteFrequencyData(frequencySamples);
  const binHz = audioContext.sampleRate / audioAnalyser.fftSize;
  const bassStart = Math.max(1, Math.floor(45 / binHz));
  const bassEnd = Math.min(frequencySamples.length - 1, Math.ceil(200 / binHz));
  // Keep the onset detector in the rhythmic low/mid band. Extending through
  // the whole spectrum makes hi-hats look like beats and produces flicker.
  const fluxEnd = Math.min(frequencySamples.length - 1, Math.ceil(1200 / binHz));
  let bass = 0;
  let bassBins = 0;
  let flux = 0;
  let fluxWeight = 0;
  for (let bin = 1; bin <= fluxEnd; bin++) {
    const value = frequencySamples[bin] / 255;
    if (bin >= bassStart && bin <= bassEnd) {
      bass += value;
      bassBins++;
    }
    const increase = Math.max(0, value - previousSpectrum[bin]);
    const weight = bin <= bassEnd ? 1.65 : 1;
    flux += increase * weight;
    fluxWeight += weight;
    previousSpectrum[bin] = value;
  }
  bass /= Math.max(1, bassBins);
  const onset = flux / Math.max(1, fluxWeight) * (.6 + bass * 1.4);
  let mean = 0;
  for (const value of onsetHistory) mean += value;
  mean /= Math.max(1, onsetHistory.length);
  let variance = 0;
  for (const value of onsetHistory) variance += (value - mean) ** 2;
  const deviation = Math.sqrt(variance / Math.max(1, onsetHistory.length));
  const threshold = mean + Math.max(.0018, deviation * 1.65);
  const beat = onsetHistory.length >= 18 && onset > Math.max(.005, threshold) && bass > .14 && now - lastBeatAt > 220;
  onsetHistory.push(onset);
  if (onsetHistory.length > 72) onsetHistory.shift();
  if (beat) lastBeatAt = now;
  return beat;
}

function updateMeter() {
  let target = 0;
  if (!previewAudio.paused && audioAnalyser && audioSamples) {
    audioAnalyser.getByteTimeDomainData(audioSamples);
    let sum = 0;
    for (const sample of audioSamples) {
      const centered = (sample - 128) / 128;
      sum += centered * centered;
    }
    const rms = Math.sqrt(sum / audioSamples.length);
    target = Math.max(0, Math.min(1, (rms - .022) * 2.8));
    if (detectBeat(performance.now())) {
      state.audioBeat = 1;
    }
  }
  const speed = target > state.audioLevel ? .34 : .09;
  state.audioLevel += (target - state.audioLevel) * speed;
  state.audioBeat *= .82;
  if (state.audioLevel < .002) state.audioLevel = 0;
  if (state.audioBeat < .002) state.audioBeat = 0;
  if (!rimSliderHeld) {
    const gain = state.audioReactive ? .6 + state.audioLevel * .75 + state.audioBeat * .85 : 1;
    const displayedIntensity = Math.min(1, state.rimIntensity * gain);
    const slider = document.querySelector<HTMLInputElement>("#rim-intensity")!;
    slider.value = String(Math.round(displayedIntensity * 100));
    slider.setAttribute("aria-valuetext", state.audioReactive ? "Music-driven light intensity" : "Light intensity");
  }
  if (!previewAudio.paused || state.audioLevel > 0 || state.audioBeat > 0) meterAnimation = requestAnimationFrame(updateMeter);
  else meterAnimation = 0;
}

function startMeter() {
  if (!meterAnimation) meterAnimation = requestAnimationFrame(updateMeter);
}

async function playTrack(track: MusicTrack) {
  if (!track.previewUrl) return;
  const status = document.querySelector<HTMLElement>("#music-status")!;
  try {
    ensureAudioGraph();
    await audioContext!.resume();
    if (currentTrack?.trackId !== track.trackId) {
      currentTrack = track;
      onsetHistory.length = 0;
      previousSpectrum?.fill(0);
      lastBeatAt = -Infinity;
      previewAudio.src = track.previewUrl.replace(/^http:/, "https:");
      document.querySelector<HTMLImageElement>("#music-artwork")!.src = artworkUrl(track);
      document.querySelector<HTMLElement>("#track-name")!.textContent = track.trackName;
      document.querySelector<HTMLElement>("#track-artist")!.textContent = track.artistName;
      const link = document.querySelector<HTMLAnchorElement>("#track-link")!;
      link.href = track.trackViewUrl ?? "https://music.apple.com/";
    }
    await previewAudio.play();
    state.audioReactive = !rimSliderHeld;
    document.querySelector<HTMLElement>("#now-playing")!.hidden = false;
    document.querySelector<HTMLElement>("#music-results")!.hidden = true;
    document.querySelector<HTMLButtonElement>("#preview-toggle")!.textContent = "Ⅱ";
    document.querySelector<HTMLButtonElement>("#preview-toggle")!.setAttribute("aria-label", "Pause preview");
    status.textContent = "Playing preview";
    startMeter();
  } catch (error) {
    state.audioReactive = false;
    console.error(error);
    status.textContent = "This preview could not be played.";
  }
}

document.querySelector<HTMLFormElement>("#music-search-form")!.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector<HTMLInputElement>("#music-search-input")!;
  const term = input.value.trim();
  if (!term) return;
  const status = document.querySelector<HTMLElement>("#music-status")!;
  status.textContent = "Searching…";
  document.querySelector<HTMLElement>("#music-results")!.hidden = true;
  try {
    const tracks = await searchMusic(term);
    renderMusicResults(tracks);
    status.textContent = tracks.length ? `${tracks.length} previews found` : "No playable previews found.";
  } catch (error) {
    console.error(error);
    status.textContent = "Search is temporarily unavailable.";
  }
});

document.querySelector<HTMLButtonElement>("#preview-toggle")!.addEventListener("click", async () => {
  const button = document.querySelector<HTMLButtonElement>("#preview-toggle")!;
  if (previewAudio.paused) {
    await audioContext?.resume();
    await previewAudio.play();
    state.audioReactive = !rimSliderHeld;
    button.textContent = "Ⅱ";
    button.setAttribute("aria-label", "Pause preview");
    document.querySelector<HTMLElement>("#music-status")!.textContent = "Playing preview";
    startMeter();
  } else {
    previewAudio.pause();
    state.audioReactive = false;
    button.textContent = "▶";
    button.setAttribute("aria-label", "Play preview");
    document.querySelector<HTMLElement>("#music-status")!.textContent = "Preview paused";
    startMeter();
  }
});

previewAudio.addEventListener("ended", () => {
  state.audioReactive = false;
  const button = document.querySelector<HTMLButtonElement>("#preview-toggle")!;
  button.textContent = "▶";
  button.setAttribute("aria-label", "Play preview");
  document.querySelector<HTMLElement>("#music-status")!.textContent = "Preview ended";
  startMeter();
});

document.querySelector<HTMLInputElement>("#auto-adjust")!.addEventListener("change", (event) => {
  state.autoAdjust = (event.target as HTMLInputElement).checked;
});
document.querySelector<HTMLInputElement>("#isolate")!.addEventListener("change", (event) => {
  state.isolate = (event.target as HTMLInputElement).checked;
  if (state.isolate) void segmenter.load();
});
document.querySelector<HTMLInputElement>("#show-mask")!.addEventListener("change", (event) => {
  showMask = (event.target as HTMLInputElement).checked;
  sources.forEach((source) => source.setMaskVisible(showMask));
});
document.querySelector<HTMLInputElement>("#reference-background")!.addEventListener("change", (event) => {
  state.replaceBackground = (event.target as HTMLInputElement).checked;
});
document.querySelector<HTMLInputElement>("#lab-transfer")!.addEventListener("change", (event) => {
  state.labTransfer = (event.target as HTMLInputElement).checked;
});
document.querySelectorAll<HTMLButtonElement>("[data-shape]").forEach((button) => {
  button.addEventListener("click", () => {
    const shape = button.dataset.shape === "portrait" ? "portrait" : "square";
    document.querySelector<HTMLElement>("#grid")!.dataset.tileShape = shape;
    document.querySelectorAll<HTMLButtonElement>("[data-shape]").forEach((item) => {
      const active = item.dataset.shape === shape;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
  });
});
const rimSlider = document.querySelector<HTMLInputElement>("#rim-intensity")!;
const beginManualRimControl = () => {
  rimSliderHeld = true;
  state.audioReactive = false;
};
const releaseManualRimControl = () => {
  if (!rimSliderHeld) return;
  rimSliderHeld = false;
  state.audioReactive = !previewAudio.paused;
  startMeter();
};
rimSlider.addEventListener("pointerdown", beginManualRimControl);
rimSlider.addEventListener("keydown", (event) => {
  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) beginManualRimControl();
});
rimSlider.addEventListener("input", (event) => {
  beginManualRimControl();
  state.rimIntensity = Number((event.target as HTMLInputElement).value) / 100;
});
rimSlider.addEventListener("keyup", releaseManualRimControl);
rimSlider.addEventListener("blur", releaseManualRimControl);
window.addEventListener("pointerup", releaseManualRimControl);
window.addEventListener("pointercancel", releaseManualRimControl);
document.querySelector<HTMLInputElement>("#rim-color")!.addEventListener("input", (event) => {
  const value = (event.target as HTMLInputElement).value;
  state.rimColor = hexToColor(value);
  document.querySelector<HTMLOutputElement>("#rim-color-output")!.value = value;
});
document.querySelector("#add-video-tile")!.addEventListener("click", () => document.querySelector<HTMLInputElement>("#file-input")!.click());
document.querySelector("#reference")!.addEventListener("click", () => document.querySelector<HTMLInputElement>("#reference-input")!.click());
async function setReferenceFile(file: File) {
  try {
    await referenceAnalyzer.analyze(file);
  } catch (error) {
    console.error(error);
    document.querySelector("#reference-stats")!.textContent = "Could not analyze this reference";
  }
}

document.querySelector<HTMLInputElement>("#reference-input")!.addEventListener("change", async (event) => {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) await setReferenceFile(file);
  input.value = "";
});
document.querySelector("#reset-reference")!.addEventListener("click", () => void loadDefaultReference());
document.querySelector<HTMLInputElement>("#file-input")!.addEventListener("change", async (event) => {
  const input = event.target as HTMLInputElement;
  await Promise.all(Array.from(input.files ?? []).map(addVideoFile));
  input.value = "";
});

function bindFileDrop(
  element: HTMLElement,
  accepts: (file: File) => boolean,
  receive: (files: File[]) => void | Promise<void>
) {
  let dragDepth = 0;
  element.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth++;
    element.classList.add("drag-active");
  });
  element.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  element.addEventListener("dragleave", () => {
    dragDepth--;
    if (dragDepth <= 0) {
      dragDepth = 0;
      element.classList.remove("drag-active");
    }
  });
  element.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    element.classList.remove("drag-active");
    const files = Array.from(event.dataTransfer?.files ?? []).filter(accepts);
    if (files.length) void receive(files);
  });
}

bindFileDrop(
  document.querySelector<HTMLElement>("#reference")!,
  (file) => file.type.startsWith("image/"),
  async ([file]) => { if (file) await setReferenceFile(file); }
);
bindFileDrop(
  document.querySelector<HTMLElement>("#reference-strip")!,
  (file) => file.type.startsWith("image/"),
  async ([file]) => { if (file) await setReferenceFile(file); }
);
bindFileDrop(
  document.querySelector<HTMLElement>("#add-video-tile")!,
  (file) => file.type.startsWith("video/"),
  async (files) => { await Promise.all(files.map(addVideoFile)); }
);
document.querySelector("#webcam")!.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    addSource(stream, "Live camera");
  } catch (error) {
    console.error(error);
    setModelStatus("Camera permission was not granted");
  }
});

async function loadDefaultReference() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}default-reference.png`);
    if (!response.ok) throw new Error(`Default reference returned ${response.status}`);
    const blob = await response.blob();
    await referenceAnalyzer.analyze(new File([blob], "retro-comig.png", { type: blob.type || "image/png" }));
  } catch (error) {
    console.error("Could not load the default reference", error);
  }
}

void segmenter.load();
void loadDefaultReference();
void restoreSessionVideos();
