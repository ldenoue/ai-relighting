# Relight Lab

A browser-only experiment that makes uploaded videos and webcam feeds feel as though they share one lighting environment. Frames stay on the device: decoding, person segmentation, compositing, color matching, and rendering all happen in the browser.

**Live demo:** [ldenoue.github.io/ai-relighting](https://ldenoue.github.io/ai-relighting/)

**Upscaling lab:** [ldenoue.github.io/ai-relighting/upscaling.html](https://ldenoue.github.io/ai-relighting/upscaling.html)

## What it does

- Lets you compare MediaPipe Selfie Multiclass Segmentation with three recurrent RVM MobileNetV3 runtimes: ONNX/WASM, TensorFlow.js/WebGPU, and LiteRT/WebGPU.
- Derives the model input from that high-resolution snapshot and pairs each result back to the same frame, avoiding both mask lag and zoomed 256-pixel video.
- Letterboxes sources before processing and automatically frames each person to roughly 80% of the tile height.
- Transfers person and background color statistics from a reference image.
- Applies Reinhard color transfer in CIELAB using the reference image's per-channel target mean and standard deviation.
- Replaces source backgrounds with the dominant background color inferred from that reference.
- Builds a layered backlight from a radial source, background bloom, confidence-aware rim lighting, inward spill, and a subtle floor pool.
- Searches Apple's public catalog for streamed 30-second previews and uses a Web Audio VU envelope to modulate the backlight intensity.
- Uses MediaPipe's continuous confidence as optical alpha, so solid pixels occlude the backlight while uncertain hair and edge pixels transmit more of it.
- Stores dropped video blobs in IndexedDB for the current browser session.
- Supports square and portrait presentation without changing the processing pipeline.
- Produces an intrinsic 640 × 640 frame with six live upscaling choices: Bilinear, Lanczos2, FSR 1.0, Anime4K Original, WebSR CNN-S, and FSRCNN ×2.

## How the pipeline works

1. Each local video or webcam is decoded once into a high-resolution square snapshot; a 256 × 256 copy is derived from it for inference.
2. The selected matte model produces a soft person alpha mask paired with the original high-resolution snapshot. MediaPipe uses inverted background confidence; every RVM backend carries four recurrent tensors per video for temporally stable hair and edge detail.
3. A GPU filter refines the matte, and a smoothed view transform normalizes subject framing.
4. Foreground luminance is measured and temporally stabilized to avoid exposure pumping.
5. A separate reference-image pass measures foreground/background RGB distributions and finds the dominant backdrop color.
6. WebGL 2 applies Reinhard CIELAB transfer, followed by background replacement, alpha compositing, and backlight layers into a shared 320 × 320 working frame.
7. Every method reconstructs the working frame at 640 × 640. Square and portrait controls remain CSS-only, so they do not change the processed pixels used in comparisons.

All six choices intentionally share one 320 × 320 input and produce a 640 × 640 result. **FSR 1.0** is the default and uses EASU's edge-directed 12-tap reconstruction; **Bilinear** is the browser baseline; **Lanczos2** uses a 16-tap kernel; **Anime4K Original** uses lightweight Sobel-guided line refinement; **WebSR CNN-S** runs the small Anime4K real-life CNN with WebGPU. **FSRCNN ×2** runs a luminance model through ONNX Runtime WebGPU and keeps chroma from a conventional resize. The independent **RCAS** toggle optionally applies the same final sharpening pass after any selected method.

The neural options load their runtimes and weights only when selected and display the latest completed result while the next frame is inferred. They require WebGPU; unsupported browsers keep showing the bilinear fallback. The first FSRCNN use downloads the roughly 24 MB ONNX Runtime WASM support module in addition to the 37 KB model. Implementations, model provenance, and licenses are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The **Matte** picker defaults to **RVM · LiteRT 256** when WebGPU is available. It uses the same 256 × 256 model-input resolution as MediaPipe, ONNX/WASM, and TensorFlow.js, making it the fairest runtime comparison. The TensorFlow.js option uses RVM's official INT8 browser graph, while both LiteRT options use FP16-weight conversions; every RVM backend keeps four recurrent tensors per video for temporally stable edges. **RVM · LiteRT 512×288** is a separate, larger fixed graph; the app letterboxes its square frame into the model's centered 288 × 288 region and crops the resulting alpha back to a square without stretching the subject. After either LiteRT model loads, its picker label reports **full GPU** or **mixed** from LiteRT's acceleration check so benchmark results do not hide CPU fallback. On a controlled 60 FPS static clip, the full application measured about 56 FPS with LiteRT 256 and 51 FPS with LiteRT 512×288 on the development machine; results depend on the browser, GPU, display cadence, and the selected relighting/upscaling path.

The LiteRT profiles share LiteRT's `GPUDevice` with a TypeGPU compute pipeline. Each video keeps a persistent float RGB `GPUBuffer`; the synchronized high-resolution snapshot is resized, letterboxed when necessary, and normalized directly into that buffer before it is wrapped as the LiteRT input tensor. This removes the model-input `getImageData()` call, JavaScript RGBA-to-RGB conversion, and CPU-backed tensor allocation. The alpha output still returns to the CPU because automatic framing, exposure/color analysis, and the current WebGL compositor consume a `Uint8Array` matte. If shared-device preprocessing cannot initialize, LiteRT automatically retains the former CPU conversion path.

**RVM · WASM** remains the compatibility path, using the official 14 MB FP32 ONNX model with ONNX Runtime SIMD WASM. The app falls back to WASM automatically if either WebGPU runtime cannot initialize. Direct ONNX Runtime WebGPU was tested but produced corrupt sparse alpha values from this opset-12 recurrent graph, so it is not exposed as a usable backend. MediaPipe remains available and continues to analyze static reference images because RVM's temporal memory provides no advantage for a single image. **Matte cutoff** remaps confidence through a soft 35-point transition—lower values recover conservative RVM pixels without creating a binary edge. The independent **Auto frame** toggle can disable the mask-derived 80%-height centering and immediately restore an identity view. RVM framing measures the same faint pixels admitted by that cutoff and caps automatic zoom at 1.8×, preventing a confident face patch from causing a face-only 4× crop.

The standalone **Upscaling Lab** accepts a local image by picker or drag-and-drop. It creates 5%, 10%, 25%, 50%, 75%, or 100% inputs relative to a fixed 512px output target while preserving aspect ratio, then reconstructs every result with the same four spatial shaders and two neural models used by the video pipeline. Each card reports elapsed time and can export its result as a PNG; RCAS can be applied consistently across all six methods.

The bundled `public/default-reference.png` is analyzed on startup. Drop another image onto the **Reference look** card to replace it; the × restores the default.

**Reinhard LAB** implements the mean/standard-deviation technique demonstrated by [chia56028/Color-Transfer-between-Images](https://github.com/chia56028/Color-Transfer-between-Images), based on *Color Transfer between Images* by Reinhard et al. It is always applied while automatic adjustments are enabled.

Music search uses Apple's public iTunes Search API through JSONP, so the GitHub Pages build needs no token or proxy. Preview audio is streamed directly from Apple's CDN, loops until paused, and is not cached. The light slider remains the baseline and visibly follows a smoothed volume envelope plus bass-weighted spectral-flux beat pulses. Dragging it temporarily takes manual control; music resumes driving it on release.

Apple publishes these clips as promotional previews and its terms restrict synchronizing them with visual media. Treat the reactive preview integration as a prototype; a production release should drive the effect from user-owned audio or a catalog whose license explicitly permits audiovisual synchronization.

## Run locally

Requirements: a modern browser with WebGL 2 and Node.js 20 or newer. WebGPU enables the faster RVM TensorFlow.js and LiteRT backends, WebSR, and FSRCNN; RVM automatically retains its SIMD WASM fallback.

```sh
npm install
npm run dev
```

Open the local URL shown by Vite. Drop videos onto the dashed tile or connect a webcam. Reference images and video frames never leave the browser.

To verify the production build:

```sh
npm run build
npm run preview
```

## GitHub Pages

The Vite build uses relative asset paths, so it works from the repository subpath at `/ai-relighting/`. Pushing `main` runs [the Pages workflow](.github/workflows/deploy-pages.yml), builds `dist/`, and deploys it with GitHub's official Pages actions.

The MediaPipe WASM runtime and model are currently fetched from public CDNs. Webcam access requires HTTPS, which GitHub Pages provides.

## Production follow-ups

- Move segmentation into a worker when the desired MediaPipe GPU/worker combination is reliable.
- Add full-resolution, edge-aware alpha refinement for finer hair detail.
- Bundle the MediaPipe runtime and model locally for a fully self-contained deployment.
- Add a shared 3D LUT, skin-tone constraints, and WebCodecs export.
