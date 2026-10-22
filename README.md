# Relight Lab

A browser-only experiment that makes uploaded videos and webcam feeds feel as though they share one lighting environment. Frames stay on the device: decoding, person segmentation, compositing, color matching, and rendering all happen in the browser.

**Live demo:** [ldenoue.github.io/ai-relighting](https://ldenoue.github.io/ai-relighting/)

**Upscaling lab:** [ldenoue.github.io/ai-relighting/upscaling.html](https://ldenoue.github.io/ai-relighting/upscaling.html)

## What it does

- Runs MediaPipe Selfie Multiclass Segmentation at 256 × 256 while retaining a synchronized high-resolution source snapshot.
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
2. A shared MediaPipe model produces a soft person confidence matte that stays paired with the original high-resolution snapshot.
3. A GPU filter refines the matte, and a smoothed view transform normalizes subject framing.
4. Foreground luminance is measured and temporally stabilized to avoid exposure pumping.
5. A separate reference-image pass measures foreground/background RGB distributions and finds the dominant backdrop color.
6. WebGL 2 applies Reinhard CIELAB transfer, followed by background replacement, alpha compositing, and backlight layers into a shared 320 × 320 working frame.
7. Every method reconstructs the working frame at 640 × 640. Square and portrait controls remain CSS-only, so they do not change the processed pixels used in comparisons.

All six choices intentionally share one 320 × 320 input and produce a 640 × 640 result. **FSR 1.0** is the default and uses EASU's edge-directed 12-tap reconstruction; **Bilinear** is the browser baseline; **Lanczos2** uses a 16-tap kernel; **Anime4K Original** uses lightweight Sobel-guided line refinement; **WebSR CNN-S** runs the small Anime4K real-life CNN with WebGPU. **FSRCNN ×2** runs a luminance model through ONNX Runtime WebGPU and keeps chroma from a conventional resize. The independent **RCAS** toggle optionally applies the same final sharpening pass after any selected method.

The neural options load their runtimes and weights only when selected and display the latest completed result while the next frame is inferred. They require WebGPU; unsupported browsers keep showing the bilinear fallback. The first FSRCNN use downloads the roughly 24 MB ONNX Runtime WASM support module in addition to the 37 KB model. Implementations, model provenance, and licenses are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The standalone **Upscaling Lab** accepts a local image by picker or drag-and-drop. It creates 5%, 10%, 25%, 50%, 75%, or 100% inputs relative to a fixed 512px output target while preserving aspect ratio, then reconstructs every result with the same four spatial shaders and two neural models used by the video pipeline. Each card reports elapsed time and can export its result as a PNG; RCAS can be applied consistently across all six methods.

The bundled `public/default-reference.png` is analyzed on startup. Drop another image onto the **Reference look** card to replace it; the × restores the default.

**Reinhard LAB** implements the mean/standard-deviation technique demonstrated by [chia56028/Color-Transfer-between-Images](https://github.com/chia56028/Color-Transfer-between-Images), based on *Color Transfer between Images* by Reinhard et al. It is always applied while automatic adjustments are enabled.

Music search uses Apple's public iTunes Search API through JSONP, so the GitHub Pages build needs no token or proxy. Preview audio is streamed directly from Apple's CDN, loops until paused, and is not cached. The light slider remains the baseline and visibly follows a smoothed volume envelope plus bass-weighted spectral-flux beat pulses. Dragging it temporarily takes manual control; music resumes driving it on release.

Apple publishes these clips as promotional previews and its terms restrict synchronizing them with visual media. Treat the reactive preview integration as a prototype; a production release should drive the effect from user-owned audio or a catalog whose license explicitly permits audiovisual synchronization.

## Run locally

Requirements: a modern browser with WebGL 2 and Node.js 20 or newer. WebGPU is additionally required for WebSR and FSRCNN.

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
