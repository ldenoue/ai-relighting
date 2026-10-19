# Relight Lab

A browser-only experiment that makes uploaded videos and webcam feeds feel as though they share one lighting environment. Frames stay on the device: decoding, person segmentation, compositing, color matching, and rendering all happen in the browser.

**Live demo:** [ldenoue.github.io/ai-relighting](https://ldenoue.github.io/ai-relighting/)

## What it does

- Runs MediaPipe Selfie Multiclass Segmentation at 256 × 256 in the browser.
- Pairs each segmentation result with its exact source frame to avoid mask lag.
- Letterboxes sources before processing and automatically frames each person to roughly 80% of the tile height.
- Transfers person and background color statistics from a reference image.
- Replaces source backgrounds with the dominant background color inferred from that reference.
- Builds a layered backlight from a radial source, background bloom, confidence-aware rim lighting, inward spill, and a subtle floor pool.
- Uses MediaPipe's continuous confidence as optical alpha, so solid pixels occlude the backlight while uncertain hair and edge pixels transmit more of it.
- Stores dropped video blobs in IndexedDB for the current browser session.
- Supports square and portrait presentation without changing the processing pipeline.

## How the pipeline works

1. Each local video or webcam is decoded by the browser and contained in a 256 × 256 frame.
2. A shared MediaPipe model produces a soft person confidence matte.
3. A GPU filter refines the matte, and a smoothed view transform normalizes subject framing.
4. Foreground luminance is measured and temporally stabilized to avoid exposure pumping.
5. A separate reference-image pass measures foreground/background RGB distributions and finds the dominant backdrop color.
6. WebGL 2 applies color transfer, exposure, background replacement, alpha compositing, and backlight layers.

The bundled `public/default-reference.png` is analyzed on startup. Drop another image onto the **Reference look** card to replace it; the × restores the default.

## Run locally

Requirements: a modern browser with WebGL 2 and Node.js 20 or newer.

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
