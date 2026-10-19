# Relight Lab

A browser-only experiment that makes uploaded videos and webcam feeds feel as though they share one lighting environment. Frames stay on the device: decoding, person segmentation, compositing, color matching, and rendering all happen in the browser.

**Live demo:** [ldenoue.github.io/ai-relighting](https://ldenoue.github.io/ai-relighting/)

## What it does

- Runs MediaPipe Selfie Multiclass Segmentation at 256 × 256 while retaining a synchronized display frame at up to 1280 × 1280.
- Derives the model input from that high-resolution snapshot and pairs each result back to the same frame, avoiding both mask lag and zoomed 256-pixel video.
- Letterboxes sources before processing and automatically frames each person to roughly 80% of the tile height.
- Transfers person and background color statistics from a reference image.
- Optionally applies Reinhard color transfer in CIELAB (per-channel target mean and standard deviation); when disabled, the existing restrained RGB transfer remains active.
- Replaces source backgrounds with the dominant background color inferred from that reference.
- Builds a layered backlight from a radial source, background bloom, confidence-aware rim lighting, inward spill, and a subtle floor pool.
- Searches Apple's public catalog for streamed 30-second previews and uses a Web Audio VU envelope to modulate the backlight intensity.
- Uses MediaPipe's continuous confidence as optical alpha, so solid pixels occlude the backlight while uncertain hair and edge pixels transmit more of it.
- Stores dropped video blobs in IndexedDB for the current browser session.
- Supports square and portrait presentation without changing the processing pipeline.

## How the pipeline works

1. Each local video or webcam is decoded once into a high-resolution square snapshot; a 256 × 256 copy is derived from it for inference.
2. A shared MediaPipe model produces a soft person confidence matte that stays paired with the original high-resolution snapshot.
3. A GPU filter refines the matte, and a smoothed view transform normalizes subject framing.
4. Foreground luminance is measured and temporally stabilized to avoid exposure pumping.
5. A separate reference-image pass measures foreground/background RGB distributions and finds the dominant backdrop color.
6. WebGL 2 applies the default RGB transfer or optional Reinhard CIELAB transfer, followed by background replacement, alpha compositing, and backlight layers.

The bundled `public/default-reference.png` is analyzed on startup. Drop another image onto the **Reference look** card to replace it; the × restores the default.

**Reinhard LAB** implements the mean/standard-deviation technique demonstrated by [chia56028/Color-Transfer-between-Images](https://github.com/chia56028/Color-Transfer-between-Images), based on *Color Transfer between Images* by Reinhard et al. It is on by default; turning the toggle off restores the existing restrained RGB method.

Music search uses Apple's public iTunes Search API through JSONP, so the GitHub Pages build needs no token or proxy. Preview audio is streamed directly from Apple's CDN, loops until paused, and is not cached. The light slider remains the baseline and visibly follows a smoothed volume envelope plus bass-weighted spectral-flux beat pulses. Dragging it temporarily takes manual control; music resumes driving it on release.

Apple publishes these clips as promotional previews and its terms restrict synchronizing them with visual media. Treat the reactive preview integration as a prototype; a production release should drive the effect from user-owned audio or a catalog whose license explicitly permits audiovisual synchronization.

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
