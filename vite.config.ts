import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset URLs work locally and under /ai-relighting/ on GitHub Pages.
  base: "./",
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        upscaling: "upscaling.html"
      }
    }
  }
});
