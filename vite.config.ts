import { defineConfig } from "vite";
import typegpu from "unplugin-typegpu/vite";

export default defineConfig({
  // Relative asset URLs work locally and under /ai-relighting/ on GitHub Pages.
  base: "./",
  plugins: [typegpu()],
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        upscaling: "upscaling.html"
      }
    }
  }
});
