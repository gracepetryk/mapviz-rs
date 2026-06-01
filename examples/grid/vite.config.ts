import { defineConfig } from "vite";

// The wasm package is a local `file:` dependency rebuilt by `wasm-pack`. Pre-
// bundling it caches the JS glue under node_modules/.vite, which then drifts out
// of sync with a freshly built mapviz_bg.wasm — producing errors like
// "import ./mapviz_bg.js:__wbg_map_new must be an object". Excluding it makes
// Vite serve the glue straight from the symlinked pkg, always matching the wasm.
export default defineConfig({
  optimizeDeps: {
    exclude: ["mapviz-wasm"],
  },
});
