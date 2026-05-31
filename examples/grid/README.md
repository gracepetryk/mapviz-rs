# grid example

A pannable, zoomable grid of solid-colored squares — the first mapviz 2D render.
Drag to pan, scroll to zoom toward the cursor.

## Run

From the repo root, build the wasm package, then start the dev server:

```sh
wasm-pack build crates/mapviz-wasm --target web --out-name mapviz
cd examples/grid
npm install
npm run dev
```

Open the printed URL in a WebGPU-capable browser (Chrome/Edge, or Safari 26+).

Re-run the `wasm-pack build` step whenever you change Rust code, then refresh.
