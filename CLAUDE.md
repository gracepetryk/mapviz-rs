# mapviz

A Rust + WebGPU library, compiled to WASM, for building beautiful interactive 2D and 3D spatial visualizations in the browser.

The library is **generic**: it provides primitives (coordinate systems, cameras, tiled basemaps, composable rendering layers, time-aware data sources) that any map-style project can build on. The driving example — live ADS-B flight tracks with dead-reckoned aircraft positions, 3D trajectories, and a globe view — is a stress test for the API, not a built-in feature. If something only makes sense for flight tracking, it belongs in an example, not the core.

## Guiding principles

- **Data-oriented scene description.** Layers describe *what* to draw as plain, backend-agnostic data (a `Frame` of `Primitive` batches); the renderer decides *how and when* to draw it. Geometry, projections, scene graph, and animation logic live in the core and must not depend on `wgpu` or any GPU types — that dependency stops at the `Frame`/`Primitive` boundary. This boundary is the decoupling that matters: it makes the scene logic (projections, interpolation, tessellation, LOD, label placement) unit-testable with no GPU, lets the renderer change its GPU strategy without touching layers, gives whole-frame visibility for global optimization (batching, culling), and fits the "produce the scene at time *t*" model. We keep it for those day-one benefits, not to swap backends.
- **wgpu is the renderer and our portability layer.** `mapviz-render` is tightly coupled to `wgpu` by design; we use its full feature set (compute shaders, storage buffers, whatever makes the visualization good) without rationing for weaker targets. wgpu already abstracts the GPU API underneath (WebGPU, and a WebGL2 fallback on the web), so it *is* our second-backend insurance for the only realistic case — browsers without WebGPU. We do not build our own `Backend` trait or capability-tiering system for that.
- **A custom backend trait is deferred until a non-wgpu target is real.** The only thing wgpu can't give us is a *non-GPU* consumer (SVG export, deterministic headless snapshot rendering). If one of those becomes a concrete need, we add a trait then, designed against that real case — and the `Frame`/`Primitive` boundary already gives us the seam to plug it in cheaply. Until then, the public API talks to the concrete wgpu renderer.
- **Streaming-first data model.** Real-time feeds (ADS-B, GPS, sensor streams) are first-class. Static datasets are just a degenerate streaming case.
- **Time is a dimension.** Every entity and layer can be timestamped; the renderer interpolates and the camera can scrub. This avoids bolt-on animation systems later.
- **Composable layers.** A scene is an ordered stack of layers (raster tiles, vector tiles, points, lines, polygons, billboards, meshes, heatmaps, custom). Users register custom layers via a small trait.
- **Small, typed JS/TS surface.** WASM is an implementation detail. The public JS API is a handful of classes (`Map`, `Layer`, `Camera`, `DataSource`) with TypeScript declarations generated from Rust.
- **No hidden globals, no panics across the FFI boundary.** Errors cross to JS as typed results.

## Architecture

```
mapviz/
├── crates/
│   ├── mapviz-core/      # geometry, projections, scene, camera, time — no GPU, no wasm
│   ├── mapviz-render/    # wgpu renderer, shaders, GPU resource management
│   ├── mapviz-tiles/     # tile sources (XYZ raster, MVT vector), caching, LOD
│   ├── mapviz-layers/    # built-in layer implementations
│   ├── mapviz-data/      # streaming data sources, spatial/temporal indices
│   └── mapviz-wasm/      # wasm-bindgen JS API, the only crate that targets wasm32
└── examples/
    ├── adsb/             # live ADS-B flight tracker (driving example)
    ├── choropleth/       # static polygon data
    └── heatmap/          # point density
```

### Core concepts

- **Coordinate systems:** WGS84 (lon/lat/alt), ECEF (earth-centered earth-fixed, for 3D globe math), local ENU tangent planes, normalized device coords. Conversions are explicit and typed — no raw `[f64; 3]` shuffling.
- **Projections:** Web Mercator (2D default), equirectangular, and a true 3D globe projection. Projections are a trait so users can add their own (UTM zones, polar stereographic, etc.).
- **Camera:** 2D (pan/zoom/rotate) and 3D (orbit, free-fly, locked-to-target). Camera state is plain data; controllers are separate and swappable.
- **Layers:** trait `Layer { fn prepare(&mut self, frame: &mut Frame); }`. A layer emits backend-agnostic `Primitive` batches into the `Frame` — it never issues GPU calls or touches a render pass, which is what keeps `wgpu` out of the scene logic. Built-ins live in `mapviz-layers`; users implement the same trait for custom layers. (Power users who need their own shader/pass will get an explicit escape-hatch primitive later; the data model covers the built-in primitive vocabulary.)
- **Data sources:** trait that produces typed features over time. Adapters for static GeoJSON, WebSocket streams, server-sent events, and arbitrary user-driven push. Spatial index (R*-tree) and temporal index built in.
- **Picking:** GPU id-buffer pass plus CPU spatial-index fallback. Returns the feature, not a pixel.
- **Labels & icons:** SDF text rendering with collision-resolved placement; billboards with screen-space sizing.

### Rendering

- The contract between scene and renderer is **data, not a trait**: a `Frame` of ordered `Primitive` batches (`mapviz-core`), which `mapviz-render` consumes. Layers fill the `Frame`; the renderer owns everything about turning it into pixels. There is no `Backend` trait today — `mapviz-render` is concretely wgpu, and the public API talks to it directly (see Guiding principles for when that would change).
- `Primitive` is a *batch* enum (e.g. `Quads(Vec<QuadInstance>)`), not one-shape-per-entry: each variant is a contiguous run drawable as a single instanced pass, and the order of batches in the `Frame` is the render (painter's) order. New primitive kinds are new variants, not new fields. `mapviz-render` mirrors each with a GPU-layout type (`GpuPrimitive`/`GpuQuad`, `bytemuck::Pod`) via `From`, so the `bytemuck` casts stay in the render crate and core has no GPU types.
- 3D is the same `Frame`/`Primitive` flow with a 3D camera matrix plus a depth attachment; no separate capability negotiation, since wgpu does everything.
- Shaders in WGSL, owned by `mapviz-render`, included with `include_str!`. Share the camera/uniform bind group across pipelines.
- Use instancing for points/billboards/aircraft-style entities. Filled polygons / meshes use indexed draws instead — the renderer keeps a small holder per *draw model* (instanced vs indexed), not per primitive type.
- LOD is per-layer, not global — tile layers care about zoom, point layers care about screen density.

### Time and interpolation

- Scene clock is independent of wall clock; supports play/pause/scrub at arbitrary rates.
- `Trajectory<T>` primitive: timestamped samples + an interpolator (linear, hermite, great-circle, dead-reckoning from velocity+heading). The ADS-B example uses dead-reckoning between updates; a GPS playback example would use hermite.
- Layers subscribe to clock ticks; the renderer only redraws dirty layers.

## Dependencies (planned, with rationale)

**Core math & geometry**
- `glam` — fast SIMD vector/matrix math, integrates with wgpu and bytemuck.
- `geo` + `geo-types` — geometry primitives, predicates, simplification.
- `rstar` — R*-tree spatial index for picking and viewport culling.
- `earcutr` — polygon triangulation.
- `lyon` — 2D path tessellation for vector strokes/fills.

**Rendering**
- `wgpu` — WebGPU backend (works on web via the browser's WebGPU, native via Vulkan/Metal/DX12).
- `bytemuck` — safe POD casts for GPU buffer uploads.
- `wgsl_to_wgsl` or inline WGSL — keep shaders versioned alongside Rust.
- `fontdue` or `cosmic-text` + an SDF generator for labels.

**Tiles & formats**
- `geozero` — zero-copy reads of GeoJSON, MVT, FlatGeobuf, WKB.
- `prost` — protobuf decoding for Mapbox Vector Tiles.
- `image` (with selected features) — PNG/JPEG raster tile decoding. Consider `zune-png`/`zune-jpeg` for smaller WASM size.

**Web/WASM**
- `wasm-bindgen`, `web-sys`, `js-sys` — JS interop.
- `console_error_panic_hook` — readable panics during development only.
- `tracing` + `tracing-wasm` — structured logging that works in browser devtools.
- `wasm-bindgen-futures` — bridge JS promises and Rust async for tile/data fetches.

**Data & time**
- `serde` + `serde_json` — config and GeoJSON.
- `time` (not `chrono` — smaller WASM footprint).
- `futures` / `futures-util` — stream combinators for live data sources.

**Dev/test**
- `wasm-pack` for building the wasm crate.
- `wasm-bindgen-test` for browser-run tests.
- `criterion` for native benches of core math/indices.
- `insta` for snapshot tests of projections and tessellation.

Keep an eye on WASM bundle size: gate optional codecs and projections behind features.

## Public API sketch (JS side)

```ts
import { Map, TileLayer, PointLayer, TrajectoryLayer, DataSource } from "mapviz";

const map = new Map(canvas, { projection: "globe", camera: "orbit" });
map.addLayer(new TileLayer({ url: "https://tiles.example/{z}/{x}/{y}.png" }));

const aircraft = DataSource.websocket("wss://adsb.example/stream");
map.addLayer(new PointLayer(aircraft, { icon: "plane", rotate: "heading" }));
map.addLayer(new TrajectoryLayer(aircraft, { interpolate: "dead-reckon", maxAgeSec: 600 }));

map.clock.play({ rate: 1.0 });
```

The Rust API mirrors this; the WASM crate is a thin adapter.

## ADS-B example as API stress test

The example is what we measure the library against. It must exercise:

- Live WebSocket stream → `DataSource` with backpressure.
- Per-aircraft state with sparse updates; dead-reckoning interpolation between samples using ground speed + heading + vertical rate.
- 3D trajectories rendered as altitude-extruded polylines.
- Billboarded aircraft icons rotated by heading, scaled by screen size.
- Picking an aircraft to show callsign/altitude/speed.
- Switching between 2D Mercator and 3D globe without rebuilding the scene.
- Time scrubbing over the last N minutes of history.

Anything the example needs that doesn't have a natural place in the generic API is a signal to revisit the design.

## Non-goals

- A full GIS engine (no analysis, routing, geocoding).
- A styling DSL like Mapbox GL Style Spec — layers are configured in code.
- Server-side rendering.
- A hand-written WebGL2 backend. If browser reach ever demands a fallback for engines without WebGPU, we use wgpu's own WebGL2 path (accepting its limits — no compute/storage buffers), not a separate backend of our own. The project targets WebGPU first; browser-support implications are documented for consumers.

## Build & dev workflow

- `cargo check --workspace` — type-check everything including the wasm crate (with `--target wasm32-unknown-unknown` for that crate).
- `cargo test -p mapviz-core` — fast native tests for geometry/projections/time.
- `wasm-pack build crates/mapviz-wasm --target web` — produce the JS package.
- Examples are standalone Vite apps under `examples/` that consume the built wasm package via a local link.

## Conventions

- Edition 2024, MSRV tracks stable.
- `#![deny(unsafe_op_in_unsafe_fn)]` workspace-wide; `unsafe` only inside `mapviz-render` for GPU buffer casts (via `bytemuck` where possible).
- No `unwrap()` / `expect()` in library code paths reachable from FFI — return typed errors.
- Public types implement `Debug`; coordinate types also implement `Display` in a human-readable form.
- Shaders live next to the Rust module that uses them; included with `include_str!`.
