//! Tile sources for mapviz.
//!
//! XYZ raster and Mapbox Vector Tile sources, with caching and per-layer LOD.
//! Tile fetches are async; decoding is gated behind features to keep the wasm
//! bundle small.
//!
//! Raster decoding turns encoded image bytes into a core
//! [`TextureImage`](mapviz_core::TextureImage) — the same backend-agnostic pixel
//! type a textured [`Shape`](mapviz_core::Shape) carries — so callers hand us
//! bytes (a fetched tile) and never deal with pixel formats themselves.

#[cfg(feature = "png")]
mod raster;

#[cfg(feature = "png")]
pub use raster::decode_png;
