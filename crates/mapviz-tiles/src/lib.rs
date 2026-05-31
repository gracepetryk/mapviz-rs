//! Tile sources for mapviz.
//!
//! XYZ raster and Mapbox Vector Tile sources, with caching and per-layer LOD.
//! Tile fetches are async; decoding is gated behind features to keep the wasm
//! bundle small.
