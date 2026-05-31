//! The wasm-bindgen JS/TS API for mapviz.
//!
//! This is the only crate that targets `wasm32-unknown-unknown`. It is a thin
//! adapter over the Rust API: a small, typed surface (`Map`, and later `Layer`,
//! `Camera`, `DataSource`) with TypeScript declarations generated from Rust.
//! WASM is an implementation detail; errors cross to JS as typed results rather
//! than panicking across the FFI boundary.
//!
//! The interop surface is `#[cfg(target_arch = "wasm32")]`-gated so the crate
//! still compiles to an (empty) native library for CI.

#[cfg(target_arch = "wasm32")]
mod map;

#[cfg(target_arch = "wasm32")]
pub use map::Map;
