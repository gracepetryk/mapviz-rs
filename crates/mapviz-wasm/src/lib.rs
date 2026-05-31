//! The wasm-bindgen JS/TS API for mapviz.
//!
//! This is the only crate that targets `wasm32-unknown-unknown`. It is a thin
//! adapter over the Rust API: a small, typed surface (`Map`, `Layer`, `Camera`,
//! `DataSource`) with TypeScript declarations generated from Rust. WASM is an
//! implementation detail; errors cross to JS as typed results rather than
//! panicking across the FFI boundary.
