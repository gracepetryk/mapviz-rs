//! The wgpu rendering backend for mapviz.
//!
//! This is the reference `Backend` implementation: it is guaranteed to support
//! the entire rendering contract, 2D and 3D, and is the baseline every feature
//! is designed and tested against. WGSL shaders live next to the modules that
//! use them and are pulled in with `include_str!`.
//!
//! `unsafe` is permitted in this crate (and only this crate) for GPU buffer
//! casts, preferably via `bytemuck`.
