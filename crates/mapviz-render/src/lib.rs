//! The wgpu rendering backend for mapviz.
//!
//! This is the reference renderer: it is the baseline every feature is designed
//! and tested against. WGSL shaders live next to the modules that use them and
//! are pulled in with `include_str!`.
//!
//! `unsafe` is permitted in this crate (and only this crate) for GPU buffer
//! casts, preferably via `bytemuck`.
//!
//! For now this exposes a concrete [`Renderer`]. A backend-agnostic `Backend`
//! trait is deferred until there is a second primitive or backend to design it
//! against (see `CLAUDE.md`).

mod renderer;

pub use renderer::{RenderError, Renderer};
