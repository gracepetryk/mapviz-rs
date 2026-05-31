//! The per-frame draw list.
//!
//! A `Frame` is the backend-agnostic hand-off between layers and the renderer:
//! layers push primitive batches into it during `prepare`, and a backend
//! consumes them in order to produce a rendered image. Because batches keep
//! their submission order, layer order *is* render order across primitive
//! kinds — a line batch pushed after a quad batch draws on top of it.

use crate::primitive::Primitive;

/// An ordered list of primitive batches to draw this frame.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Primitive batches, in the order layers emitted them (render order).
    pub primitives: Vec<Primitive>,
}

impl Frame {
    /// An empty frame.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a primitive batch to the draw list.
    pub fn push(&mut self, primitive: Primitive) {
        self.primitives.push(primitive);
    }

    /// Drop all primitives, keeping allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.primitives.clear();
    }
}
