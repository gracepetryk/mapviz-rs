//! The per-frame draw list.
//!
//! A `Frame` is the backend-agnostic hand-off between layers and the renderer:
//! layers push [`Shape`]s into it during `prepare`, and a backend tessellates
//! and draws them in order to produce an image. Because shapes keep their
//! submission order, layer order *is* render order — a shape pushed later draws
//! on top of one pushed earlier.

use crate::geometry::Shape;

/// An ordered list of styled geometries to draw this frame.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Shapes, in the order layers emitted them (render / painter's order).
    pub shapes: Vec<Shape>,
}

impl Frame {
    /// An empty frame.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a shape to the draw list.
    pub fn push(&mut self, shape: Shape) {
        self.shapes.push(shape);
    }

    /// Drop all shapes, keeping allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.shapes.clear();
    }
}
