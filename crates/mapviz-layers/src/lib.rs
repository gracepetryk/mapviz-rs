//! Built-in layer implementations for mapviz.
//!
//! Each layer implements the core [`Layer`](mapviz_core::Layer) trait and emits
//! backend-agnostic primitive data into the frame — layers never touch a
//! rendering backend directly. Users add their own layers by implementing the
//! same trait.

use mapviz_core::{Frame, Layer, Primitive, QuadInstance};

/// A layer of solid-colored quads.
///
/// The quads are fixed at construction; the layer simply copies them into the
/// frame each time it's prepared. Useful on its own and as the test pattern for
/// bringing up 2D rendering (see [`QuadLayer::grid`]).
pub struct QuadLayer {
    quads: Vec<QuadInstance>,
}

impl QuadLayer {
    /// A layer from an explicit set of quads.
    pub fn new(quads: Vec<QuadInstance>) -> Self {
        Self { quads }
    }

    /// A `cols × rows` grid of squares centered on the world origin.
    ///
    /// `spacing` is the world-space distance between adjacent square centers and
    /// `square_size` is the full side length of each square. Each cell gets a
    /// deterministic color that varies across the grid (red along +x, green
    /// along +y), so the orientation is obvious while panning and zooming.
    pub fn grid(cols: u32, rows: u32, spacing: f32, square_size: f32) -> Self {
        let half = square_size * 0.5;
        let origin_x = -((cols.saturating_sub(1)) as f32) * spacing * 0.5;
        let origin_y = -((rows.saturating_sub(1)) as f32) * spacing * 0.5;
        let denom_x = (cols.saturating_sub(1)).max(1) as f32;
        let denom_y = (rows.saturating_sub(1)).max(1) as f32;

        let mut quads = Vec::with_capacity((cols * rows) as usize);
        for row in 0..rows {
            for col in 0..cols {
                let center = [
                    origin_x + col as f32 * spacing,
                    origin_y + row as f32 * spacing,
                ];
                let color = [col as f32 / denom_x, row as f32 / denom_y, 0.5, 1.0];
                quads.push(QuadInstance::square(center, half, color));
            }
        }
        Self::new(quads)
    }

    /// The quads this layer will emit.
    pub fn quads(&self) -> &[QuadInstance] {
        &self.quads
    }
}

impl Layer for QuadLayer {
    fn prepare(&mut self, frame: &mut Frame) {
        frame.push(Primitive::Quads(self.quads.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_has_one_quad_per_cell_centered_on_origin() {
        let layer = QuadLayer::grid(4, 3, 2.0, 1.0);
        assert_eq!(layer.quads().len(), 12);

        // Centers are symmetric about the origin, so their mean is ~zero.
        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for q in layer.quads() {
            sx += q.center[0];
            sy += q.center[1];
        }
        assert!(sx.abs() < 1e-4 && sy.abs() < 1e-4);
    }

    #[test]
    fn prepare_pushes_one_quad_batch() {
        let mut layer = QuadLayer::grid(2, 2, 1.0, 0.5);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.primitives.len(), 1);
        match &frame.primitives[0] {
            Primitive::Quads(quads) => assert_eq!(quads.len(), 4),
        }
    }
}
