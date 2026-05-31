//! Built-in layer implementations for mapviz.
//!
//! Each layer implements the core [`Layer`](mapviz_core::Layer) trait and emits
//! backend-agnostic primitive data into the frame — layers never touch a
//! rendering backend directly. Users add their own layers by implementing the
//! same trait.

use mapviz_core::{Frame, Layer, LineInstance, Polyline, Primitive, QuadInstance};

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

/// A layer of solid-colored line segments.
///
/// Like [`QuadLayer`], the lines are fixed at construction and copied into the
/// frame each time the layer is prepared.
pub struct LineLayer {
    lines: Vec<LineInstance>,
}

impl LineLayer {
    /// A layer from an explicit set of line segments.
    pub fn new(lines: Vec<LineInstance>) -> Self {
        Self { lines }
    }

    /// An axis-aligned rectangle outline (4 segments) from `min` to `max`.
    pub fn rect(min: [f32; 2], max: [f32; 2], width: f32, color: [f32; 4]) -> Self {
        let [x0, y0] = min;
        let [x1, y1] = max;
        let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        let lines = (0..4)
            .map(|i| LineInstance::new(corners[i], corners[(i + 1) % 4], width, color))
            .collect();
        Self::new(lines)
    }

    /// The lines this layer will emit.
    pub fn lines(&self) -> &[LineInstance] {
        &self.lines
    }
}

impl Layer for LineLayer {
    fn prepare(&mut self, frame: &mut Frame) {
        frame.push(Primitive::Lines(self.lines.clone()));
    }
}

/// A layer of multi-vertex polylines (thick stroked paths).
///
/// Each [`Polyline`] is expanded into [`LineInstance`] segments at
/// [`prepare`](Layer::prepare) time via [`Polyline::expand`]. The resulting
/// segments are emitted as a single `Primitive::Lines` batch — one batch per
/// `prepare` call regardless of how many polylines the layer holds.
///
/// This maps directly to the MVT (Mapbox Vector Tile spec 2.1) `LINESTRING`
/// geometry type: each MVT linestring becomes one [`Polyline`].
pub struct PolylineLayer {
    polylines: Vec<Polyline>,
}

impl PolylineLayer {
    /// A layer from a list of polylines.
    pub fn new(polylines: Vec<Polyline>) -> Self {
        Self { polylines }
    }

    /// A layer containing a single polyline built from the given `points`,
    /// `width`, and `color`.
    pub fn from_points(points: Vec<[f32; 2]>, width: f32, color: [f32; 4]) -> Self {
        Self::new(vec![Polyline::new(points, width, color)])
    }

    /// The polylines held by this layer.
    pub fn polylines(&self) -> &[Polyline] {
        &self.polylines
    }
}

impl Layer for PolylineLayer {
    fn prepare(&mut self, frame: &mut Frame) {
        let segments: Vec<LineInstance> = self
            .polylines
            .iter()
            .flat_map(|pl| pl.expand())
            .collect();
        frame.push(Primitive::Lines(segments));
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
    fn quad_layer_prepare_pushes_one_quad_batch() {
        let mut layer = QuadLayer::grid(2, 2, 1.0, 0.5);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.primitives.len(), 1);
        match &frame.primitives[0] {
            Primitive::Quads(quads) => assert_eq!(quads.len(), 4),
            other => panic!("expected a quad batch, got {other:?}"),
        }
    }

    #[test]
    fn rect_has_four_segments() {
        let layer = LineLayer::rect([-1.0, -1.0], [1.0, 1.0], 0.1, [1.0; 4]);
        assert_eq!(layer.lines().len(), 4);
    }

    #[test]
    fn line_layer_prepare_pushes_one_line_batch() {
        let mut layer = LineLayer::rect([0.0, 0.0], [1.0, 1.0], 0.1, [1.0; 4]);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.primitives.len(), 1);
        match &frame.primitives[0] {
            Primitive::Lines(lines) => assert_eq!(lines.len(), 4),
            other => panic!("expected a line batch, got {other:?}"),
        }
    }

    #[test]
    fn polyline_layer_n_points_produces_n_minus_1_segments() {
        // 5-point path → 4 segments.
        let points = vec![
            [0.0f32, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 0.0],
        ];
        let mut layer = PolylineLayer::from_points(points, 0.1, [1.0, 0.0, 0.0, 1.0]);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.primitives.len(), 1);
        match &frame.primitives[0] {
            Primitive::Lines(lines) => assert_eq!(lines.len(), 4),
            other => panic!("expected a line batch, got {other:?}"),
        }
    }

    #[test]
    fn polyline_layer_multiple_polylines_in_one_batch() {
        // Two 3-point polylines → 2 + 2 = 4 segments in one batch.
        let pl1 = Polyline::new(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], 0.1, [1.0; 4]);
        let pl2 = Polyline::new(vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], 0.1, [1.0; 4]);
        let mut layer = PolylineLayer::new(vec![pl1, pl2]);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.primitives.len(), 1, "all polylines in one batch");
        match &frame.primitives[0] {
            Primitive::Lines(lines) => assert_eq!(lines.len(), 4),
            other => panic!("expected a line batch, got {other:?}"),
        }
    }

    #[test]
    fn polyline_layer_degenerate_fewer_than_2_points_emits_empty_batch() {
        let mut layer = PolylineLayer::from_points(vec![[0.0, 0.0]], 0.1, [1.0; 4]);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        match &frame.primitives[0] {
            Primitive::Lines(lines) => assert!(lines.is_empty()),
            other => panic!("expected an empty line batch, got {other:?}"),
        }
    }
}
