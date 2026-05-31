//! Built-in layer implementations for mapviz.
//!
//! Each layer implements the core [`Layer`](mapviz_core::Layer) trait and emits
//! backend-agnostic primitive data into the frame — layers never touch a
//! rendering backend directly. Users add their own layers by implementing the
//! same trait.

use mapviz_core::{Frame, FillVertex, Layer, LineInstance, Primitive, QuadInstance};

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

/// A filled-polygon layer backed by pre-tessellated triangle mesh data.
///
/// Polygon geometry is tessellated once at construction time (via
/// [`mapviz_core::tessellate`]) and the resulting `(vertices, indices)` are
/// cached. Each `prepare` call emits a single [`Primitive::Mesh`] batch.
///
/// Coordinate convention follows **MVT 2.1**: exterior rings are wound
/// counter-clockwise; interior rings (holes) are wound clockwise.
pub struct PolygonLayer {
    vertices: Vec<FillVertex>,
    indices: Vec<u32>,
}

impl PolygonLayer {
    /// Build a polygon layer from pre-tessellated vertex/index data.
    ///
    /// This is the low-level constructor. Prefer [`PolygonLayer::from_ring`] or
    /// [`PolygonLayer::with_holes`] for typical use.
    pub fn from_mesh(vertices: Vec<FillVertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Build a filled polygon from a single exterior ring (no holes).
    ///
    /// Returns an error if tessellation fails (e.g. fewer than 3 points).
    pub fn from_ring(
        exterior: &[[f32; 2]],
        color: [f32; 4],
    ) -> mapviz_core::Result<Self> {
        let (vertices, indices) = mapviz_core::tessellate(exterior, &[], color)?;
        Ok(Self::from_mesh(vertices, indices))
    }

    /// Build a filled polygon with an exterior ring and one or more hole rings.
    ///
    /// Hole rings should be wound clockwise (MVT 2.1 convention).
    /// Returns an error if tessellation fails.
    pub fn with_holes(
        exterior: &[[f32; 2]],
        holes: &[Vec<[f32; 2]>],
        color: [f32; 4],
    ) -> mapviz_core::Result<Self> {
        let (vertices, indices) = mapviz_core::tessellate(exterior, holes, color)?;
        Ok(Self::from_mesh(vertices, indices))
    }

    /// The tessellated vertices this layer will emit.
    pub fn vertices(&self) -> &[FillVertex] {
        &self.vertices
    }

    /// The triangle indices this layer will emit.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}

impl Layer for PolygonLayer {
    fn prepare(&mut self, frame: &mut Frame) {
        frame.push(Primitive::Mesh {
            vertices: self.vertices.clone(),
            indices: self.indices.clone(),
        });
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
    fn polygon_layer_from_ring_produces_valid_mesh() {
        // A unit square exterior.
        let ring = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let layer = PolygonLayer::from_ring(&ring, [1.0, 0.0, 0.0, 1.0])
            .expect("tessellation should succeed");
        assert_eq!(layer.vertices().len(), 4, "4 unique vertices");
        assert_eq!(layer.indices().len(), 6, "2 triangles = 6 indices");
        for &i in layer.indices() {
            assert!((i as usize) < layer.vertices().len());
        }
    }

    #[test]
    fn polygon_layer_prepare_pushes_one_mesh_batch() {
        let ring = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut layer = PolygonLayer::from_ring(&ring, [0.5, 0.5, 0.5, 1.0])
            .expect("tessellation should succeed");
        let mut frame = mapviz_core::Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.primitives.len(), 1);
        match &frame.primitives[0] {
            Primitive::Mesh { vertices, indices } => {
                assert_eq!(vertices.len(), 4);
                assert_eq!(indices.len(), 6);
            }
            other => panic!("expected a mesh batch, got {other:?}"),
        }
    }

    #[test]
    fn polygon_layer_with_hole_has_expected_vertex_count() {
        let exterior = vec![[0.0f32, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let hole = vec![[0.5f32, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]];
        let layer = PolygonLayer::with_holes(&exterior, &[hole], [0.0, 1.0, 0.0, 1.0])
            .expect("tessellation should succeed");
        assert_eq!(layer.vertices().len(), 8, "4 outer + 4 inner vertices");
        assert_eq!(layer.indices().len(), 24, "8 triangles = 24 indices");
    }
}
