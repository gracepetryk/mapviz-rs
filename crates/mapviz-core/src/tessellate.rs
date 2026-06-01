//! GPU-free tessellation of styled `geo` geometry into draw instances.
//!
//! [`tessellate_shape`] turns a [`Shape`](crate::Shape) into a [`DrawData`]
//! bundle of flat instances ([`QuadInstance`] markers, [`LineInstance`]
//! strokes, and a triangulated [`FillVertex`] mesh) that a backend uploads and
//! draws. All the geometry math lives here, with no GPU types, so it is unit
//! testable on its own.
//!
//! Polygon fills are triangulated with `earcutr`. Ring winding follows the
//! usual convention (exterior counter-clockwise, holes clockwise), but
//! `earcutr` does not require it — it works from the raw coordinates and
//! hole-start indices.

use crate::error::{Error, Result};
use crate::geometry::{Shape, Style};
use crate::primitive::{FillVertex, LineInstance, QuadInstance};
use crate::texture::TextureHandle;
use geo::{BoundingRect, Geometry, LineString, Point, Polygon};

/// An axis-aligned quad in world space painted with a [`TextureHandle`].
///
/// `center`/`half_extent` are in world units; the image's UV `0..1` range maps
/// across the full quad. The `texture` is opaque image data — carrying it here
/// keeps the geometry→quad math GPU-free, and a backend uploads/caches the
/// pixels when it draws the quad.
#[derive(Clone, Debug, PartialEq)]
pub struct TexturedQuad {
    /// Center of the quad, in world units.
    pub center: [f32; 2],
    /// Half-width and half-height, in world units.
    pub half_extent: [f32; 2],
    /// Image to sample across the quad.
    pub texture: TextureHandle,
}

/// The flat draw instances produced from one or more [`Shape`]s.
///
/// Each field maps to a distinct draw model: `markers` are instanced quads,
/// `strokes` are instanced line segments, `fill_vertices`/`fill_indices` form
/// one indexed triangle mesh, and `textured_quads` are image-painted quads.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawData {
    /// Point markers (instanced quads).
    pub markers: Vec<QuadInstance>,
    /// Line / outline segments (instanced).
    pub strokes: Vec<LineInstance>,
    /// Triangle-mesh vertices for polygon fills.
    pub fill_vertices: Vec<FillVertex>,
    /// Triangle indices into `fill_vertices` (three per triangle).
    pub fill_indices: Vec<u32>,
    /// Image-textured quads (e.g. map tiles).
    pub textured_quads: Vec<TexturedQuad>,
}

/// Tessellate one styled geometry into [`DrawData`].
///
/// A shape's [`texture`](Shape::texture), if any, becomes a [`TexturedQuad`]
/// spanning the geometry's bounding rectangle. The fill (under) and stroke (on
/// top) from the [`Style`] are still emitted, so a texture can be layered
/// between them.
pub fn tessellate_shape(shape: &Shape) -> DrawData {
    let mut out = DrawData::default();
    add_geometry(&shape.geometry, &shape.style, &mut out);
    if let Some(texture) = &shape.texture {
        add_texture(&shape.geometry, texture, &mut out);
    }
    out
}

/// Emit a [`TexturedQuad`] covering `geom`'s axis-aligned bounding rectangle.
/// A degenerate (zero-area or empty) bounding box produces nothing.
fn add_texture(geom: &Geometry<f32>, texture: &TextureHandle, out: &mut DrawData) {
    let Some(rect) = geom.bounding_rect() else {
        return;
    };
    let c = rect.center();
    let half_extent = [rect.width() * 0.5, rect.height() * 0.5];
    if half_extent[0] <= 0.0 || half_extent[1] <= 0.0 {
        return;
    }
    out.textured_quads.push(TexturedQuad {
        center: [c.x, c.y],
        half_extent,
        texture: texture.clone(),
    });
}

fn add_geometry(geom: &Geometry<f32>, style: &Style, out: &mut DrawData) {
    match geom {
        Geometry::Point(p) => add_point(p, style, out),
        Geometry::MultiPoint(mp) => {
            for p in &mp.0 {
                add_point(p, style, out);
            }
        }
        Geometry::Line(l) => {
            if let Some(color) = style.stroke {
                let coords = [[l.start.x, l.start.y], [l.end.x, l.end.y]];
                stroke_coords(&coords, style.stroke_width, color, out);
            }
        }
        Geometry::LineString(ls) => add_linestring(ls, style, out),
        Geometry::MultiLineString(mls) => {
            for ls in &mls.0 {
                add_linestring(ls, style, out);
            }
        }
        Geometry::Polygon(poly) => add_polygon(poly, style, out),
        Geometry::MultiPolygon(mp) => {
            for poly in &mp.0 {
                add_polygon(poly, style, out);
            }
        }
        Geometry::Rect(r) => add_polygon(&r.to_polygon(), style, out),
        Geometry::Triangle(t) => add_polygon(&t.to_polygon(), style, out),
        Geometry::GeometryCollection(gc) => {
            for g in &gc.0 {
                add_geometry(g, style, out);
            }
        }
    }
}

fn add_point(p: &Point<f32>, style: &Style, out: &mut DrawData) {
    if let Some(color) = style.fill {
        out.markers
            .push(QuadInstance::square([p.x(), p.y()], style.point_size, color));
    }
}

fn add_linestring(ls: &LineString<f32>, style: &Style, out: &mut DrawData) {
    if let Some(color) = style.stroke {
        let coords = coords_of(ls);
        stroke_coords(&coords, style.stroke_width, color, out);
    }
}

fn add_polygon(poly: &Polygon<f32>, style: &Style, out: &mut DrawData) {
    if let Some(color) = style.fill {
        fill_polygon(poly, color, out);
    }
    if let Some(color) = style.stroke {
        // Polygon rings are closed, so the raw coordinates already include the
        // segment back to the start.
        stroke_coords(&coords_of(poly.exterior()), style.stroke_width, color, out);
        for ring in poly.interiors() {
            stroke_coords(&coords_of(ring), style.stroke_width, color, out);
        }
    }
}

fn fill_polygon(poly: &Polygon<f32>, color: [f32; 4], out: &mut DrawData) {
    let exterior = open_ring(poly.exterior());
    let holes: Vec<Vec<[f32; 2]>> = poly.interiors().iter().map(open_ring).collect();
    if let Ok((vertices, indices)) = tessellate(&exterior, &holes, color) {
        let base = out.fill_vertices.len() as u32;
        out.fill_vertices.extend(vertices);
        out.fill_indices.extend(indices.into_iter().map(|i| i + base));
    }
}

/// All ring coordinates as `[x, y]` pairs (closed, as `geo` stores them).
fn coords_of(ls: &LineString<f32>) -> Vec<[f32; 2]> {
    ls.coords().map(|c| [c.x, c.y]).collect()
}

/// Ring coordinates with the duplicate closing point dropped, for triangulation.
fn open_ring(ls: &LineString<f32>) -> Vec<[f32; 2]> {
    let mut v = coords_of(ls);
    if v.len() >= 2 && v.first() == v.last() {
        v.pop();
    }
    v
}

/// Emit a [`LineInstance`] for each consecutive pair of distinct points.
fn stroke_coords(coords: &[[f32; 2]], width: f32, color: [f32; 4], out: &mut DrawData) {
    let mut prev: Option<[f32; 2]> = None;
    for &p in coords {
        if let Some(a) = prev {
            if a != p {
                out.strokes.push(LineInstance::new(a, p, width, color));
            }
        }
        prev = Some(p);
    }
}

/// Tessellate a polygon (one exterior ring + zero or more holes) into a
/// `(vertices, indices)` triangle mesh.
///
/// `exterior` and each hole are `[x, y]` world-space rings *without* a repeated
/// closing point. `color` is applied to every vertex. Returns an error if the
/// exterior has fewer than three points or `earcutr` produces a malformed
/// triangulation.
pub fn tessellate(
    exterior: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    color: [f32; 4],
) -> Result<(Vec<FillVertex>, Vec<u32>)> {
    if exterior.len() < 3 {
        return Err(Error::Tessellation(
            "exterior ring must have at least 3 points".into(),
        ));
    }

    // earcutr expects a flat `Vec<f64>` of all coordinates (exterior then holes
    // in order), plus a `Vec<usize>` of the starting vertex index of each hole.
    let mut flat_coords: Vec<f64> =
        Vec::with_capacity((exterior.len() + holes.iter().map(|h| h.len()).sum::<usize>()) * 2);
    let mut hole_starts: Vec<usize> = Vec::with_capacity(holes.len());

    for &[x, y] in exterior {
        flat_coords.push(x as f64);
        flat_coords.push(y as f64);
    }
    for hole in holes {
        hole_starts.push(flat_coords.len() / 2);
        for &[x, y] in hole {
            flat_coords.push(x as f64);
            flat_coords.push(y as f64);
        }
    }

    let raw_indices = earcutr::earcut(&flat_coords, &hole_starts, 2)
        .map_err(|e| Error::Tessellation(format!("earcutr failed: {e:?}")))?;

    // Build FillVertex for every coordinate in flat_coords.
    let total_verts = flat_coords.len() / 2;
    let mut vertices = Vec::with_capacity(total_verts);
    for i in 0..total_verts {
        vertices.push(FillVertex::new(
            [flat_coords[i * 2] as f32, flat_coords[i * 2 + 1] as f32],
            color,
        ));
    }

    let indices: Vec<u32> = raw_indices.into_iter().map(|i| i as u32).collect();

    if indices.len() % 3 != 0 {
        return Err(Error::Tessellation(
            "tessellation produced a non-multiple-of-3 index count".into(),
        ));
    }
    for &idx in &indices {
        if idx as usize >= vertices.len() {
            return Err(Error::Tessellation(format!(
                "tessellation produced out-of-range index {idx} (vertex count {})",
                vertices.len()
            )));
        }
    }

    Ok((vertices, indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Shape;
    use geo::{Coord, LineString, MultiPoint, Point, Polygon};

    /// A unit square: 4 corners → 2 triangles → 6 indices.
    #[test]
    fn square_produces_two_triangles() {
        let ring = vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (verts, indices) = tessellate(&ring, &[], [1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(verts.len(), 4, "four unique vertices");
        assert_eq!(indices.len(), 6, "two triangles = 6 indices");
        for &i in &indices {
            assert!((i as usize) < verts.len());
        }
        for v in &verts {
            assert_eq!(v.color, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    /// A square with a square hole: 4 outer + 4 inner vertices, 8 triangles.
    #[test]
    fn square_with_hole_has_expected_triangle_count() {
        let exterior = vec![[0.0f32, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let hole = vec![[0.5f32, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]];
        let (verts, indices) = tessellate(&exterior, &[hole], [0.0, 1.0, 0.0, 1.0]).unwrap();
        assert_eq!(verts.len(), 8, "4 outer + 4 inner vertices");
        assert_eq!(indices.len(), 24, "8 triangles = 24 indices");
        for &i in &indices {
            assert!((i as usize) < verts.len(), "index {i} out of range");
        }
    }

    #[test]
    fn too_few_points_returns_error() {
        let result = tessellate(&[[0.0, 0.0], [1.0, 0.0]], &[], [1.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn point_fill_makes_a_marker() {
        let shape = Shape::new(Point::new(3.0f32, 4.0), Style::marker([1.0, 0.0, 0.0, 1.0], 0.5));
        let data = tessellate_shape(&shape);
        assert_eq!(data.markers.len(), 1);
        assert_eq!(data.markers[0].center, [3.0, 4.0]);
        assert_eq!(data.markers[0].half_extent, [0.5, 0.5]);
        assert!(data.strokes.is_empty() && data.fill_vertices.is_empty());
    }

    #[test]
    fn multipoint_makes_one_marker_each() {
        let mp = MultiPoint::new(vec![Point::new(0.0f32, 0.0), Point::new(1.0, 1.0)]);
        let data = tessellate_shape(&Shape::new(mp, Style::marker([1.0; 4], 1.0)));
        assert_eq!(data.markers.len(), 2);
    }

    #[test]
    fn linestring_stroke_makes_n_minus_1_segments() {
        let ls = LineString::from(vec![(0.0f32, 0.0), (1.0, 0.0), (1.0, 1.0)]);
        let data = tessellate_shape(&Shape::new(ls, Style::stroke([1.0; 4], 0.1)));
        assert_eq!(data.strokes.len(), 2);
    }

    #[test]
    fn polygon_fill_and_stroke() {
        let ext = LineString::from(vec![
            (0.0f32, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]);
        let poly = Polygon::new(ext, vec![]);
        let style = Style::fill([0.0, 0.0, 1.0, 1.0]).with_stroke([1.0; 4], 0.1);
        let data = tessellate_shape(&Shape::new(poly, style));
        assert_eq!(data.fill_vertices.len(), 4, "closing point dropped for fill");
        assert_eq!(data.fill_indices.len(), 6);
        // Closed ring (5 coords) → 4 outline segments.
        assert_eq!(data.strokes.len(), 4);
    }

    #[test]
    fn texture_emits_one_quad_over_bounding_rect() {
        use crate::texture::TextureImage;
        let tex = TextureImage::new(1, 1, vec![255, 0, 0, 255])
            .unwrap()
            .into_handle();
        // A 4-wide, 2-tall rect from (0,0) to (4,2).
        let rect = geo::Rect::new(Coord { x: 0.0f32, y: 0.0 }, Coord { x: 4.0, y: 2.0 });
        let data = tessellate_shape(&Shape::textured(rect, tex));
        assert_eq!(data.textured_quads.len(), 1);
        let q = &data.textured_quads[0];
        assert_eq!(q.center, [2.0, 1.0]);
        assert_eq!(q.half_extent, [2.0, 1.0]);
        // Default style paints nothing else.
        assert!(data.markers.is_empty() && data.strokes.is_empty());
        assert!(data.fill_vertices.is_empty());
    }

    #[test]
    fn fill_only_polygon_has_no_strokes() {
        let poly = Polygon::new(
            LineString::from(vec![
                Coord { x: 0.0f32, y: 0.0 },
                Coord { x: 1.0, y: 0.0 },
                Coord { x: 0.0, y: 1.0 },
            ]),
            vec![],
        );
        let data = tessellate_shape(&Shape::new(poly, Style::fill([1.0; 4])));
        assert!(data.strokes.is_empty());
        assert_eq!(data.fill_indices.len(), 3);
    }
}
