//! GPU-free polygon tessellation.
//!
//! Converts a polygon (one exterior ring plus zero or more hole rings) into a
//! flat vertex/index list suitable for [`Primitive::Mesh`] indexed triangle-list
//! rendering.
//!
//! Ring convention follows **MVT 2.1**: exterior rings are wound
//! counter-clockwise; interior rings (holes) are wound clockwise. `earcutr`
//! does not require any particular winding — it accepts the raw coordinates and
//! hole-start indices — so this module simply feeds those through.

use crate::primitive::FillVertex;
use crate::error::{Error, Result};

/// Tessellate a polygon into a `(vertices, indices)` triangle mesh.
///
/// `exterior` is the outer ring; each element is a `[x, y]` world-space
/// coordinate. `holes` is a slice of interior rings, each wound opposite to the
/// exterior. `color` is applied uniformly to every vertex.
///
/// Returns an error if `exterior` has fewer than three points, or if `earcutr`
/// fails to produce a valid triangulation (degenerate geometry).
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

    let raw_indices = earcutr::earcut(&flat_coords, &hole_starts, 2).map_err(|e| {
        Error::Tessellation(format!("earcutr failed: {e:?}"))
    })?;

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

    /// A unit square: 4 corners → 2 triangles → 6 indices.
    #[test]
    fn square_produces_two_triangles() {
        let ring = vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (verts, indices) = tessellate(&ring, &[], [1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(verts.len(), 4, "four unique vertices");
        assert_eq!(indices.len(), 6, "two triangles = 6 indices");
        // All indices are in range.
        for &i in &indices {
            assert!((i as usize) < verts.len());
        }
        // All vertices got the right color.
        for v in &verts {
            assert_eq!(v.color, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    /// A square with a square hole: 4 outer + 4 inner vertices.
    /// earcutr should produce 8 triangles (24 indices) for the annular region.
    #[test]
    fn square_with_hole_has_expected_triangle_count() {
        // Exterior: counter-clockwise 2×2 square.
        let exterior = vec![[0.0f32, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        // Hole: clockwise 1×1 square centered at (1,1).
        let hole = vec![[0.5f32, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]];
        let (verts, indices) =
            tessellate(&exterior, &[hole], [0.0, 1.0, 0.0, 1.0]).unwrap();
        assert_eq!(verts.len(), 8, "4 outer + 4 inner vertices");
        // Triangulating an annular quadrilateral yields 8 triangles.
        assert_eq!(indices.len(), 24, "8 triangles = 24 indices");
        // All indices in range.
        for &i in &indices {
            assert!((i as usize) < verts.len(), "index {i} out of range");
        }
    }

    /// Too few points should return an error.
    #[test]
    fn too_few_points_returns_error() {
        let result = tessellate(&[[0.0, 0.0], [1.0, 0.0]], &[], [1.0; 4]);
        assert!(result.is_err());
    }
}
