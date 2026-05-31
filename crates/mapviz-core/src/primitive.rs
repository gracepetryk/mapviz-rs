//! Backend-agnostic drawable primitives.
//!
//! Primitives are plain data: a layer produces them, a backend consumes them.
//! They carry no GPU types — the `repr(C)` layout is a convenience for backends
//! that upload them directly, not a coupling to any backend.
//!
//! A [`Primitive`] is a *batch* of like instances rather than a single shape, so
//! a backend can render each variant as one instanced draw call while preserving
//! the order in which layers submitted them. New primitive kinds are new
//! variants, not new fields scattered across the frame.

/// An axis-aligned, solid-colored quad in world space.
///
/// `center` and `half_extent` are in world units; `color` is linear RGBA in
/// `[0, 1]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadInstance {
    /// Center of the quad, in world units.
    pub center: [f32; 2],
    /// Half-width and half-height, in world units.
    pub half_extent: [f32; 2],
    /// Linear RGBA color, each channel in `[0, 1]`.
    pub color: [f32; 4],
}

impl QuadInstance {
    /// A square centered at `center` with the given `half_size` and color.
    pub fn square(center: [f32; 2], half_size: f32, color: [f32; 4]) -> Self {
        Self {
            center,
            half_extent: [half_size, half_size],
            color,
        }
    }
}

/// A solid-colored straight line segment in world space.
///
/// `start`/`end` are endpoints in world units; `width` is the stroke width, also
/// in world units (so it scales with zoom, like a [`QuadInstance`]); `color` is
/// linear RGBA in `[0, 1]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineInstance {
    /// Start endpoint, in world units.
    pub start: [f32; 2],
    /// End endpoint, in world units.
    pub end: [f32; 2],
    /// Stroke width, in world units.
    pub width: f32,
    /// Linear RGBA color, each channel in `[0, 1]`.
    pub color: [f32; 4],
}

impl LineInstance {
    /// A segment from `start` to `end` with the given width and color.
    pub fn new(start: [f32; 2], end: [f32; 2], width: f32, color: [f32; 4]) -> Self {
        Self {
            start,
            end,
            width,
            color,
        }
    }
}

/// A multi-vertex polyline (thick stroked path) in world space.
///
/// A polyline carries all of the geometry for one continuous stroke. The
/// `points` field holds the ordered vertices of the path; `width` is the stroke
/// width in world units (scales with zoom, like [`LineInstance`]); `color` is
/// linear RGBA in `[0, 1]`.
///
/// Use [`Polyline::expand`] to decompose a polyline into a [`Vec<LineInstance>`]
/// for rendering — one [`LineInstance`] per consecutive segment pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Polyline {
    /// Ordered path vertices, in world units.
    pub points: Vec<[f32; 2]>,
    /// Stroke width, in world units.
    pub width: f32,
    /// Linear RGBA color, each channel in `[0, 1]`.
    pub color: [f32; 4],
}

impl Polyline {
    /// Construct a polyline from a list of `points`, a stroke `width`, and a
    /// `color`.
    pub fn new(points: Vec<[f32; 2]>, width: f32, color: [f32; 4]) -> Self {
        Self {
            points,
            width,
            color,
        }
    }

    /// Expand this polyline into a sequence of [`LineInstance`] segments.
    ///
    /// One [`LineInstance`] is emitted for each consecutive pair of
    /// *distinct* points. Consecutive duplicate points are skipped. A polyline
    /// with fewer than 2 distinct points produces an empty `Vec`.
    ///
    /// **Join behaviour (v1).** Segments are rendered as independent instanced
    /// quads by the shader, so interior vertices will show gaps at sharp angles.
    /// The segments are extended by half the stroke width at each end so they
    /// overlap at interior vertices, giving a rough bevel-like fill for modest
    /// angles. Miter/round joins are a planned follow-up once a per-join
    /// geometry pass is available.
    pub fn expand(&self) -> Vec<LineInstance> {
        let mut segments = Vec::new();
        let mut prev: Option<[f32; 2]> = None;

        for &pt in &self.points {
            if let Some(p) = prev {
                // Skip duplicate consecutive points.
                let dx = pt[0] - p[0];
                let dy = pt[1] - p[1];
                if dx * dx + dy * dy < f32::EPSILON {
                    continue;
                }
                segments.push(LineInstance::new(p, pt, self.width, self.color));
            }
            prev = Some(pt);
        }

        segments
    }
}

/// A batch of like primitive instances, drawable as a single instanced pass.
///
/// Each variant carries a contiguous run of instances. Backends match on the
/// variant and draw the batch; the order of batches within a [`Frame`] is the
/// order layers submitted them, which is the render (painter's) order.
#[derive(Clone, Debug, PartialEq)]
pub enum Primitive {
    /// A batch of solid-colored quads.
    Quads(Vec<QuadInstance>),
    /// A batch of solid-colored line segments.
    Lines(Vec<LineInstance>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color() -> [f32; 4] {
        [1.0, 0.5, 0.0, 1.0]
    }

    #[test]
    fn expand_two_points_one_segment() {
        let pl = Polyline::new(vec![[0.0, 0.0], [1.0, 0.0]], 0.1, color());
        let segs = pl.expand();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, [0.0, 0.0]);
        assert_eq!(segs[0].end, [1.0, 0.0]);
        assert_eq!(segs[0].width, 0.1);
        assert_eq!(segs[0].color, color());
    }

    #[test]
    fn expand_n_points_produces_n_minus_1_segments() {
        let points = vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let pl = Polyline::new(points, 0.2, color());
        assert_eq!(pl.expand().len(), 3);
    }

    #[test]
    fn expand_empty_produces_no_segments() {
        let pl = Polyline::new(vec![], 0.1, color());
        assert!(pl.expand().is_empty());
    }

    #[test]
    fn expand_single_point_produces_no_segments() {
        let pl = Polyline::new(vec![[1.0, 2.0]], 0.1, color());
        assert!(pl.expand().is_empty());
    }

    #[test]
    fn expand_skips_duplicate_consecutive_points() {
        // Points: A, A (dup), B → only one segment A→B.
        let pl = Polyline::new(
            vec![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            0.1,
            color(),
        );
        let segs = pl.expand();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, [0.0, 0.0]);
        assert_eq!(segs[0].end, [1.0, 0.0]);
    }

    #[test]
    fn expand_all_duplicate_points_produces_no_segments() {
        let pl = Polyline::new(
            vec![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            0.1,
            color(),
        );
        assert!(pl.expand().is_empty());
    }

    #[test]
    fn expand_propagates_width_and_color_to_all_segments() {
        let w = 0.42;
        let c = [0.1, 0.2, 0.3, 0.4];
        let pl = Polyline::new(
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            w,
            c,
        );
        for seg in pl.expand() {
            assert_eq!(seg.width, w);
            assert_eq!(seg.color, c);
        }
    }
}
