//! Low-level draw instances — the output of tessellating styled geometry.
//!
//! These are *not* authored directly. A scene is described with
//! [`Shape`](crate::Shape)s (`geo` geometry + [`Style`](crate::Style)); the
//! [`tessellate`](crate::tessellate) module turns each shape into these flat,
//! GPU-friendly instances, which a backend uploads and draws:
//!
//! - [`QuadInstance`] — a point marker.
//! - [`LineInstance`] — one segment of a stroked line / polygon outline.
//! - [`FillVertex`] — a vertex of a triangulated polygon fill.
//!
//! They carry no GPU types — the `repr(C)` layout is a convenience for backends
//! that upload them directly, not a coupling to any backend.

/// An axis-aligned, solid-colored quad in world space (a point marker).
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

/// A single vertex in a filled-polygon triangle mesh.
///
/// `position` is in world units; `color` is linear RGBA in `[0, 1]`. This is
/// plain `repr(C)` data — no GPU types. The render crate has a matching
/// `GpuFillVertex` with `bytemuck` derives.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillVertex {
    /// Position of the vertex, in world units.
    pub position: [f32; 2],
    /// Linear RGBA color, each channel in `[0, 1]`.
    pub color: [f32; 4],
}

impl FillVertex {
    /// Construct a vertex at `position` with the given `color`.
    pub fn new(position: [f32; 2], color: [f32; 4]) -> Self {
        Self { position, color }
    }
}
