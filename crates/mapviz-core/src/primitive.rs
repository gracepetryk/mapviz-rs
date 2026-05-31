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

/// A solid-colored filled circle (disc) in world space.
///
/// `center` is the disc center in world units; `radius` is the disc radius,
/// also in world units (scales with zoom like a [`QuadInstance`]); `color` is
/// linear RGBA in `[0, 1]`. This is the natural primitive for MVT POINT
/// geometry — one `CircleInstance` per point feature.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircleInstance {
    /// Center of the disc, in world units.
    pub center: [f32; 2],
    /// Radius of the disc, in world units.
    pub radius: f32,
    /// Linear RGBA color, each channel in `[0, 1]`.
    pub color: [f32; 4],
}

impl CircleInstance {
    /// A disc centered at `center` with the given `radius` and `color`.
    pub fn new(center: [f32; 2], radius: f32, color: [f32; 4]) -> Self {
        Self {
            center,
            radius,
            color,
        }
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
    /// A batch of solid-colored filled circles (discs).
    Circles(Vec<CircleInstance>),
}
