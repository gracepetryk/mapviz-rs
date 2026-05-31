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

/// A batch of like primitive instances, drawable as a single instanced pass.
///
/// Each variant carries a contiguous run of instances. Backends match on the
/// variant and draw the batch; the order of batches within a [`Frame`] is the
/// order layers submitted them, which is the render (painter's) order.
///
/// The `Mesh` variant uses an indexed draw rather than instancing — it is a
/// single triangle-list mesh — but it participates in the same painter's order.
#[derive(Clone, Debug, PartialEq)]
pub enum Primitive {
    /// A batch of solid-colored quads.
    Quads(Vec<QuadInstance>),
    /// A batch of solid-colored line segments.
    Lines(Vec<LineInstance>),
    /// A filled triangle mesh (indexed draw, triangle-list topology).
    ///
    /// `vertices` are the unique vertex positions/colors; `indices` are `u32`
    /// indices into `vertices`, grouped in triples (one triangle each).
    Mesh {
        /// Vertex data for the mesh.
        vertices: Vec<FillVertex>,
        /// Triangle indices into `vertices`, three per triangle.
        indices: Vec<u32>,
    },
}
