//! Styled geometry — the unit of scene description.
//!
//! A scene is a list of [`Shape`]s: a `geo` geometry plus the [`Style`] it
//! should be drawn with. Geometry comes straight from the [`geo`] crate
//! (`Point`, `LineString`, `Polygon`, their `Multi*` forms, etc.), so callers
//! describe *what* is on the map in standard spatial types and leave *how* it
//! rasterizes to the renderer. Coordinates are `f32` world units.

pub use geo::Geometry;

use crate::texture::TextureHandle;

/// Linear RGBA, each channel in `[0, 1]`.
pub type Rgba = [f32; 4];

/// How a [`Shape`]'s geometry is painted.
///
/// A geometry can carry both a `fill` and a `stroke`; which ones apply depends
/// on the geometry kind:
/// - **Polygons** use `fill` (interior) and `stroke` (outline).
/// - **Lines / line strings** use `stroke`.
/// - **Points** use `fill` as the marker color, sized by `point_size`.
///
/// `None` for a channel means "don't draw that part".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// Interior color for polygons / marker color for points.
    pub fill: Option<Rgba>,
    /// Outline color for polygons / color for line geometry.
    pub stroke: Option<Rgba>,
    /// Stroke width, in world units (scales with zoom).
    pub stroke_width: f32,
    /// Marker half-extent for points, in world units.
    pub point_size: f32,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            fill: None,
            stroke: None,
            stroke_width: 1.0,
            point_size: 1.0,
        }
    }
}

impl Style {
    /// A filled style (polygon interior / point marker) with no outline.
    pub fn fill(color: Rgba) -> Self {
        Self {
            fill: Some(color),
            ..Self::default()
        }
    }

    /// A stroked style (line / polygon outline) with no fill.
    pub fn stroke(color: Rgba, width: f32) -> Self {
        Self {
            stroke: Some(color),
            stroke_width: width,
            ..Self::default()
        }
    }

    /// A point-marker style: a filled square of half-extent `size`.
    pub fn marker(color: Rgba, size: f32) -> Self {
        Self {
            fill: Some(color),
            point_size: size,
            ..Self::default()
        }
    }

    /// Set the fill color.
    pub fn with_fill(mut self, color: Rgba) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set the stroke color and width.
    pub fn with_stroke(mut self, color: Rgba, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Set the point marker half-extent.
    pub fn with_point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }
}

/// A `geo` geometry paired with the [`Style`] it should be drawn with.
///
/// This is the scene's unit of work: layers emit `Shape`s into a
/// [`Frame`](crate::Frame), and the renderer tessellates each one. Group many
/// like primitives into a single `Multi*` geometry (e.g. [`geo::MultiPoint`])
/// to have them drawn in one batched pass.
///
/// An optional [`texture`](Shape::texture) paints the geometry's bounding
/// rectangle with an image (e.g. a map tile) instead of, or in addition to, the
/// `Style`. When set, the texture draws on top of any fill and beneath any
/// stroke.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    /// The geometry, in `f32` world coordinates.
    pub geometry: Geometry<f32>,
    /// How to paint it.
    pub style: Style,
    /// An optional image painted over the geometry's bounding rectangle.
    pub texture: Option<TextureHandle>,
}

impl Shape {
    /// Pair any `geo` geometry with a style.
    pub fn new(geometry: impl Into<Geometry<f32>>, style: Style) -> Self {
        Self {
            geometry: geometry.into(),
            style,
            texture: None,
        }
    }

    /// Paint `geometry`'s bounding rectangle with `texture`. The geometry is
    /// typically a rectangle (e.g. a tile's world bounds); the image's UV `0..1`
    /// range maps across that bounding box. Uses a default (paint-nothing)
    /// `Style`, so only the texture shows; chain [`Shape::with_style`] to add a
    /// fill or stroke.
    pub fn textured(geometry: impl Into<Geometry<f32>>, texture: TextureHandle) -> Self {
        Self {
            geometry: geometry.into(),
            style: Style::default(),
            texture: Some(texture),
        }
    }

    /// Replace this shape's style.
    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}
