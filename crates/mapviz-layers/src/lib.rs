//! Built-in layer implementations for mapviz.
//!
//! Each layer implements the core [`Layer`](mapviz_core::Layer) trait and emits
//! [`Shape`]s (styled `geo` geometry) into the frame — layers never touch a
//! rendering backend directly. Users add their own layers by implementing the
//! same trait.

use mapviz_core::{Frame, Layer, Shape};

/// A layer holding a fixed list of [`Shape`]s.
///
/// This is the general-purpose layer: build any `geo` geometry, pair it with a
/// [`Style`](mapviz_core::Style), and the layer copies it into the frame each
/// time it's prepared. Group many like primitives into one `Multi*` geometry to
/// have them drawn in a single batched pass.
#[derive(Default)]
pub struct ShapeLayer {
    shapes: Vec<Shape>,
}

impl ShapeLayer {
    /// A layer from an explicit list of shapes.
    pub fn new(shapes: Vec<Shape>) -> Self {
        Self { shapes }
    }

    /// Append a shape to the layer.
    pub fn push(&mut self, shape: Shape) {
        self.shapes.push(shape);
    }

    /// The shapes this layer will emit.
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }
}

impl Layer for ShapeLayer {
    fn prepare(&mut self, frame: &mut Frame) {
        frame.shapes.extend(self.shapes.iter().cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mapviz_core::geo::{Point, Polygon};
    use mapviz_core::{Style, geo::LineString};

    #[test]
    fn prepare_extends_frame_in_order() {
        let mut layer = ShapeLayer::new(vec![
            Shape::new(Point::new(0.0f32, 0.0), Style::marker([1.0; 4], 1.0)),
            Shape::new(Point::new(1.0f32, 1.0), Style::marker([1.0; 4], 1.0)),
        ]);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        assert_eq!(frame.shapes.len(), 2);
    }

    #[test]
    fn push_appends() {
        let mut layer = ShapeLayer::default();
        assert_eq!(layer.shapes().len(), 0);
        let poly = Polygon::new(
            LineString::from(vec![(0.0f32, 0.0), (1.0, 0.0), (0.0, 1.0)]),
            vec![],
        );
        layer.push(Shape::new(poly, Style::fill([1.0; 4])));
        assert_eq!(layer.shapes().len(), 1);
    }

    #[test]
    fn prepare_preserves_layer_contents_across_calls() {
        let mut layer = ShapeLayer::new(vec![Shape::new(
            Point::new(0.0f32, 0.0),
            Style::marker([1.0; 4], 1.0),
        )]);
        let mut frame = Frame::new();
        layer.prepare(&mut frame);
        layer.prepare(&mut frame);
        assert_eq!(frame.shapes.len(), 2, "re-preparing re-emits the shapes");
    }
}
