//! The scene: an ordered stack of layers.

use crate::frame::Frame;
use crate::layer::Layer;

/// An ordered stack of layers. Lower indices render first (underneath).
#[derive(Default)]
pub struct Scene {
    layers: Vec<Box<dyn Layer>>,
}

impl Scene {
    /// An empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a layer on top of the existing stack.
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    /// Number of layers in the scene.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the scene has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Build this frame's draw list by asking every layer, in order, to emit
    /// its primitives into a fresh [`Frame`].
    pub fn build_frame(&mut self) -> Frame {
        let mut frame = Frame::new();
        for layer in &mut self.layers {
            layer.prepare(&mut frame);
        }
        frame
    }
}
