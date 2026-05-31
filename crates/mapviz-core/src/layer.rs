//! The `Layer` trait.
//!
//! A layer contributes primitives to a frame. Built-in layers live in
//! `mapviz-layers`; users implement this trait for custom rendering. Layers
//! emit backend-agnostic primitive data into the [`Frame`] rather than issuing
//! draw calls, so they never depend on any rendering backend.

use crate::frame::Frame;

/// A source of drawable primitives in a scene.
///
/// `prepare` is called once per frame (for now); the layer pushes its
/// primitives into `frame`. Later this gains a context argument (camera, clock,
/// viewport) so layers can do level-of-detail and time-based work.
pub trait Layer {
    /// Emit this layer's primitives into the frame's draw list.
    fn prepare(&mut self, frame: &mut Frame);
}
