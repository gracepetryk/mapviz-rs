//! Renderer-agnostic core for mapviz.
//!
//! This crate owns the parts of mapviz that have nothing to do with any
//! particular rendering backend: coordinate systems, projections, the scene
//! graph, cameras, and the scene clock. It must not depend on `wgpu` or any
//! other backend's types — see `CLAUDE.md` for the rationale.
//!
//! The rendering contract lives here too, as the `Backend` trait (with a 2D
//! surface every backend must implement and an optional 3D capability surface),
//! but the implementations live in backend crates such as `mapviz-render`.

pub mod camera;
pub mod coords;
pub mod error;
pub mod frame;
pub mod geometry;
pub mod layer;
pub mod primitive;
pub mod scene;
pub mod tessellate;

pub use camera::Camera2d;
pub use error::{Error, Result};
pub use frame::Frame;
pub use geometry::{Shape, Style};
pub use layer::Layer;
pub use primitive::{FillVertex, LineInstance, QuadInstance};
pub use scene::Scene;
pub use tessellate::{DrawData, tessellate, tessellate_shape};

/// Re-export of the [`geo`] crate so downstream crates can build geometry
/// without depending on `geo` directly.
pub use geo;

// Planned modules, added as each area is implemented (see CLAUDE.md):
//   pub mod projection;  // Web Mercator, equirectangular, 3D globe; pluggable trait
//   pub mod backend;     // `Backend` trait + `Capabilities` query (2D mandatory, 3D opt-in)
//   pub mod time;        // scene `Clock`, play/pause/scrub
//   pub mod trajectory;  // `Trajectory<T>`: timestamped samples + interpolator
