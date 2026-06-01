//! Error types for `mapviz-core`.
//!
//! Errors are typed and never panic across the FFI boundary — library paths
//! reachable from the wasm crate return `Result` rather than unwrapping.

use thiserror::Error;

/// The crate-wide result type.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Errors produced by the core geometry, projection, scene, and time APIs.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum Error {
    /// A coordinate or parameter was outside its valid range
    /// (e.g. a latitude beyond ±90°).
    #[error("value out of range: {0}")]
    OutOfRange(String),
    /// Polygon tessellation failed (degenerate geometry or too few points).
    #[error("tessellation error: {0}")]
    Tessellation(String),
    /// A texture image's pixel buffer did not match its declared dimensions.
    #[error("texture error: {0}")]
    Texture(String),
    /// Decoding encoded data (e.g. a PNG/JPEG raster tile) failed.
    #[error("decode error: {0}")]
    Decode(String),
}
