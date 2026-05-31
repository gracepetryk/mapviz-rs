//! Built-in layer implementations for mapviz.
//!
//! Each layer implements the core `Layer` trait and renders through the
//! backend-agnostic rendering contract — layers query the backend's
//! capabilities and degrade gracefully (substitute or skip) when an optional
//! 3D primitive is unavailable. Users add their own layers by implementing the
//! same trait.
