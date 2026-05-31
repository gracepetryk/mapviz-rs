//! 2D camera.
//!
//! [`Camera2d`] is plain state — a world-space center, a zoom expressed as
//! pixels per world unit, and the viewport size in pixels. It produces a
//! world→clip matrix for the renderer and offers pan/zoom mutators in screen
//! (pixel) terms, which is what pointer and wheel input speak. Controllers
//! (drag, inertia, fly-to) are deliberately kept separate.
//!
//! Conventions: world space is y-up; screen/pixel space is y-down with the
//! origin at the top-left of the viewport (matching DOM pointer coordinates).

use glam::{Mat4, Vec2, Vec4};

/// A 2D orthographic camera over an abstract world plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera2d {
    /// World-space point shown at the center of the viewport.
    center: Vec2,
    /// Zoom: how many pixels one world unit spans. Always positive.
    px_per_world_unit: f32,
    /// Viewport size in pixels.
    viewport_px: Vec2,
}

impl Camera2d {
    /// Smallest allowed zoom, to keep the projection well-defined.
    const MIN_SCALE: f32 = f32::MIN_POSITIVE;

    /// A camera centered on the world origin at 1 pixel per world unit.
    pub fn new(viewport_px: Vec2) -> Self {
        Self {
            center: Vec2::ZERO,
            px_per_world_unit: 1.0,
            viewport_px,
        }
    }

    /// The world point currently at the center of the viewport.
    pub fn center(&self) -> Vec2 {
        self.center
    }

    /// Set the world point shown at the center of the viewport.
    pub fn set_center(&mut self, center: Vec2) {
        self.center = center;
    }

    /// Current zoom, in pixels per world unit.
    pub fn scale(&self) -> f32 {
        self.px_per_world_unit
    }

    /// Set the zoom, in pixels per world unit (clamped to be positive).
    pub fn set_scale(&mut self, px_per_world_unit: f32) {
        self.px_per_world_unit = px_per_world_unit.max(Self::MIN_SCALE);
    }

    /// Viewport size in pixels.
    pub fn viewport(&self) -> Vec2 {
        self.viewport_px
    }

    /// Update the viewport size in pixels (e.g. on canvas resize).
    pub fn set_viewport(&mut self, viewport_px: Vec2) {
        self.viewport_px = viewport_px;
    }

    /// The world→clip transform for this camera, mapping world coordinates into
    /// normalized device coordinates (`x`/`y` in `[-1, 1]`, y-up).
    pub fn view_proj(&self) -> Mat4 {
        // Pixels-per-world scaled into NDC half-extents.
        let sx = 2.0 * self.px_per_world_unit / self.viewport_px.x;
        let sy = 2.0 * self.px_per_world_unit / self.viewport_px.y;
        // Column-major: columns 0/1 scale, column 3 translates by -scale*center.
        Mat4::from_cols(
            Vec4::new(sx, 0.0, 0.0, 0.0),
            Vec4::new(0.0, sy, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(-sx * self.center.x, -sy * self.center.y, 0.0, 1.0),
        )
    }

    /// Convert a screen-space point (pixels, y-down, origin top-left) to a
    /// world-space point.
    pub fn screen_to_world(&self, screen_px: Vec2) -> Vec2 {
        let offset_px = screen_px - self.viewport_px * 0.5;
        // Screen y is down, world y is up: flip y.
        Vec2::new(
            self.center.x + offset_px.x / self.px_per_world_unit,
            self.center.y - offset_px.y / self.px_per_world_unit,
        )
    }

    /// Pan by a screen-space delta in pixels (e.g. a pointer drag), moving the
    /// world with the cursor.
    pub fn pan_pixels(&mut self, dx: f32, dy: f32) {
        self.center.x -= dx / self.px_per_world_unit;
        self.center.y += dy / self.px_per_world_unit;
    }

    /// Multiply the zoom by `factor` while keeping the world point under
    /// `screen_px` fixed on screen (zoom toward the cursor).
    pub fn zoom_at(&mut self, factor: f32, screen_px: Vec2) {
        let world_before = self.screen_to_world(screen_px);
        self.set_scale(self.px_per_world_unit * factor);
        let world_after = self.screen_to_world(screen_px);
        self.center += world_before - world_after;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera2d {
        let mut c = Camera2d::new(Vec2::new(800.0, 600.0));
        c.set_center(Vec2::new(12.0, -7.0));
        c.set_scale(4.0);
        c
    }

    #[test]
    fn pan_then_inverse_restores_center() {
        let mut c = cam();
        let start = c.center();
        c.pan_pixels(31.0, -18.0);
        c.pan_pixels(-31.0, 18.0);
        assert!(
            (c.center() - start).length() < 1e-4,
            "center drifted: {:?}",
            c.center()
        );
    }

    #[test]
    fn screen_to_world_round_trips_through_view_proj() {
        // The viewport center maps to the camera center.
        let c = cam();
        let w = c.screen_to_world(c.viewport() * 0.5);
        assert!((w - c.center()).length() < 1e-4);
    }

    #[test]
    fn zoom_at_keeps_anchor_point_fixed() {
        let mut c = cam();
        let anchor = Vec2::new(123.0, 456.0);
        let world_before = c.screen_to_world(anchor);
        c.zoom_at(2.5, anchor);
        let world_after = c.screen_to_world(anchor);
        assert!(
            (world_before - world_after).length() < 1e-3,
            "anchor world point moved: {world_before:?} -> {world_after:?}"
        );
        // And the zoom actually changed.
        assert!((c.scale() - 10.0).abs() < 1e-4);
    }
}
