//! The `Map` class exposed to JavaScript.

use glam::Vec2;
use mapviz_core::{Camera2d, Scene};
use mapviz_layers::QuadLayer;
use mapviz_render::Renderer;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// An interactive 2D map rendered into a canvas.
///
/// Construct it with [`Map::create`] (async), then drive it from JS: call
/// [`Map::render`] each animation frame and feed input through [`Map::pan`] /
/// [`Map::zoom_at`] / [`Map::resize`].
#[wasm_bindgen]
pub struct Map {
    renderer: Renderer,
    scene: Scene,
    camera: Camera2d,
}

#[wasm_bindgen]
impl Map {
    /// Create a map that renders into `canvas`. The canvas's `width`/`height`
    /// attributes must already be set to the desired physical pixel size.
    ///
    /// Async because GPU adapter/device acquisition is async.
    pub async fn create(canvas: HtmlCanvasElement) -> Result<Map, JsError> {
        console_error_panic_hook::set_once();

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let renderer = Renderer::new(canvas, width, height)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        // A placeholder grid until real data sources land.
        let cols = 20;
        let rows = 20;
        let spacing = 2.0;
        let square = 1.5;
        let mut scene = Scene::new();
        scene.add_layer(Box::new(QuadLayer::grid(cols, rows, spacing, square)));

        let mut camera = Camera2d::new(Vec2::new(width as f32, height as f32));
        // Fit the grid comfortably within the smaller viewport dimension.
        let grid_extent = (cols.max(rows) - 1) as f32 * spacing + square;
        camera.set_scale(width.min(height) as f32 / (grid_extent * 1.1));

        Ok(Map {
            renderer,
            scene,
            camera,
        })
    }

    /// Render one frame.
    pub fn render(&mut self) -> Result<(), JsError> {
        let frame = self.scene.build_frame();
        self.renderer
            .render(&self.camera, &frame)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Resize the drawing surface to `width` × `height` physical pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
        self.camera
            .set_viewport(Vec2::new(width.max(1) as f32, height.max(1) as f32));
    }

    /// Pan by a screen-space delta in physical pixels (e.g. a pointer drag).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan_pixels(dx, dy);
    }

    /// Multiply the zoom by `factor`, keeping the world point under the cursor
    /// (`x`, `y` in physical pixels from the top-left) fixed on screen.
    pub fn zoom_at(&mut self, factor: f32, x: f32, y: f32) {
        self.camera.zoom_at(factor, Vec2::new(x, y));
    }
}
