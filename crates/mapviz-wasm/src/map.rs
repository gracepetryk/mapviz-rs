//! The `Map` class exposed to JavaScript.

use glam::Vec2;
use mapviz_core::{Camera2d, LineInstance, Polyline, Scene};
use mapviz_layers::{LineLayer, PolylineLayer, QuadLayer};
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

        // A placeholder scene until real data sources land: a colored grid with
        // axes and a border drawn on top (later layers render over earlier ones,
        // which also exercises cross-primitive draw order).
        let cols = 20;
        let rows = 20;
        let spacing = 2.0;
        let square = 1.5;
        let grid_extent = (cols.max(rows) - 1) as f32 * spacing + square;
        let border = grid_extent * 0.5 + spacing;
        let line_w = spacing * 0.15;

        let mut scene = Scene::new();
        scene.add_layer(Box::new(QuadLayer::grid(cols, rows, spacing, square)));
        // Axes through the origin.
        scene.add_layer(Box::new(LineLayer::new(vec![
            LineInstance::new([-border, 0.0], [border, 0.0], line_w, [1.0, 1.0, 1.0, 0.85]),
            LineInstance::new([0.0, -border], [0.0, border], line_w, [1.0, 1.0, 1.0, 0.85]),
        ])));
        // Border around the grid.
        scene.add_layer(Box::new(LineLayer::rect(
            [-border, -border],
            [border, border],
            line_w,
            [0.25, 0.8, 1.0, 0.9],
        )));
        // A multi-vertex polyline zigzag across the top of the grid, exercising
        // the PolylineLayer / MVT LINESTRING path.
        let zz_y = border * 0.7;
        let zz_step = border * 0.4;
        let zz_amp = border * 0.25;
        scene.add_layer(Box::new(PolylineLayer::new(vec![Polyline::new(
            vec![
                [-border, zz_y],
                [-border + zz_step, zz_y + zz_amp],
                [-border + zz_step * 2.0, zz_y],
                [-border + zz_step * 3.0, zz_y + zz_amp],
                [-border + zz_step * 4.0, zz_y],
                [border, zz_y],
            ],
            line_w * 1.5,
            [1.0, 0.8, 0.1, 0.9],
        )])));

        let mut camera = Camera2d::new(Vec2::new(width as f32, height as f32));
        // Fit the bordered grid comfortably within the smaller viewport dimension.
        let view_extent = border * 2.0;
        camera.set_scale(width.min(height) as f32 / (view_extent * 1.1));

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
