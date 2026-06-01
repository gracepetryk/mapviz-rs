//! The `Map` class exposed to JavaScript.

use glam::Vec2;
use mapviz_core::geo::{Coord, Rect};
use mapviz_core::{Camera2d, Scene, Shape};
use mapviz_layers::ShapeLayer;
use mapviz_render::Renderer;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// An interactive 2D map rendered into a canvas.
///
/// Construct it with [`Map::create`] (async), then drive it from JS: call
/// [`Map::render`] each animation frame and feed input through [`Map::pan`] /
/// [`Map::zoom_at`] / [`Map::resize`]. Populate the scene with content such as
/// [`Map::show_tile`].
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
    /// The scene starts empty; add content with methods like
    /// [`Map::show_tile`]. Async because GPU adapter/device acquisition is async.
    pub async fn create(canvas: HtmlCanvasElement) -> Result<Map, JsError> {
        console_error_panic_hook::set_once();

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let renderer = Renderer::new(canvas, width, height)
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;

        let scene = Scene::new();
        let camera = Camera2d::new(Vec2::new(width as f32, height as f32));

        Ok(Map {
            renderer,
            scene,
            camera,
        })
    }

    /// Replace the scene with a single rectangle textured by a PNG image.
    ///
    /// `png` is the raw bytes of a PNG file (e.g. a fetched map tile); decoding
    /// to pixels happens here in wasm, so callers never convert formats
    /// themselves. The rectangle is centered at the world origin, two world
    /// units tall, with its width matching the image's aspect ratio, and the
    /// camera is fit to show it.
    pub fn show_tile(&mut self, png: &[u8]) -> Result<(), JsError> {
        let texture = mapviz_tiles::decode_png(png)
            .map_err(|e| JsError::new(&e.to_string()))?
            .into_handle();
        let width = texture.width;
        let height = texture.height;

        // World rectangle: 2 units tall, width scaled by the image aspect ratio.
        let aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };
        let half_h = 1.0;
        let half_w = half_h * aspect;
        let rect = Rect::new(
            Coord {
                x: -half_w,
                y: -half_h,
            },
            Coord {
                x: half_w,
                y: half_h,
            },
        );

        let mut layer = ShapeLayer::default();
        layer.push(Shape::textured(rect, texture));

        let mut scene = Scene::new();
        scene.add_layer(Box::new(layer));
        self.scene = scene;

        // Fit the rectangle within the viewport with a little margin.
        let viewport = self.camera.viewport();
        let scale = (viewport.min_element() / (half_h.max(half_w) * 2.0 * 1.1)).max(f32::MIN_POSITIVE);
        self.camera.set_center(Vec2::ZERO);
        self.camera.set_scale(scale);
        Ok(())
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
