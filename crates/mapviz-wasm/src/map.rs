//! The `Map` class exposed to JavaScript.

use glam::Vec2;
use mapviz_core::geo::{LineString, MultiLineString, Point, Polygon};
use mapviz_core::{Camera2d, Scene, Shape, Style};
use mapviz_layers::ShapeLayer;
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

        // A placeholder scene until real data sources land, expressed entirely
        // in `geo` geometry: a gradient grid of point markers, a filled polygon
        // with a hole, the axes, and a border (later layers draw on top, which
        // also exercises cross-shape draw order).
        let cols: u32 = 20;
        let rows: u32 = 20;
        let spacing = 2.0;
        let square = 1.5;
        let grid_extent = (cols.max(rows) - 1) as f32 * spacing + square;
        let border = grid_extent * 0.5 + spacing;
        let line_w = spacing * 0.15;

        let mut scene = Scene::new();

        // Grid of gradient point markers (one shape per cell).
        let half = square * 0.5;
        let origin_x = -((cols - 1) as f32) * spacing * 0.5;
        let origin_y = -((rows - 1) as f32) * spacing * 0.5;
        let denom_x = (cols - 1).max(1) as f32;
        let denom_y = (rows - 1).max(1) as f32;
        let mut grid = ShapeLayer::default();
        for row in 0..rows {
            for col in 0..cols {
                let cx = origin_x + col as f32 * spacing;
                let cy = origin_y + row as f32 * spacing;
                let color = [col as f32 / denom_x, row as f32 / denom_y, 0.5, 1.0];
                grid.push(Shape::new(Point::new(cx, cy), Style::marker(color, half)));
            }
        }
        scene.add_layer(Box::new(grid));

        // A filled magenta polygon with a square hole, to exercise the fill path.
        let fc = border * 0.45;
        let s = border * 0.18;
        let h = s * 0.45;
        let exterior = LineString::from(vec![
            (fc - s, fc - s),
            (fc + s, fc - s),
            (fc + s, fc + s),
            (fc - s, fc + s),
            (fc - s, fc - s),
        ]);
        let hole = LineString::from(vec![
            (fc - h, fc - h),
            (fc - h, fc + h),
            (fc + h, fc + h),
            (fc + h, fc - h),
            (fc - h, fc - h),
        ]);
        let poly = Polygon::new(exterior, vec![hole]);
        scene.add_layer(Box::new(ShapeLayer::new(vec![Shape::new(
            poly,
            Style::fill([1.0, 0.2, 0.8, 0.85]).with_stroke([1.0, 1.0, 1.0, 0.9], line_w * 0.6),
        )])));

        // Axes through the origin (one shape, two segments).
        let axes = MultiLineString::new(vec![
            LineString::from(vec![(-border, 0.0), (border, 0.0)]),
            LineString::from(vec![(0.0, -border), (0.0, border)]),
        ]);
        scene.add_layer(Box::new(ShapeLayer::new(vec![Shape::new(
            axes,
            Style::stroke([1.0, 1.0, 1.0, 0.85], line_w),
        )])));

        // Border around the grid (a polygon outline, drawn on top).
        let border_ring = LineString::from(vec![
            (-border, -border),
            (border, -border),
            (border, border),
            (-border, border),
            (-border, -border),
        ]);
        scene.add_layer(Box::new(ShapeLayer::new(vec![Shape::new(
            Polygon::new(border_ring, vec![]),
            Style::stroke([0.25, 0.8, 1.0, 0.9], line_w),
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
