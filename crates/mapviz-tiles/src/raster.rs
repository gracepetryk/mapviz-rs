//! Raster tile decoding into core [`TextureImage`]s.
//!
//! Gated behind the `png` feature. Decoding is pure-Rust (the `image` crate's
//! png codec) so it runs in the wasm binary — clients pass encoded bytes, not
//! pre-decoded pixels.

use image::ImageFormat;
use mapviz_core::{Error, Result, TextureImage};

/// Decode PNG bytes into an RGBA8 [`TextureImage`].
///
/// Any PNG color type (palette, grayscale, RGB, …) is expanded to straight
/// (non-premultiplied) RGBA8, top row first — exactly the layout a textured
/// [`Shape`](mapviz_core::Shape) expects.
pub fn decode_png(bytes: &[u8]) -> Result<TextureImage> {
    let image = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .map_err(|e| Error::Decode(format!("png decode failed: {e}")))?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    TextureImage::new(width, height, rgba.into_raw())
}
