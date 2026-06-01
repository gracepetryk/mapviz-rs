//! Backend-agnostic texture image data.
//!
//! A [`TextureImage`] is just decoded pixels — dimensions plus a tightly packed
//! `width * height * 4` buffer of 8-bit RGBA, top row first. It carries no GPU
//! types; a backend (e.g. `mapviz-render`) uploads it to a GPU texture when a
//! [`Shape`](crate::Shape) referencing it is drawn. Map tiles are the motivating
//! case: a tile is a fetched-and-decoded image painted onto a rectangle.
//!
//! Images are shared by reference ([`TextureHandle`] = `Arc<TextureImage>`) so a
//! shape can be cloned into a frame each tick without copying pixels, and so a
//! backend can cache the uploaded GPU texture by the handle's identity.

use std::sync::Arc;

use crate::error::{Error, Result};

/// Decoded RGBA8 image pixels: `width * height * 4` bytes, top row first, each
/// channel in `[0, 255]` and *not* premultiplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tightly packed RGBA bytes, length `width * height * 4`.
    pub rgba: Vec<u8>,
}

impl TextureImage {
    /// Build an image from raw RGBA bytes, validating that the buffer length
    /// matches `width * height * 4`.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| Error::Texture("image dimensions overflow".into()))?;
        if rgba.len() != expected {
            return Err(Error::Texture(format!(
                "pixel buffer is {} bytes, expected {expected} for {width}x{height} RGBA",
                rgba.len()
            )));
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Wrap this image in a shareable [`TextureHandle`].
    pub fn into_handle(self) -> TextureHandle {
        Arc::new(self)
    }
}

/// A shared, reference-counted [`TextureImage`]. Cloning is cheap (a refcount
/// bump), and a backend can key its uploaded-texture cache on the handle's
/// pointer identity.
pub type TextureHandle = Arc<TextureImage>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_buffer_length() {
        // 2x2 RGBA needs 16 bytes.
        assert!(TextureImage::new(2, 2, vec![0u8; 16]).is_ok());
        assert!(TextureImage::new(2, 2, vec![0u8; 15]).is_err());
        assert!(TextureImage::new(2, 2, vec![0u8; 17]).is_err());
    }

    #[test]
    fn zero_sized_needs_empty_buffer() {
        assert!(TextureImage::new(0, 0, vec![]).is_ok());
        assert!(TextureImage::new(0, 0, vec![1]).is_err());
    }
}
