//! Provider-side image dimension guard.
//!
//! Anthropic rejects any image whose width or height exceeds
//! [`MAX_IMAGE_DIMENSION`] pixels with a fatal, non-retryable
//! `400 invalid_request_error: ... image dimensions exceed max allowed size:
//! 8000 pixels`. Because the full conversation history is re-sent every turn,
//! one oversized image baked into a stored `tool_result` (e.g. a full-page
//! browser screenshot staged by `read_image` or an MCP screenshot tool)
//! *wedges the whole session*: every subsequent request — including a fresh
//! user message — re-hits the same 400 until the poison image is removed.
//!
//! This module is the single source of truth for keeping images within the
//! cap. It is wired at two seams:
//!
//! * **Ingest** (`read_image`, `tools` crate): clamp when the image is first
//!   staged, so it never enters history oversized.
//! * **Wire lowering** (`convert_messages`): clamp again when stored history is
//!   lowered to the provider wire form. This is what *un-wedges an
//!   already-poisoned session* without history surgery, and it also covers the
//!   paste and MCP staging paths that do not pass through `read_image`.
//!
//! The dimension check is header-only: on the raw-bytes ingest path
//! ([`guard_image_bytes`]) via [`image::ImageReader::into_dimensions`], and on
//! the hot base64 wire path ([`guard_wire_image_base64`]) by decoding only a
//! bounded header-sized *prefix* of the payload — so a common in-cap image
//! never pays a full multi-MB base64 decode when the whole history is lowered
//! every turn. The expensive full decode + resize + re-encode runs *only* for
//! the rare oversized image (which must be decoded to be downscaled) or an
//! image whose header does not fit the probe window (a correctness fallback).
//! Rescaled output is always re-encoded as PNG (lossless, universally
//! accepted), so a marginal codec (e.g. WEBP encode) can never turn a
//! recoverable image into an unrecoverable one.

use std::io::Cursor;

use base64::Engine as _;

/// Maximum width or height (in pixels) Anthropic accepts for a single image.
/// A dimension strictly greater than this triggers the fatal 400; a dimension
/// equal to it is accepted, so the downscale target box is exactly this value.
pub const MAX_IMAGE_DIMENSION: u32 = 8000;

/// Header-probe budget in *base64 characters* for the wire fast path (4 chars
/// encode 3 bytes → ~132 KiB decoded). Sized to comfortably contain the
/// dimension fields of every supported format: PNG/GIF/WEBP put them in the
/// first ~40 bytes, while a JPEG's SOF marker can sit behind large APP1 (EXIF
/// thumbnail) / APP2 (ICC) segments — anything past this window falls back to a
/// full decode. Kept a whole number of 4-char base64 groups so a prefix slice
/// decodes cleanly (interior groups carry no padding).
const DIMENSION_PROBE_B64_CHARS: usize = 176 * 1024;

/// Outcome of guarding one image's dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageGuardOutcome {
    /// Within the cap, or the dimensions could not be read at all (so the image
    /// cannot be *proven* oversized). Send the original bytes unchanged — never
    /// destroy a payload we merely failed to inspect.
    Keep,
    /// Confirmed oversized and successfully downscaled to fit within
    /// [`MAX_IMAGE_DIMENSION`] on both axes. Always re-encoded as PNG.
    Rescaled {
        /// New media type — always `image/png` after re-encoding.
        media_type: String,
        /// Downscaled PNG bytes (not base64).
        bytes: Vec<u8>,
    },
    /// Header dimensions proved the image oversized, but the pixels could not be
    /// decoded/re-encoded (e.g. a truncated payload). The caller must drop the
    /// pixels — resending them would only re-trigger the fatal 400.
    DropOversized {
        /// The oversized dimensions read from the header, for the placeholder
        /// note the caller substitutes.
        width: u32,
        height: u32,
    },
}

/// Guard one image's raw (decoded) bytes against the dimension cap.
///
/// See the module docs for the header-only fast path and the PNG re-encode
/// policy. This never returns an error: an unreadable header degrades to
/// [`ImageGuardOutcome::Keep`] (we cannot prove it oversized), matching the
/// "never damage what we could not inspect" rule.
#[must_use]
pub fn guard_image_bytes(bytes: &[u8]) -> ImageGuardOutcome {
    let Some((width, height)) = read_dimensions(bytes) else {
        // Dimensions unreadable → cannot prove oversized → leave untouched.
        return ImageGuardOutcome::Keep;
    };
    if width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION {
        return ImageGuardOutcome::Keep;
    }
    // Confirmed oversized: pay the full decode + resize + PNG re-encode.
    match downscale_to_png(bytes) {
        Some(png) => ImageGuardOutcome::Rescaled {
            media_type: "image/png".to_string(),
            bytes: png,
        },
        None => ImageGuardOutcome::DropOversized { width, height },
    }
}

/// Guard a base64-encoded wire image.
///
/// The hot path is intentionally bounded: first decode only a fixed-size base64
/// prefix and read dimensions from that header slice. If the image is proven
/// in-cap, return [`WireImageOutcome::Keep`] without allocating/decoding the
/// full image payload. Full base64 decode is reserved for:
///
/// * confirmed-oversized images, which must be decoded to be downscaled; and
/// * rare headers whose dimension fields do not fit in the probe window, where
///   a full decode is the correctness fallback.
#[must_use]
pub fn guard_wire_image_base64(data_b64: &str) -> WireImageOutcome {
    if let Some((width, height)) = read_dimensions_from_base64_probe(data_b64) {
        if width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION {
            return WireImageOutcome::Keep;
        }
        return guard_oversized_wire_image(data_b64, width, height);
    }

    // Probe could not read dimensions (e.g. JPEG SOF after a very large EXIF/ICC
    // segment, or malformed/truncated base64 prefix). Fall back to the old full
    // decode path for correctness; an undecodable full payload stays untouched.
    let Some(bytes) = decode_full_base64(data_b64) else {
        return WireImageOutcome::Keep;
    };
    match guard_image_bytes(&bytes) {
        ImageGuardOutcome::Keep => WireImageOutcome::Keep,
        ImageGuardOutcome::Rescaled { media_type, bytes } => WireImageOutcome::Rescaled {
            media_type,
            data_b64: base64::engine::general_purpose::STANDARD.encode(&bytes),
        },
        ImageGuardOutcome::DropOversized { width, height } => WireImageOutcome::Drop { width, height },
    }
}

/// Decode only enough base64 to cover the image header and read dimensions from
/// that partial byte buffer. Returns `None` when the header is not readable from
/// the probe window; callers fall back to full decode in that case.
fn read_dimensions_from_base64_probe(data_b64: &str) -> Option<(u32, u32)> {
    let bytes = data_b64.as_bytes();
    let mut len = bytes.len().min(DIMENSION_PROBE_B64_CHARS);
    if bytes.len() > DIMENSION_PROBE_B64_CHARS {
        // Interior base64 has no padding, so decode a complete number of
        // 4-character groups. (When the whole payload is shorter than the probe,
        // keep its real length so normal tail padding is preserved.)
        len -= len % 4;
    }
    if len == 0 {
        return None;
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&bytes[..len])
        .ok()?;
    read_dimensions(&decoded)
}

fn decode_full_base64(data_b64: &str) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(data_b64.as_bytes())
        .ok()
}

fn guard_oversized_wire_image(data_b64: &str, width: u32, height: u32) -> WireImageOutcome {
    let Some(bytes) = decode_full_base64(data_b64) else {
        // The header prefix proved the payload would violate the provider's
        // dimension cap. If the tail is not valid base64, resending it cannot
        // succeed either; drop with a placeholder instead of re-wedging.
        return WireImageOutcome::Drop { width, height };
    };
    match downscale_to_png(&bytes) {
        Some(bytes) => WireImageOutcome::Rescaled {
            media_type: "image/png".to_string(),
            data_b64: base64::engine::general_purpose::STANDARD.encode(&bytes),
        },
        None => WireImageOutcome::Drop { width, height },
    }
}

/// Base64-level counterpart of [`ImageGuardOutcome`] for the wire seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireImageOutcome {
    /// Send the original base64 block unchanged.
    Keep,
    /// Replace with a downscaled PNG (base64), setting `media_type`.
    Rescaled {
        /// Always `image/png`.
        media_type: String,
        /// Downscaled PNG, base64-encoded.
        data_b64: String,
    },
    /// Drop the image; substitute a text placeholder built from these
    /// (oversized) dimensions.
    Drop {
        /// Oversized width read from the header.
        width: u32,
        /// Oversized height read from the header.
        height: u32,
    },
}

/// Human-readable placeholder for a dropped oversized image, so the model still
/// learns an image was present and why it is absent.
#[must_use]
pub fn oversized_placeholder(width: u32, height: u32) -> String {
    format!(
        "[image dropped: {width}x{height}px exceeds the {MAX_IMAGE_DIMENSION}px \
         per-dimension limit and could not be downscaled]"
    )
}

/// Read an image's dimensions from its header only (no full pixel decode).
/// Returns `None` when the format is unrecognized or the header is malformed.
fn read_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Fully decode, downscale to fit within [`MAX_IMAGE_DIMENSION`] on both axes
/// (aspect ratio preserved), and re-encode as PNG. `None` when the pixels
/// cannot be decoded or the PNG encode fails.
fn downscale_to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let image = image::load_from_memory(bytes).ok()?;
    // `resize` fits the image within the target box, preserving aspect ratio;
    // since at least one dimension exceeds the cap it only ever shrinks here.
    let resized = image.resize(
        MAX_IMAGE_DIMENSION,
        MAX_IMAGE_DIMENSION,
        image::imageops::FilterType::Triangle,
    );
    let mut out = Cursor::new(Vec::new());
    resized.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, RgbImage};

    /// Encode a solid-color `w`x`h` image to PNG bytes.
    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(RgbImage::new(w, h));
        let mut out = Cursor::new(Vec::new());
        image
            .write_to(&mut out, ImageFormat::Png)
            .expect("encode test PNG");
        out.into_inner()
    }

    #[test]
    fn within_cap_image_is_kept_untouched() {
        let bytes = png_bytes(100, 100);
        assert_eq!(guard_image_bytes(&bytes), ImageGuardOutcome::Keep);
    }

    #[test]
    fn exactly_at_cap_is_kept() {
        // 8000px is the max *allowed*; only strictly greater is rejected.
        let bytes = png_bytes(MAX_IMAGE_DIMENSION, 10);
        assert_eq!(guard_image_bytes(&bytes), ImageGuardOutcome::Keep);
    }

    #[test]
    fn oversized_image_is_downscaled_within_cap_preserving_aspect() {
        // 9000x300 → must fit within 8000 on the long axis; short axis scales
        // proportionally (3000/90 = 33.3 → 267).
        let bytes = png_bytes(9000, 300);
        let outcome = guard_image_bytes(&bytes);
        let ImageGuardOutcome::Rescaled { media_type, bytes } = outcome else {
            panic!("oversized image must be rescaled, got {outcome:?}");
        };
        assert_eq!(media_type, "image/png", "rescaled output is always PNG");
        let (w, h) = read_dimensions(&bytes).expect("rescaled PNG has readable dims");
        assert!(w <= MAX_IMAGE_DIMENSION && h <= MAX_IMAGE_DIMENSION, "fits cap: {w}x{h}");
        assert_eq!(w, MAX_IMAGE_DIMENSION, "long axis pinned to the cap");
        // Aspect ratio preserved: 9000/300 = 30 ≈ 8000/h.
        assert_eq!(h, 267, "short axis scaled proportionally");
    }

    #[test]
    fn oversized_tall_image_is_downscaled_on_the_height_axis() {
        // Mirrors the real symptom: a full-page browser screenshot far taller
        // than 8000px.
        let bytes = png_bytes(1200, 12000);
        let ImageGuardOutcome::Rescaled { bytes, .. } = guard_image_bytes(&bytes) else {
            panic!("tall screenshot must be rescaled");
        };
        let (w, h) = read_dimensions(&bytes).expect("dims");
        assert_eq!(h, MAX_IMAGE_DIMENSION, "height pinned to the cap");
        assert!(w <= MAX_IMAGE_DIMENSION);
    }

    #[test]
    fn undecodable_bytes_are_kept_not_destroyed() {
        // Cannot read dimensions → cannot prove oversized → keep.
        assert_eq!(guard_image_bytes(b"not an image at all"), ImageGuardOutcome::Keep);
    }

    #[test]
    fn wire_base64_roundtrips_rescale() {
        let bytes = png_bytes(10000, 500);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let WireImageOutcome::Rescaled { media_type, data_b64 } = guard_wire_image_base64(&b64)
        else {
            panic!("oversized wire image must rescale");
        };
        assert_eq!(media_type, "image/png");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(data_b64.as_bytes())
            .expect("valid base64 out");
        let (w, h) = read_dimensions(&decoded).expect("dims");
        assert!(w <= MAX_IMAGE_DIMENSION && h <= MAX_IMAGE_DIMENSION);
    }

    #[test]
    fn wire_base64_within_cap_is_kept() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes(64, 64));
        assert_eq!(guard_wire_image_base64(&b64), WireImageOutcome::Keep);
    }

    #[test]
    fn wire_in_cap_header_uses_probe_without_decoding_invalid_tail() {
        // Regression for the hot-path perf defect: stored history is lowered on
        // every turn, so in-cap images must be classified from a bounded header
        // probe and must not allocate/decode the full multi-MB payload. Make the
        // payload longer than the probe and corrupt the tail; the old full-decode
        // path would fail to decode, while the probe path can still read the PNG
        // dimensions from the first bytes and return Keep immediately.
        let mut bytes = png_bytes(64, 64);
        let target_len = (DIMENSION_PROBE_B64_CHARS / 4 * 3) + 4096;
        bytes.resize(target_len, 0xA5);
        let mut b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        assert!(b64.len() > DIMENSION_PROBE_B64_CHARS);
        b64.replace_range(b64.len() - 1.., "@");

        assert_eq!(guard_wire_image_base64(&b64), WireImageOutcome::Keep);
    }

    #[test]
    fn wire_oversized_header_with_invalid_tail_drops_instead_of_rewedging() {
        // If the bounded probe proves the image is oversized but the full base64
        // payload is corrupt, sending the original would still hit the provider
        // dimension cap. Drop it with a placeholder directive instead.
        let mut bytes = png_bytes(MAX_IMAGE_DIMENSION + 1, 1);
        let target_len = (DIMENSION_PROBE_B64_CHARS / 4 * 3) + 4096;
        bytes.resize(target_len, 0xA5);
        let mut b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        assert!(b64.len() > DIMENSION_PROBE_B64_CHARS);
        b64.replace_range(b64.len() - 1.., "@");

        assert_eq!(
            guard_wire_image_base64(&b64),
            WireImageOutcome::Drop {
                width: MAX_IMAGE_DIMENSION + 1,
                height: 1,
            }
        );
    }

    #[test]
    fn wire_non_base64_is_kept() {
        assert_eq!(guard_wire_image_base64("@@@ not base64 @@@"), WireImageOutcome::Keep);
    }

    #[test]
    fn placeholder_names_dimensions_and_limit() {
        let note = oversized_placeholder(1200, 12000);
        assert!(note.contains("1200x12000"));
        assert!(note.contains("8000"));
    }
}
