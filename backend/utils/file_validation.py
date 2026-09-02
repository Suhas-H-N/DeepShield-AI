"""
Upload validation: enforce size limits and verify file content actually
matches its claimed type via magic-byte sniffing (never trust the filename
extension or the browser-supplied Content-Type header alone).
"""

from dataclasses import dataclass
from typing import Optional

# Minimal magic-byte signatures — avoids pulling in an extra dependency
# (python-magic) for a handful of well-known formats.
_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
}
_VIDEO_SIGNATURES = {
    b"\x00\x00\x00\x18ftyp": "video/mp4",
    b"\x00\x00\x00\x20ftyp": "video/mp4",
    b"RIFF": "video/avi",
    b"\x1aE\xdf\xa3": "video/webm",
}


@dataclass
class ValidationResult:
    ok: bool
    media_type: Optional[str] = None  # "image" | "video"
    reason: Optional[str] = None


def sniff_media_type(header: bytes) -> Optional[str]:
    for sig, _ in _IMAGE_SIGNATURES.items():
        if header.startswith(sig):
            return "image"
    for sig, _ in _VIDEO_SIGNATURES.items():
        if header.startswith(sig):
            return "video"
    # QuickTime/MOV containers vary; check for 'ftyp' anywhere in the first
    # bytes as a pragmatic fallback for mov/mp4 variants.
    if b"ftyp" in header[:32]:
        return "video"
    return None


def validate_upload(contents: bytes, max_image_mb: int, max_video_mb: int) -> ValidationResult:
    if not contents:
        return ValidationResult(ok=False, reason="Empty file")

    size_mb = len(contents) / (1024 * 1024)
    media_type = sniff_media_type(contents[:64])

    if media_type is None:
        return ValidationResult(
            ok=False,
            reason="Unrecognized or unsupported file content (only JPG, PNG, MP4, MOV, AVI, WEBM are accepted)",
        )

    limit = max_image_mb if media_type == "image" else max_video_mb
    if size_mb > limit:
        return ValidationResult(
            ok=False,
            reason=f"File too large ({size_mb:.1f}MB). Limit for {media_type} is {limit}MB.",
        )

    return ValidationResult(ok=True, media_type=media_type)
