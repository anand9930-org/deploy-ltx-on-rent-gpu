"""Request-time input materialisation for the pipeline.

I2V (``image.py``) helpers crack caller-supplied URLs / base64 payloads into
a tempfile path that the pipeline consumes.
"""

from src.pipeline.inputs.image import (
    derive_orientation,
    materialize_image,
)

__all__ = [
    "derive_orientation",
    "materialize_image",
]
