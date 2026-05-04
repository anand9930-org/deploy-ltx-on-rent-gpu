"""Request-time input materialisation for the unified ICLora pipeline.

I2V (``image.py``) and V2V (``video.py``) helpers crack caller-supplied
URLs / base64 payloads into a tempfile path that ``ICLoraPipeline``
consumes. Distinct from build-time helpers in ``core.py``: these run on
every request, not at boot.
"""

from src.pipeline.inputs.image import (
    derive_dims_from_image,
    materialize_image,
)
from src.pipeline.inputs.video import materialize_video

__all__ = [
    "derive_dims_from_image",
    "materialize_image",
    "materialize_video",
]
