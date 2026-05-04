"""V2V scenario: per-call upstream invocation when a reference video is
supplied. The unified ``ICLoraPipeline`` build itself lives in ``i2v.py``
(``_build_unified``) and is shared — V2V's only specific code is the
per-call kwargs assembled in ``_run_v2v``: an optional first-frame image
plus the ``video_conditioning`` tuple.
"""

import logging

logger = logging.getLogger(__name__)


class V2VMixin:
    """Method used when the request supplies a reference video. Mixed into
    ``LTXVideoGenerator`` via multiple inheritance — do not instantiate
    directly. The unified pipeline build is inherited from ``I2VMixin``."""

    def _run_v2v(
        self, *,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        image_path: str | None,
        ref_video_path: str,
        reference_video_strength: float,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        from src.upstream import ImageConditioningInput
        images = (
            [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
            if image_path is not None else []
        )
        kwargs = dict(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            images=images,
            video_conditioning=[(ref_video_path, reference_video_strength)],
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
        )
        if tiling_config is not None:
            kwargs["tiling_config"] = tiling_config
        return self._pipeline(**kwargs)
