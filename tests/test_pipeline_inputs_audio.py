"""Tests for src/pipeline/inputs/audio.py — URL fetch, base64 decode, validation,
and stereo normalisation. ffmpeg is mocked so tests run without the binary."""

import base64
import os
import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.pipeline.inputs.audio import (
    MAX_DOWNLOAD_BYTES,
    materialize_audio,
)

DUMMY_WAV = b"RIFF" + b"\x00" * 40


def _mock_response(content: bytes, status_code: int = 200, content_type: str = "audio/wav"):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.reason_phrase = "OK" if status_code == 200 else "ERR"
    resp.headers = {"content-type": content_type}
    resp.content = content
    return resp


@pytest.fixture(autouse=True)
def _fake_ffmpeg(request):
    """Replace ffmpeg with a byte-copy: read the -i input, write to the
    output positional argument. Lets the rest of materialize_audio's
    tempfile/path logic exercise as before. Tests that need ffmpeg failures
    opt out via ``@pytest.mark.no_fake_ffmpeg``."""
    if request.node.get_closest_marker("no_fake_ffmpeg"):
        yield
        return

    def _ffmpeg_copy(cmd, **kwargs):
        # cmd is the ffmpeg argv. Walk it: -i <in> ... <out>
        in_path = cmd[cmd.index("-i") + 1]
        out_path = cmd[-1]
        with open(in_path, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("src.pipeline.inputs.audio.subprocess.run", side_effect=_ffmpeg_copy) as m:
        yield m


class TestMaterializeAudio:
    def test_url_fetch_writes_tempfile(self):
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(DUMMY_WAV)):
            path = materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)
        try:
            assert os.path.exists(path)
            assert path.endswith(".stereo.wav")
            with open(path, "rb") as f:
                assert f.read() == DUMMY_WAV
        finally:
            os.unlink(path)

    def test_url_fetch_sends_browser_user_agent(self):
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(DUMMY_WAV)) as mock_get:
            path = materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)
        try:
            headers = mock_get.call_args.kwargs["headers"]
            ua = headers["User-Agent"]
            assert "Mozilla/" in ua and "python-httpx" not in ua.lower()
        finally:
            os.unlink(path)

    def test_url_403_message_includes_host(self):
        with patch(
            "src.pipeline.inputs.audio.httpx.get",
            return_value=_mock_response(b"", status_code=403),
        ):
            with pytest.raises(ValueError, match="403.*example.com"):
                materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)

    def test_b64_decode_writes_tempfile(self):
        b64 = base64.b64encode(DUMMY_WAV).decode("ascii")
        path = materialize_audio(audio_url=None, audio_b64=b64)
        try:
            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read() == DUMMY_WAV
        finally:
            os.unlink(path)

    def test_b64_strips_data_uri_prefix(self):
        data_uri = "data:audio/wav;base64," + base64.b64encode(DUMMY_WAV).decode("ascii")
        path = materialize_audio(audio_url=None, audio_b64=data_uri)
        try:
            with open(path, "rb") as f:
                assert f.read() == DUMMY_WAV
        finally:
            os.unlink(path)

    def test_b64_tolerates_embedded_whitespace(self):
        b64 = base64.b64encode(DUMMY_WAV).decode("ascii")
        wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76)) + "\n"
        path = materialize_audio(audio_url=None, audio_b64=wrapped)
        try:
            with open(path, "rb") as f:
                assert f.read() == DUMMY_WAV
        finally:
            os.unlink(path)

    def test_rejects_both_inputs(self):
        with pytest.raises(ValueError, match="at most one"):
            materialize_audio(audio_url="https://x", audio_b64="abc")

    def test_rejects_neither_input(self):
        with pytest.raises(ValueError, match="required"):
            materialize_audio(audio_url=None, audio_b64=None)

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="http"):
            materialize_audio(audio_url="ftp://x/y.wav", audio_b64=None)

    def test_rejects_non_audio_content_type(self):
        with patch(
            "src.pipeline.inputs.audio.httpx.get",
            return_value=_mock_response(b"<html></html>", content_type="text/html"),
        ):
            with pytest.raises(ValueError, match="Content-Type"):
                materialize_audio(audio_url="https://x/y", audio_b64=None)

    def test_accepts_octet_stream_content_type(self):
        with patch(
            "src.pipeline.inputs.audio.httpx.get",
            return_value=_mock_response(DUMMY_WAV, content_type="application/octet-stream"),
        ):
            path = materialize_audio(audio_url="https://x/y.wav", audio_b64=None)
        try:
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_rejects_oversize_url(self):
        big = b"x" * (MAX_DOWNLOAD_BYTES + 1)
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(big)):
            with pytest.raises(ValueError, match="exceeds"):
                materialize_audio(audio_url="https://x/y.wav", audio_b64=None)

    def test_rejects_http_error(self):
        with patch(
            "src.pipeline.inputs.audio.httpx.get",
            return_value=_mock_response(b"", status_code=404),
        ):
            with pytest.raises(ValueError, match="404"):
                materialize_audio(audio_url="https://x/missing", audio_b64=None)

    def test_rejects_invalid_base64(self):
        with pytest.raises(ValueError, match="base64"):
            materialize_audio(audio_url=None, audio_b64="this is not !!base64!!")

    def test_output_always_stereo_wav_regardless_of_input_format(self, _fake_ffmpeg):
        """ffmpeg normalises every payload to 2-channel PCM WAV — the input
        format only affects the intermediate tempfile we hand ffmpeg, not
        the final path returned to the caller."""
        with patch(
            "src.pipeline.inputs.audio.httpx.get",
            return_value=_mock_response(DUMMY_WAV, content_type="audio/mpeg"),
        ):
            path = materialize_audio(audio_url="https://x/y.mp3", audio_b64=None)
        try:
            assert path.endswith(".stereo.wav")
            # ffmpeg input was the .mp3 intermediate; output is the .stereo.wav.
            cmd = _fake_ffmpeg.call_args.args[0]
            assert cmd[cmd.index("-i") + 1].endswith(".mp3")
            assert cmd[-1].endswith(".stereo.wav")
            assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2"
        finally:
            os.unlink(path)

    def test_flac_input_routed_through_ffmpeg(self, _fake_ffmpeg):
        with patch(
            "src.pipeline.inputs.audio.httpx.get",
            return_value=_mock_response(DUMMY_WAV, content_type="audio/flac"),
        ):
            path = materialize_audio(audio_url="https://x/y.flac", audio_b64=None)
        try:
            cmd = _fake_ffmpeg.call_args.args[0]
            assert cmd[cmd.index("-i") + 1].endswith(".flac")
            assert path.endswith(".stereo.wav")
        finally:
            os.unlink(path)


class TestStereoNormalization:
    """LTX-2.3 audio VAE conv_in expects 2 channels — pin the contract."""

    def test_ffmpeg_called_with_ac_2_and_pcm_s16le(self, _fake_ffmpeg):
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(DUMMY_WAV)):
            path = materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)
        try:
            cmd = _fake_ffmpeg.call_args.args[0]
            assert cmd[0] == "ffmpeg"
            assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2", \
                f"missing -ac 2 (stereo) in ffmpeg command: {cmd}"
            assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "pcm_s16le", \
                f"missing -c:a pcm_s16le in ffmpeg command: {cmd}"
        finally:
            os.unlink(path)

    def test_intermediate_tempfile_deleted_after_normalisation(self, _fake_ffmpeg):
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(DUMMY_WAV)):
            path = materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)
        try:
            cmd = _fake_ffmpeg.call_args.args[0]
            intermediate = cmd[cmd.index("-i") + 1]
            assert not os.path.exists(intermediate), \
                "mono/source tempfile should be cleaned up after stereo normalisation"
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    @pytest.mark.no_fake_ffmpeg
    def test_ffmpeg_failure_surfaces_as_value_error(self):
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(DUMMY_WAV)), \
             patch("src.pipeline.inputs.audio.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd=["ffmpeg"], stderr=b"Invalid data found when processing input",
            )
            with pytest.raises(ValueError, match="ffmpeg failed.*Invalid data"):
                materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)

    @pytest.mark.no_fake_ffmpeg
    def test_ffmpeg_timeout_surfaces_as_value_error(self):
        with patch("src.pipeline.inputs.audio.httpx.get", return_value=_mock_response(DUMMY_WAV)), \
             patch("src.pipeline.inputs.audio.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=60)
            with pytest.raises(ValueError, match="timed out"):
                materialize_audio(audio_url="https://example.com/x.wav", audio_b64=None)
