import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(REPO_ROOT, "tests", "assets", "test.opus")


@pytest.mark.timeout(600)
def test_diarize_cli_produces_txt_and_srt(tmp_path):
    audio_copy = tmp_path / "test.opus"
    audio_copy.write_bytes(open(AUDIO, "rb").read())

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "diarize.py"),
            "-a", str(audio_copy),
            "--whisper-model", "tiny.en",
            "--diarizer", "sortformer",
            "--device", "cpu",
            "--no-stem",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    txt_path = tmp_path / "test.txt"
    srt_path = tmp_path / "test.srt"
    assert txt_path.exists()
    assert srt_path.exists()

    txt_content = txt_path.read_text(encoding="utf-8-sig")
    assert txt_content.strip() != ""
    assert ":" in txt_content
