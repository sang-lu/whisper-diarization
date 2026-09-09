import os
import socket
import subprocess
import sys
import time

import pytest
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(REPO_ROOT, "tests", "assets", "test.opus")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{base_url}/jobs/nonexistent-id", timeout=2)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise TimeoutError("api_server.py did not start listening in time")


@pytest.mark.timeout(900)
def test_end_to_end_transcription(tmp_path):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    jobs_dir = tmp_path / "jobs"

    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "api_server.py"),
            "--port", str(port),
            "--max-parallel", "1",
            "--whisper-model", "tiny.en",
            "--diarizer", "sortformer",
            "--device", "cpu",
            "--jobs-dir", str(jobs_dir),
        ],
    )
    try:
        _wait_for_server(base_url)

        with open(AUDIO, "rb") as f:
            response = requests.post(
                f"{base_url}/jobs",
                files={"file": ("test.opus", f, "audio/opus")},
            )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        deadline = time.time() + 600
        result = None
        while time.time() < deadline:
            poll = requests.get(f"{base_url}/jobs/{job_id}")
            assert poll.status_code == 200
            body = poll.json()
            if body["status"] == "completed":
                result = body["result"]
                break
            if body["status"] == "failed":
                pytest.fail(f"job failed: {body.get('error')}")
            time.sleep(2)

        assert result is not None, "job did not complete in time"
        assert result["text"].strip() != ""
        assert len(result["utterances"]) > 0
        for utt in result["utterances"]:
            assert utt["speaker"]
            assert isinstance(utt["start"], int)
            assert isinstance(utt["end"], int)
            assert utt["text"].strip() != ""
            assert len(utt["words"]) > 0
            for word in utt["words"]:
                assert set(word.keys()) == {"text", "start", "end", "speaker"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
