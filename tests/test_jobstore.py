import json

import pytest

from jobstore import ALLOWED_AUDIO_EXTENSIONS, JobStore


def test_create_job_writes_audio_params_and_queued_status(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"fake-audio-bytes", ".wav", {"language": "en"})

    job_dir = tmp_path / job_id
    assert (job_dir / "audio.wav").read_bytes() == b"fake-audio-bytes"
    assert json.loads((job_dir / "params.json").read_text()) == {"language": "en"}
    assert json.loads((job_dir / "status.json").read_text()) == {"status": "queued"}


def test_create_job_rejects_disallowed_extension(tmp_path):
    store = JobStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.create_job(b"data", ".exe", {})


def test_audio_path_returns_saved_file(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".mp3", {})
    assert store.audio_path(job_id) == str(tmp_path / job_id / "audio.mp3")


def test_audio_path_missing_job_raises(tmp_path):
    store = JobStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.audio_path("does-not-exist")


def test_set_and_get_status_roundtrip(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    store.set_status(job_id, {"status": "processing"})
    assert store.get_status(job_id) == {"status": "processing"}


def test_set_status_is_atomic_no_tmp_file_left_behind(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    store.set_status(job_id, {"status": "completed"})
    job_dir = tmp_path / job_id
    assert not (job_dir / "status.json.tmp").exists()
    assert (job_dir / "status.json").exists()


def test_set_and_get_result_roundtrip(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    result = {"id": job_id, "text": "hello world"}
    store.set_result(job_id, result)
    assert store.get_result(job_id) == result


def test_job_exists(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    assert store.job_exists(job_id) is True
    assert store.job_exists("nope") is False


def test_delete_job_removes_directory(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    store.delete_job(job_id)
    assert not (tmp_path / job_id).exists()


def test_delete_missing_job_raises(tmp_path):
    store = JobStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.delete_job("nope")


def test_allowed_extensions_constant():
    assert ALLOWED_AUDIO_EXTENSIONS == {
        ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".webm"
    }
