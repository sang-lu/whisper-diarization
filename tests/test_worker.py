import queue

import pytest

import worker
from jobstore import JobStore


class FakeModels:
    device = "cpu"
    whisper_model = object()
    whisper_pipeline = object()
    alignment_model = object()
    alignment_tokenizer = object()
    diarizer_model = object()
    punct_model = object()


@pytest.fixture
def store(tmp_path):
    return JobStore(str(tmp_path))


def _patch_happy_path(monkeypatch, tmp_path):
    monkeypatch.setattr(worker.faster_whisper, "decode_audio", lambda path: "fake-waveform")
    monkeypatch.setattr(worker.pipeline, "separate_vocals", lambda audio, out, dev: audio)
    monkeypatch.setattr(
        worker.pipeline,
        "transcribe",
        lambda *a, **k: ("hello world", type("Info", (), {"language": "en"})()),
    )
    monkeypatch.setattr(worker.pipeline, "align", lambda *a, **k: ["word_ts"])
    monkeypatch.setattr(worker.pipeline, "diarize", lambda *a, **k: [(0, 500, 0)])
    monkeypatch.setattr(
        worker.pipeline,
        "map_words",
        lambda *a, **k: [
            {"word": "hello", "start_time": 0, "end_time": 250, "speaker": 0},
            {"word": "world", "start_time": 250, "end_time": 500, "speaker": 0},
        ],
    )
    monkeypatch.setattr(worker.pipeline, "restore_punctuation", lambda punct_model, wsm, lang: wsm)


def test_process_one_job_writes_completed_status_and_result(monkeypatch, store, tmp_path):
    _patch_happy_path(monkeypatch, tmp_path)
    job_id = store.create_job(
        b"fake-audio",
        ".wav",
        {
            "language": None,
            "no_stem": True,
            "suppress_numerals": False,
            "batch_size": 8,
        },
    )

    worker.process_one_job(FakeModels(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    status = store.get_status(job_id)
    assert status == {"status": "completed"}
    result = store.get_result(job_id)
    assert result["text"] == "hello world"
    assert result["language_code"] == "en"


def test_process_one_job_writes_failed_status_on_exception(monkeypatch, store, tmp_path):
    monkeypatch.setattr(worker.faster_whisper, "decode_audio", lambda path: "fake-waveform")
    monkeypatch.setattr(worker.pipeline, "separate_vocals", lambda audio, out, dev: audio)

    def boom(*a, **k):
        raise RuntimeError("transcription exploded")

    monkeypatch.setattr(worker.pipeline, "transcribe", boom)
    job_id = store.create_job(
        b"fake-audio",
        ".wav",
        {
            "language": None,
            "no_stem": True,
            "suppress_numerals": False,
            "batch_size": 8,
        },
    )

    worker.process_one_job(FakeModels(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    status = store.get_status(job_id)
    assert status["status"] == "failed"
    assert "transcription exploded" in status["error"]


def test_process_one_job_respects_no_stem_false(monkeypatch, store, tmp_path):
    _patch_happy_path(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        worker.pipeline,
        "separate_vocals",
        lambda audio, out, dev: calls.append(True) or audio,
    )
    job_id = store.create_job(
        b"fake-audio",
        ".wav",
        {
            "language": None,
            "no_stem": False,
            "suppress_numerals": False,
            "batch_size": 8,
        },
    )

    worker.process_one_job(FakeModels(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    assert calls == [True]


def test_worker_loop_stops_on_none_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(worker.pipeline, "load_models", lambda *a, **k: FakeModels())
    q = queue.Queue()
    q.put(None)

    worker.worker_loop(q, str(tmp_path), "tiny.en", "cpu", "sortformer")


def test_worker_loop_processes_queued_job_then_stops(monkeypatch, tmp_path):
    _patch_happy_path(monkeypatch, tmp_path)
    monkeypatch.setattr(worker.pipeline, "load_models", lambda *a, **k: FakeModels())

    store = JobStore(str(tmp_path))
    job_id = store.create_job(
        b"fake-audio",
        ".wav",
        {
            "language": None,
            "no_stem": True,
            "suppress_numerals": False,
            "batch_size": 8,
        },
    )

    q = queue.Queue()
    q.put(job_id)
    q.put(None)

    worker.worker_loop(q, str(tmp_path), "tiny.en", "cpu", "sortformer")

    assert store.get_status(job_id) == {"status": "completed"}
