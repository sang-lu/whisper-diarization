import queue
import subprocess
import sys

from fastapi.testclient import TestClient

from api_server import create_app
from jobstore import JobStore


def _client(tmp_path, token=None):
    store = JobStore(str(tmp_path))
    job_queue = queue.Queue()
    app = create_app(store, job_queue, token=token)
    return TestClient(app), store, job_queue


def test_post_jobs_returns_queued_job_id_and_enqueues_it(tmp_path):
    client, store, job_queue = _client(tmp_path)

    response = client.post(
        "/jobs",
        files={"file": ("audio.wav", b"fake-audio-bytes", "audio/wav")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    assert store.job_exists(job_id)
    assert job_queue.get_nowait() == job_id


def test_post_jobs_rejects_disallowed_extension(tmp_path):
    client, _store, _q = _client(tmp_path)

    response = client.post(
        "/jobs",
        files={"file": ("payload.exe", b"data", "application/octet-stream")},
    )

    assert response.status_code == 400


def test_post_jobs_defaults_no_stem_true(tmp_path):
    client, store, _q = _client(tmp_path)

    response = client.post("/jobs", files={"file": ("audio.wav", b"data", "audio/wav")})
    job_id = response.json()["job_id"]

    assert store.params(job_id)["no_stem"] is True


def test_post_jobs_accepts_optional_params(tmp_path):
    client, store, _q = _client(tmp_path)

    response = client.post(
        "/jobs",
        files={"file": ("audio.wav", b"data", "audio/wav")},
        data={"language": "en", "no_stem": "false", "suppress_numerals": "true", "batch_size": "4"},
    )
    job_id = response.json()["job_id"]
    params = store.params(job_id)

    assert params == {
        "language": "en",
        "no_stem": False,
        "suppress_numerals": True,
        "batch_size": 4,
    }


def test_get_jobs_unknown_id_returns_404(tmp_path):
    client, _store, _q = _client(tmp_path)
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_get_jobs_returns_queued_status(tmp_path):
    client, _store, _q = _client(tmp_path)
    job_id = client.post("/jobs", files={"file": ("a.wav", b"d", "audio/wav")}).json()["job_id"]

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "status": "queued"}


def test_get_jobs_returns_completed_result(tmp_path):
    client, store, _q = _client(tmp_path)
    job_id = client.post("/jobs", files={"file": ("a.wav", b"d", "audio/wav")}).json()["job_id"]
    result = {"id": job_id, "status": "completed", "text": "hi"}
    store.set_status(job_id, {"status": "completed"})
    store.set_result(job_id, result)

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"] == result


def test_get_jobs_returns_failed_error(tmp_path):
    client, store, _q = _client(tmp_path)
    job_id = client.post("/jobs", files={"file": ("a.wav", b"d", "audio/wav")}).json()["job_id"]
    store.set_status(job_id, {"status": "failed", "error": "boom"})

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "status": "failed", "error": "boom"}


def test_delete_jobs_removes_job(tmp_path):
    client, store, _q = _client(tmp_path)
    job_id = client.post("/jobs", files={"file": ("a.wav", b"d", "audio/wav")}).json()["job_id"]

    response = client.delete(f"/jobs/{job_id}")

    assert response.status_code == 204
    assert not store.job_exists(job_id)


def test_delete_jobs_unknown_id_returns_404(tmp_path):
    client, _store, _q = _client(tmp_path)
    response = client.delete("/jobs/does-not-exist")
    assert response.status_code == 404


def test_configured_token_rejects_missing_credentials_on_all_routes(tmp_path):
    client, _store, _q = _client(tmp_path, token="server-secret")

    post_response = client.post(
        "/jobs",
        files={"file": ("audio.wav", b"data", "audio/wav")},
    )
    get_response = client.get("/jobs/does-not-exist")
    delete_response = client.delete("/jobs/does-not-exist")

    assert post_response.status_code == 401
    assert get_response.status_code == 401
    assert delete_response.status_code == 401


def test_configured_token_rejects_malformed_credentials(tmp_path):
    client, _store, _q = _client(tmp_path, token="server-secret")

    response = client.get(
        "/jobs/does-not-exist",
        headers={"Authorization": "Basic not-a-bearer-token"},
    )

    assert response.status_code == 401


def test_configured_token_rejects_incorrect_credentials(tmp_path):
    client, _store, _q = _client(tmp_path, token="server-secret")

    response = client.get(
        "/jobs/does-not-exist",
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert response.status_code == 401


def test_correct_token_allows_all_routes(tmp_path):
    client, store, _q = _client(tmp_path, token="server-secret")
    headers = {"Authorization": "Bearer server-secret"}

    post_response = client.post(
        "/jobs",
        files={"file": ("audio.wav", b"data", "audio/wav")},
        headers=headers,
    )
    job_id = post_response.json()["job_id"]
    get_response = client.get(f"/jobs/{job_id}", headers=headers)
    delete_response = client.delete(f"/jobs/{job_id}", headers=headers)

    assert post_response.status_code == 202
    assert get_response.status_code == 200
    assert delete_response.status_code == 204
    assert not store.job_exists(job_id)


def test_no_configured_token_allows_request_without_authentication(tmp_path):
    client, _store, _q = _client(tmp_path)

    response = client.get("/jobs/does-not-exist")

    assert response.status_code == 404


def test_importing_api_server_does_not_import_torch_or_model_code():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import api_server, sys\n"
            "leaked = {'torch', 'faster_whisper', 'pipeline', 'worker'} & set(sys.modules)\n"
            "assert not leaked, f'api_server.py pulled in: {leaked}'\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
