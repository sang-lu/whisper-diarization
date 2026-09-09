import glob
import json
import os
import shutil
import uuid

ALLOWED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".webm"}
)


class JobStore:
    def __init__(self, jobs_dir: str):
        self.jobs_dir = jobs_dir
        os.makedirs(self.jobs_dir, exist_ok=True)

    def _job_dir(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, job_id)

    def _require_job_dir(self, job_id: str) -> str:
        job_dir = self._job_dir(job_id)
        if not os.path.isdir(job_dir):
            raise FileNotFoundError(f"No such job: {job_id}")
        return job_dir

    def _atomic_write_json(self, path: str, data: dict) -> None:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)

    def create_job(self, audio_bytes: bytes, extension: str, params: dict) -> str:
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(f"Disallowed audio extension: {extension!r}")

        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        os.makedirs(job_dir)

        with open(os.path.join(job_dir, f"audio{extension}"), "wb") as f:
            f.write(audio_bytes)

        self._atomic_write_json(os.path.join(job_dir, "params.json"), params)
        self._atomic_write_json(os.path.join(job_dir, "status.json"), {"status": "queued"})

        return job_id

    def audio_path(self, job_id: str) -> str:
        job_dir = self._require_job_dir(job_id)
        matches = glob.glob(os.path.join(job_dir, "audio.*"))
        if not matches:
            raise FileNotFoundError(f"No audio file found for job: {job_id}")
        return matches[0]

    def params(self, job_id: str) -> dict:
        job_dir = self._require_job_dir(job_id)
        with open(os.path.join(job_dir, "params.json"), encoding="utf-8") as f:
            return json.load(f)

    def get_status(self, job_id: str) -> dict:
        job_dir = self._require_job_dir(job_id)
        with open(os.path.join(job_dir, "status.json"), encoding="utf-8") as f:
            return json.load(f)

    def set_status(self, job_id: str, status: dict) -> None:
        job_dir = self._require_job_dir(job_id)
        self._atomic_write_json(os.path.join(job_dir, "status.json"), status)

    def set_result(self, job_id: str, result: dict) -> None:
        job_dir = self._require_job_dir(job_id)
        self._atomic_write_json(os.path.join(job_dir, "result.json"), result)

    def get_result(self, job_id: str) -> dict:
        job_dir = self._require_job_dir(job_id)
        with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as f:
            return json.load(f)

    def job_exists(self, job_id: str) -> bool:
        return os.path.isdir(self._job_dir(job_id))

    def delete_job(self, job_id: str) -> None:
        job_dir = self._require_job_dir(job_id)
        shutil.rmtree(job_dir)
