import argparse
import multiprocessing
import os
import secrets

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import worker_entrypoint
from jobstore import ALLOWED_AUDIO_EXTENSIONS, JobStore


def create_app(store: JobStore, job_queue, token: str | None = None) -> FastAPI:
    security = HTTPBearer(auto_error=False)

    def verify_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> None:
        if token is None:
            return
        if credentials is None or not secrets.compare_digest(credentials.credentials, token):
            raise HTTPException(status_code=401, detail="Invalid or missing token")

    app = FastAPI(dependencies=[Depends(verify_token)])

    @app.post("/jobs", status_code=202)
    async def create_job(
        file: UploadFile = File(...),
        language: str = Form(None),
        no_stem: bool = Form(True),
        suppress_numerals: bool = Form(False),
        batch_size: int = Form(8),
    ):
        extension = os.path.splitext(file.filename or "")[1].lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio extension: {extension!r}",
            )

        audio_bytes = await file.read()
        params = {
            "language": language,
            "no_stem": no_stem,
            "suppress_numerals": suppress_numerals,
            "batch_size": batch_size,
        }
        job_id = store.create_job(audio_bytes, extension, params)
        job_queue.put(job_id)

        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        if not store.job_exists(job_id):
            raise HTTPException(status_code=404, detail="Unknown job_id")

        status = store.get_status(job_id)
        if status["status"] == "completed":
            return {"job_id": job_id, "status": "completed", "result": store.get_result(job_id)}
        return {"job_id": job_id, **status}

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        if not store.job_exists(job_id):
            raise HTTPException(status_code=404, detail="Unknown job_id")
        store.delete_job(job_id)
        return Response(status_code=204)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--whisper-model", default="medium.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--diarizer", default="msdd", choices=["msdd", "sortformer"])
    parser.add_argument("--jobs-dir", default="./jobs")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    store = JobStore(args.jobs_dir)
    ctx = multiprocessing.get_context("spawn")
    job_queue = ctx.Queue()

    workers = []
    for _ in range(args.max_parallel):
        process = ctx.Process(
            target=worker_entrypoint.run,
            args=(job_queue, args.jobs_dir, args.whisper_model, args.device, args.diarizer),
            daemon=True,
        )
        process.start()
        workers.append(process)

    app = create_app(store, job_queue, token=args.token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
