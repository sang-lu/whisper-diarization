def run(job_queue, jobs_dir: str, whisper_model_name: str, device: str, diarizer: str) -> None:
    from worker import worker_loop

    worker_loop(job_queue, jobs_dir, whisper_model_name, device, diarizer)
