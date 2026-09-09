import logging
import os

import faster_whisper

import pipeline
from jobstore import JobStore
from schema import build_result


def process_one_job(models, store: JobStore, job_id: str, *, temp_dir: str) -> None:
    try:
        store.set_status(job_id, {"status": "processing"})
        params = store.params(job_id)
        audio_path = store.audio_path(job_id)

        os.makedirs(temp_dir, exist_ok=True)

        if params.get("no_stem", True):
            vocal_target = audio_path
        else:
            vocal_target = pipeline.separate_vocals(audio_path, temp_dir, models.device)

        audio_waveform = faster_whisper.decode_audio(vocal_target)

        full_transcript, info = pipeline.transcribe(
            models.whisper_model,
            models.whisper_pipeline,
            audio_waveform,
            language=params.get("language"),
            suppress_numerals=params.get("suppress_numerals", False),
            batch_size=params.get("batch_size", 8),
        )

        word_timestamps = pipeline.align(
            models.alignment_model,
            models.alignment_tokenizer,
            audio_waveform,
            full_transcript,
            info.language,
            params.get("batch_size", 8),
        )

        speaker_ts = pipeline.diarize(models.diarizer_model, audio_waveform)

        wsm = pipeline.map_words(word_timestamps, speaker_ts)
        wsm = pipeline.restore_punctuation(models.punct_model, wsm, info.language)
        wsm, _ssm = pipeline.finalize_mappings(wsm, speaker_ts)

        audio_duration_seconds = int(len(audio_waveform) / 16000)
        result = build_result(job_id, info.language, audio_duration_seconds, wsm)

        store.set_result(job_id, result)
        store.set_status(job_id, {"status": "completed"})
    except Exception as e:  # noqa: BLE001 - one bad job must not kill the worker
        logging.exception(f"Job {job_id} failed")
        store.set_status(job_id, {"status": "failed", "error": str(e)})


def worker_loop(
    job_queue, jobs_dir: str, whisper_model_name: str, device: str, diarizer: str
) -> None:
    models = pipeline.load_models(whisper_model_name, device, diarizer)
    store = JobStore(jobs_dir)
    temp_dir = os.path.join(jobs_dir, f"_worker_temp_{os.getpid()}")

    while True:
        job_id = job_queue.get()
        if job_id is None:
            return
        process_one_job(models, store, job_id, temp_dir=temp_dir)
