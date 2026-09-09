# Transcription API Server — Design

Date: 2026-09-09
Status: Approved for implementation

## 1. Overview

This project currently exposes its Whisper + NeMo diarization pipeline only
as a one-shot CLI script (`diarize.py`). This spec adds an HTTP API server
that accepts audio file uploads, runs the same pipeline, and returns
diarized transcription results as JSON. Multiple audio files can be
submitted concurrently; a configurable limit caps how many are actually
processed in parallel, with the rest queued.

## 2. Goals

- Accept audio uploads over HTTP and return diarized transcripts as JSON.
- Support concurrent submissions from multiple clients.
- Cap real parallel GPU work via a `--max-parallel` server argument; excess
  jobs wait in a FIFO queue and are processed as capacity frees up.
- Avoid reloading the (large) Whisper/alignment/diarization/punctuation
  models on every request — load once per worker, reuse across jobs.
- Produce JSON shaped as a trimmed subset of the AssemblyAI transcript
  schema (`assemblyai.json` in the repo root was used as the reference),
  since that's the shape the consuming client already expects.

## 3. Non-goals

- Changing `diarize.py` / `diarize_parallel.py` CLI behavior (they keep
  working as they do today; the pipeline logic they use is extracted, not
  removed).
- Authentication/authorization, rate limiting, TLS termination — assumed to
  be handled by a reverse proxy or the deployment environment if needed.
- Multi-GPU scheduling logic — `--max-parallel` workers all use the single
  `--device` given at startup. (Multi-GPU is listed under Future Work.)
- Streaming/partial results, webhooks — only queue + poll is in scope.
- Full AssemblyAI field parity (sentiment, chapters, redaction, etc.) — only
  the fields this pipeline can actually produce are included.

## 4. Architecture

```
                         ┌─────────────────────────┐
   client ── HTTP ──────▶│   FastAPI app (uvicorn)  │
                         │   - POST /jobs           │
                         │   - GET  /jobs/{id}      │
                         │   - DELETE /jobs/{id}    │
                         └──────────┬───────────────┘
                                    │ job_id (multiprocessing.Queue)
                                    ▼
                    ┌───────────────────────────────┐
                    │      N worker processes        │
                    │  (spawn, one per --max-parallel)│
                    │  each: loads models ONCE,       │
                    │  then loops: pull job_id,       │
                    │  run pipeline, write result      │
                    └───────────────────────────────┘
                                    │
                                    ▼
                     jobs/{job_id}/  (on disk)
                       - audio.<ext>
                       - params.json
                       - status.json
                       - result.json (on success)
```

- The FastAPI process never touches CUDA/torch; it only manages HTTP,
  files, and the queue. This keeps the async event loop responsive
  regardless of how long GPU work takes.
- Worker processes are started with `multiprocessing.get_context("spawn")`
  (required for CUDA — `fork` after CUDA init is unsafe) by the launcher
  script before uvicorn starts serving.
- Job state lives on disk (not in a `multiprocessing.Manager` dict), so:
  - it survives an API process restart while workers keep draining the
    queue,
  - there's no shared-memory/pickling complexity for large result payloads
    across the process boundary.

## 5. Components

### 5.1 `pipeline.py` (new)

Extracted from `diarize.py`. Two entry points:

- `load_models(whisper_model_name, device, diarizer) -> Models`
  Loads the Whisper model, alignment model, diarizer model, and
  punctuation model once. Returns a small dataclass/namedtuple bundling
  them. This is the expensive, one-time-per-worker call.

- `run_pipeline(models, audio_path, *, language=None, stemming=True, suppress_numerals=False, batch_size=8) -> PipelineResult`
  Runs stemming → transcription → forced alignment → diarization →
  punctuation restoration → word/sentence speaker mapping, using the
  already-loaded `models`. Returns the same kind of structure `diarize.py`
  currently derives (`wsm`, `ssm`, detected language, audio duration) so
  both the CLI and the API server can build their own output format from
  it.

`diarize.py` is updated to call `load_models()` once then `run_pipeline()`
once, instead of inlining the logic — behavior unchanged.

### 5.2 `api_server.py` (new)

The launcher + FastAPI app:

- CLI args: `--host`, `--port`, `--max-parallel` (default `1`),
  `--whisper-model`, `--device`, `--diarizer`, `--jobs-dir` (default
  `./jobs`).
- On startup: creates the jobs directory, creates a
  `multiprocessing.Queue`, starts `--max-parallel` worker processes (each
  running `worker.worker_loop(queue, jobs_dir, whisper_model, device,
  diarizer)`), then runs uvicorn.
- Routes (see §6).

### 5.3 `worker.py` (new)

- `worker_loop(queue, jobs_dir, whisper_model, device, diarizer)`:
  calls `pipeline.load_models(...)` once, then loops forever:
  `job_id = queue.get()` → read `jobs/{job_id}/params.json` → write
  `status.json = {"status": "processing"}` → call `run_pipeline` → convert
  result to the output schema (§7) → write `result.json` → write
  `status.json = {"status": "completed"}`. On exception, writes
  `status.json = {"status": "failed", "error": "<message>"}` and continues
  the loop (one bad file must not kill the worker).

## 6. API Contract

### `POST /jobs`

Multipart form:
- `file` (required): the audio file.
- `language` (optional): same choices as CLI `--language`.
- `no_stem` (optional bool, default `false`).
- `suppress_numerals` (optional bool, default `false`).
- `batch_size` (optional int, default `8`).

Response `202 Accepted`:
```json
{"job_id": "b3f1...", "status": "queued"}
```

### `GET /jobs/{job_id}`

Response `200 OK`, one of:
```json
{"job_id": "...", "status": "queued"}
{"job_id": "...", "status": "processing"}
{"job_id": "...", "status": "failed", "error": "..."}
{"job_id": "...", "status": "completed", "result": { ... see §7 ... }}
```
`404` if `job_id` is unknown.

### `DELETE /jobs/{job_id}`

Removes `jobs/{job_id}/` from disk. `204 No Content`, or `404` if unknown.

## 7. Output JSON Schema

A trimmed AssemblyAI-shaped subset — only fields this pipeline can
actually populate; everything else in the AssemblyAI schema is omitted
rather than emitted as `null`:

```json
{
  "id": "b3f1...",
  "status": "completed",
  "language_code": "en",
  "audio_duration": 260.4,
  "text": "full transcript text...",
  "utterances": [
    {
      "speaker": "A",
      "start": 2275,
      "end": 7975,
      "text": "sentence text...",
      "words": [
        {"text": "word", "start": 2275, "end": 2555, "speaker": "A"}
      ]
    }
  ]
}
```

Notes:
- `start`/`end` are milliseconds, matching AssemblyAI and matching what
  `helpers.get_words_speaker_mapping` already computes.
- Speaker labels are remapped from the pipeline's `0, 1, 2...` to
  `"A", "B", "C"...` to match AssemblyAI convention.
- No `confidence`/`channel` fields — this pipeline doesn't produce
  per-word confidence scores or multi-channel audio.
- `utterances` is built directly from `ssm` (sentence speaker mapping);
  each utterance's `words` slice comes from the corresponding `wsm`
  entries.

## 8. Concurrency & Configuration

- `--max-parallel` (default `1`) = number of worker processes = number of
  resident copies of the full model stack. Given the target deployment
  (single shared 48GB GPU, other workloads running on it), the default of
  `1` is intentionally conservative; operators raise it only if they've
  confirmed enough free VRAM for N copies.
- Requests beyond the running jobs simply sit in the `multiprocessing.Queue`
  (FIFO) — no additional queue library needed.
- `--whisper-model`, `--device`, `--diarizer` are fixed for the server's
  lifetime (they determine what each worker loads). Changing them requires
  restarting the server.

## 9. Error Handling

- Upload validation: reject unreadable/empty files with `400` before
  queuing.
- Pipeline failures (bad audio, OOM, unsupported language, etc.) are
  caught in the worker loop per job and reported via `status.json`
  (`"failed"` + message) — never crash the worker process itself.
- If a worker process dies unexpectedly (e.g., segfault), jobs it already
  pulled off the queue stay `"processing"` forever; that failure mode is
  accepted for v1 (documented as a known limitation) rather than adding
  a heartbeat/watchdog now.

## 10. Testing

- Extend `tests/` with `tests/test_api_server.py`: starts
  `api_server.py --max-parallel 1 --whisper-model tiny.en --diarizer sortformer`
  as a subprocess, `POST`s `tests/assets/test.opus` to `/jobs`, polls
  `GET /jobs/{id}` until `completed` or `failed`, and asserts the response
  matches the schema in §7 (non-empty `text`, at least one utterance with
  `speaker`/`start`/`end`/`text`/`words`).
- Wire this into `.github/workflows/test_run.yml` alongside the existing
  CLI-based checks (`tiny.en`, both diarizers), on CPU, same as today's
  CLI tests.

## 11. Future Work (explicitly out of scope for this spec)

- Multi-GPU worker assignment (`cuda:0`, `cuda:1`, ...).
- Persisting job metadata in a real database instead of files, for
  higher job volumes.
- Webhook/callback delivery instead of polling.
- Confidence scores per word if a future Whisper/alignment path exposes
  them.
