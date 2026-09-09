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

- The FastAPI process never touches CUDA/torch **at the import level, not
  just at runtime** — `api_server.py` must not import `worker.py` directly,
  since `worker.py` imports `pipeline.py`, which imports `torch`. A small
  `worker_entrypoint.py` module (§5.3) breaks that chain: its only
  top-level content is a function whose body does `from worker import
  worker_loop`, so the import happens inside the spawned child process
  when the function runs, never in the parent. This keeps the async event
  loop responsive regardless of how long GPU work takes, and keeps torch
  out of the FastAPI process's memory entirely.
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

Extracted from `diarize.py`, as **stage functions** rather than one
monolithic `run_pipeline`, so the CLI and the API server can each control
model lifetime independently:

- `load_models(whisper_model_name, device, diarizer) -> Models`
  Loads the Whisper model, alignment model, diarizer model, and
  punctuation model. Returns a small dataclass bundling them. Any subset
  of these fields may be `None` — see below.
- `transcribe(whisper_model, whisper_pipeline, audio_waveform, *, language, suppress_numerals, batch_size) -> (full_transcript, info)`
- `align(alignment_model, alignment_tokenizer, audio_waveform, full_transcript, language, batch_size) -> word_timestamps`
- `diarize(diarizer_model, audio_waveform) -> speaker_ts`
- `restore_punctuation(punct_model, wsm, language) -> wsm` (no-op passthrough if `language not in punct_model_langs`)
- `build_mappings(word_timestamps, speaker_ts) -> (wsm, ssm)` (wraps the
  existing `helpers.get_words_speaker_mapping` +
  `get_realigned_ws_mapping_with_punctuation` +
  `get_sentences_speaker_mapping` calls)

**`diarize.py` keeps its current memory profile.** It still calls
`load_models()` for one stage at a time (e.g. `load_models(name, device,
diarizer=None)` to get only the Whisper models), uses it, `del`s it, and
calls `empty_cache()`, exactly as today — it just calls the extracted
stage functions instead of inlining the code. Peak VRAM for the CLI stays
`max(whisper, alignment, diarizer)`, unchanged from today.

**The API server's `worker.py` calls `load_models()` once with everything
populated**, then calls the stage functions per job without ever
`del`-ing. This means, unlike the CLI, a worker's peak VRAM is
`whisper + alignment + diarizer + punctuation` resident simultaneously —
see §8 for the sizing implication.

### 5.2 `api_server.py` (new)

The launcher + FastAPI app:

- CLI args: `--host`, `--port`, `--max-parallel` (default `1`),
  `--whisper-model`, `--device`, `--diarizer`, `--jobs-dir` (default
  `./jobs`).
- On startup: creates the jobs directory, creates a
  `multiprocessing.Queue`, starts `--max-parallel` worker processes with
  `target=worker_entrypoint.run` (see §5.3), then runs uvicorn.
- Routes (see §6).
- `api_server.py` itself imports only `jobstore` and `worker_entrypoint`
  (plus FastAPI/uvicorn/stdlib) — it must never import `worker` or
  `pipeline` directly.

### 5.3 `worker_entrypoint.py` (new)

A one-function module with no top-level imports beyond the standard
library:

```python
def run(job_queue, jobs_dir, whisper_model_name, device, diarizer):
    from worker import worker_loop

    worker_loop(job_queue, jobs_dir, whisper_model_name, device, diarizer)
```

This is the actual `multiprocessing.Process(target=...)` passed from
`api_server.py`. Since `multiprocessing`'s `spawn` context resolves a
`Process` target by module + qualified name and only calls it inside the
freshly-started child interpreter, the `from worker import worker_loop`
line — and everything it drags in (`pipeline`, `faster_whisper`, `torch`)
— never executes in the parent (FastAPI/uvicorn) process.

### 5.4 `worker.py` (new)

- `worker_loop(queue, jobs_dir, whisper_model, device, diarizer)`:
  calls `pipeline.load_models(...)` once (all four models), then loops
  forever: `job_id = queue.get()` → read `jobs/{job_id}/params.json` →
  write `status.json` (atomically, see §9) as `{"status": "processing"}`
  → run the stage functions from §5.1 in sequence → convert the resulting
  `wsm` to the output schema (§7) → write `result.json` and `status.json`
  (`{"status": "completed"}`), both atomically. On exception, writes
  `status.json = {"status": "failed", "error": "<message>"}` and continues
  the loop (one bad file must not kill the worker).

## 6. API Contract

### `POST /jobs`

Multipart form:
- `file` (required): the audio file.
- `language` (optional): same choices as CLI `--language`.
- `no_stem` (optional bool, default `true`). Default is `true` (unlike the
  CLI's `stemming=True` default) because stemming shells out to a
  separate `demucs` process per job — see §8 for why that matters under
  concurrency. Callers that want source separation opt in explicitly.
- `suppress_numerals` (optional bool, default `false`).
- `batch_size` (optional int, default `8`).

The uploaded file is saved to `jobs/{job_id}/audio<ext>`, where `ext` is
taken from an allowlist derived from the upload's declared content-type /
filename suffix (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.mp4`,
`.webm`) — never the raw client-supplied filename, which is discarded
after extension extraction. This avoids passing untrusted path
components into the stemming subprocess call.

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
  "audio_duration": 260,
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

Notes (verified against the reference `assemblyai.json`, where
`len(utterances) == len(itertools.groupby(words, key=speaker))`, i.e. one
utterance per speaker turn, not per sentence):
- **`utterances` are built by grouping `wsm` (word-speaker mapping) on
  speaker change only** — a new utterance starts whenever
  `word["speaker"] != previous_word["speaker"]`. This does **not** use
  `ssm` (`helpers.get_sentences_speaker_mapping`), because `ssm` also
  splits mid-turn on sentence boundaries and would produce far more,
  shorter utterances than AssemblyAI's format does. `ssm` remains used
  only by the CLI's existing `.txt`/`.srt` output, which is unaffected by
  this spec.
  - Each utterance's `start`/`end` are the first word's `start_time` and
    last word's `end_time` in its group.
  - Each utterance's `text` is its words joined with `" "`.
  - Each utterance's `words` list is exactly the grouped `wsm` entries,
    mapped to `{"text": w["word"], "start": w["start_time"], "end": w["end_time"], "speaker": <letter>}`.
- **Speaker letters**: `wsm`/`ssm` speaker values are the diarizer's raw
  int IDs (not necessarily `0, 1, 2...` contiguous or first-seen-ordered).
  Assign `"A", "B", "C"...` **by order of first appearance** while
  scanning `wsm` — i.e. the first distinct speaker ID encountered becomes
  `"A"`, the second distinct ID becomes `"B"`, etc. — matching AssemblyAI's
  convention, not `chr(ord("A") + speaker_id)`.
- **`text`** (top-level) is the join of the *final*, post-punctuation,
  post-realignment `wsm` words (`" ".join(w["word"] for w in wsm)`) — not
  `full_transcript` from the raw Whisper output, which predates
  punctuation restoration and word realignment and will not match the
  concatenation of `words`/`utterances`.
- **`audio_duration`** is an integer number of seconds
  (`int(len(audio_waveform) / 16000)`), matching the reference file's
  `260` (not `260.4`).
- `start`/`end` are milliseconds, matching AssemblyAI and matching what
  `helpers.get_words_speaker_mapping` already computes.
- No `confidence`/`channel` fields — this pipeline doesn't produce
  per-word confidence scores or multi-channel audio.

## 8. Concurrency & Configuration

- `--max-parallel` (default `1`) = number of worker processes = number of
  resident copies of the full model stack. As noted in §5.1, each worker
  keeps **all four models loaded simultaneously** (no `del`/`empty_cache`
  between stages, unlike the CLI), so peak VRAM is approximately:
  `--max-parallel × (whisper_model_vram + alignment_model_vram + diarizer_vram + punct_model_vram)`.
  Given the target deployment (single shared 48GB GPU, other workloads
  already running on it), the default of `1` is intentionally
  conservative; operators raise it only after confirming enough free
  VRAM for N copies using this formula.
- If a job has `no_stem=false` (source separation enabled), the worker's
  stemming stage shells out to a separate `python -m demucs.separate`
  process, which loads its own model into GPU memory **outside** the
  worker-process accounting above. With `--max-parallel > 1`, up to N such
  demucs processes can run concurrently, each adding to peak VRAM. This is
  why the API default is `no_stem=true` (§6) — operators who need stemming
  under high concurrency must account for this separately.
- Requests beyond the running jobs simply sit in the `multiprocessing.Queue`
  (FIFO) — no additional queue library needed.
- `--whisper-model`, `--device`, `--diarizer` are fixed for the server's
  lifetime (they determine what each worker loads). Changing them requires
  restarting the server.

## 9. Error Handling

- Upload validation: reject unreadable/empty files, or files whose
  extension isn't in the allowlist (§6), with `400` before queuing.
- Pipeline failures (bad audio, OOM, unsupported language, etc.) are
  caught in the worker loop per job and reported via `status.json`
  (`"failed"` + message) — never crash the worker process itself.
- If a worker process dies unexpectedly (e.g., segfault), jobs it already
  pulled off the queue stay `"processing"` forever; that failure mode is
  accepted for v1 (documented as a known limitation) rather than adding
  a heartbeat/watchdog now.
- **Atomic status/result writes**: `status.json` and `result.json` are
  never written in place. Each write goes to a sibling temp file
  (`status.json.tmp` / `result.json.tmp`) which is then moved into place
  with `os.replace()` (atomic on POSIX and Windows). This prevents a
  `GET /jobs/{id}` poll from ever reading a partially-written file and
  getting a JSON decode error / 500.
- The stemming subprocess call is built with `subprocess.run([...])`
  (argument list, `shell=False`), not a `os.system(f"...")` shell string,
  since the audio path now originates from an HTTP upload rather than a
  trusted CLI argument.

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
- A separate, fast regression test asserts the import boundary from §4/§5.2
  directly: run `python -c "import api_server"` in a fresh subprocess and
  assert `torch`, `faster_whisper`, `pipeline`, and `worker` are absent
  from `sys.modules` afterward. This is what would have caught
  `api_server.py` importing `worker` at module level.

## 11. Future Work (explicitly out of scope for this spec)

- Multi-GPU worker assignment (`cuda:0`, `cuda:1`, ...).
- Persisting job metadata in a real database instead of files, for
  higher job volumes.
- Webhook/callback delivery instead of polling.
- Confidence scores per word if a future Whisper/alignment path exposes
  them.
