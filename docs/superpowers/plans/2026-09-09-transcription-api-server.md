# Transcription API Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HTTP API server that accepts audio uploads, runs the existing Whisper+NeMo diarization pipeline, and returns AssemblyAI-shaped JSON, with a configurable cap on parallel jobs and FIFO queuing beyond that cap.

**Architecture:** Extract the pipeline in `diarize.py` into reusable stage functions in a new `pipeline.py` (transcribe, align, diarize, restore_punctuation, map/finalize mappings), keeping `diarize.py`'s per-stage load/delete/empty_cache VRAM behavior unchanged. A new `api_server.py` (FastAPI, single process, never touches CUDA) manages job files on disk and a `multiprocessing.Queue`; N `worker.py` processes (spawned once, one per `--max-parallel`) each load the full model stack once and keep it resident, pulling job IDs off the queue and writing `status.json`/`result.json` atomically.

**Tech Stack:** FastAPI, Uvicorn, python-multipart (upload parsing), pytest + httpx (`TestClient`) for tests. No new production dependencies beyond these three.

**Spec:** `docs/superpowers/specs/2026-09-09-transcription-api-server-design.md`

## Global Constraints

- CLI behavior of `diarize.py` / `diarize_parallel.py` must not change — same output files, same peak VRAM profile (max of one stage at a time, via `del` + `torch.cuda.empty_cache()` between stages). (Spec §3, §5.1)
- API server process (`api_server.py`, FastAPI/uvicorn) must never import `torch`/`faster_whisper`/model code directly — only `worker.py` (running in spawned subprocesses) touches models. (Spec §4)
- Worker processes use `multiprocessing.get_context("spawn")`, not `fork`. (Spec §4)
- Job state (`status.json`, `result.json`) lives on disk under `jobs/{job_id}/`, written atomically via a `.tmp` file + `os.replace()` — never written in place. (Spec §4, §9)
- API default `no_stem=true` (opposite of the CLI's `stemming=True` default). (Spec §6)
- Uploaded audio is saved using an extension from a fixed allowlist (`.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.mp4`, `.webm`) derived from the upload, never the raw client filename. (Spec §6, §9)
- The stemming subprocess call uses `subprocess.run([...])` (argument list, `shell=False`), never a shell string. (Spec §9)
- Output JSON `utterances` are grouped from `wsm` by **speaker change only** (not `ssm`'s sentence-split groups); speaker letters (`"A"`, `"B"`, ...) are assigned by order of first appearance in `wsm`, not by numeric offset from the raw speaker id; top-level `text` is the join of final (post-punctuation) `wsm` words; `audio_duration` is an integer number of seconds. (Spec §7)

---

## Task 1: Extract pipeline stage functions; refactor `diarize.py` to use them

**Files:**
- Create: `pipeline.py`
- Modify: `diarize.py`
- Test: `tests/test_diarize_cli.py`

**Interfaces:**
- Produces (used by Tasks 4 and 6):
  - `pipeline.mtypes: dict` (moved from `diarize.py`, unchanged: `{"cpu": "int8", "cuda": "float16"}`)
  - `pipeline.Models` — dataclass with fields `whisper_model`, `whisper_pipeline`, `alignment_model`, `alignment_tokenizer`, `diarizer_model`, `punct_model` (all `Optional`, default `None`), `device: str`
  - `pipeline.load_models(whisper_model_name: str, device: str, diarizer: Optional[str] = None, *, load_whisper: bool = True, load_alignment: bool = True, load_diarizer: bool = True, load_punct: bool = True) -> Models`
  - `pipeline.separate_vocals(audio_path: str, output_dir: str, device: str) -> str` (returns path to vocals, or `audio_path` on failure)
  - `pipeline.transcribe(whisper_model, whisper_pipeline, audio_waveform, *, language=None, suppress_numerals=False, batch_size=8) -> tuple[str, object]` (`full_transcript`, faster-whisper `info`)
  - `pipeline.align(alignment_model, alignment_tokenizer, audio_waveform, full_transcript: str, language: str, batch_size: int = 8) -> list` (`word_timestamps`)
  - `pipeline.diarize(diarizer_model, audio_waveform) -> list` (`speaker_ts`)
  - `pipeline.map_words(word_timestamps: list, speaker_ts: list) -> list` (`wsm`)
  - `pipeline.restore_punctuation(punct_model, wsm: list, language: str) -> list` (returns `wsm` unchanged if `language not in helpers.punct_model_langs`)
  - `pipeline.finalize_mappings(wsm: list, speaker_ts: list) -> tuple[list, list]` (`wsm`, `ssm`)

**Step 1: Read the current `diarize.py` end-to-end**

Already done in design; re-read `diarize.py:1-244` if you need to re-confirm exact behavior before extracting. Do not skip this — the extraction must be byte-for-byte behavior preserving.

- [ ] Confirm you've read the full current `diarize.py`.

**Step 2: Write the regression (characterization) test against the current CLI**

This is a refactor task, not new behavior — the test asserts today's `diarize.py` output shape so the refactor can be checked against it. Create `tests/test_diarize_cli.py`:

```python
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(REPO_ROOT, "tests", "assets", "test.opus")


@pytest.mark.timeout(600)
def test_diarize_cli_produces_txt_and_srt(tmp_path):
    audio_copy = tmp_path / "test.opus"
    audio_copy.write_bytes(open(AUDIO, "rb").read())

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "diarize.py"),
            "-a", str(audio_copy),
            "--whisper-model", "tiny.en",
            "--diarizer", "sortformer",
            "--device", "cpu",
            "--no-stem",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    txt_path = tmp_path / "test.txt"
    srt_path = tmp_path / "test.srt"
    assert txt_path.exists()
    assert srt_path.exists()

    txt_content = txt_path.read_text(encoding="utf-8-sig")
    assert txt_content.strip() != ""
    assert ":" in txt_content  # "Speaker N: ..." format
```

Note: this test downloads `tiny.en` + alignment + sortformer models on first run (same models the existing `.github/workflows/test_run.yml` `test-run` job already downloads) and can take several minutes. This matches the project's existing testing style (integration-level, no mocked models). The `@pytest.mark.timeout(...)` decorator used here and in later tasks needs `pytest-timeout` (added to `requirements-dev.txt` in Task 6) to actually enforce the timeout; without it pytest just emits an "unknown mark" warning and runs normally, so this test still works standalone before Task 6.

- [ ] Write the test file above.

**Step 3: Run it against the *current*, unrefactored `diarize.py` to confirm the baseline passes**

Run: `python -m pytest tests/test_diarize_cli.py -v`
Expected: PASS (this confirms your test is a correct characterization of current behavior before you touch `diarize.py`).

- [ ] Ran and confirmed PASS.

**Step 4: Create `pipeline.py` with the stage functions**

```python
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
)

mtypes = {"cpu": "int8", "cuda": "float16"}


@dataclass
class Models:
    device: str
    whisper_model: Optional[object] = None
    whisper_pipeline: Optional[object] = None
    alignment_model: Optional[object] = None
    alignment_tokenizer: Optional[object] = None
    diarizer_model: Optional[object] = None
    punct_model: Optional[object] = None


def load_models(
    whisper_model_name: str,
    device: str,
    diarizer: Optional[str] = None,
    *,
    load_whisper: bool = True,
    load_alignment: bool = True,
    load_diarizer: bool = True,
    load_punct: bool = True,
) -> Models:
    models = Models(device=device)

    if load_whisper:
        models.whisper_model = faster_whisper.WhisperModel(
            whisper_model_name, device=device, compute_type=mtypes[device]
        )
        models.whisper_pipeline = faster_whisper.BatchedInferencePipeline(models.whisper_model)

    if load_alignment:
        models.alignment_model, models.alignment_tokenizer = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    if load_diarizer:
        if diarizer == "msdd":
            from diarization import MSDDDiarizer

            models.diarizer_model = MSDDDiarizer(device=device)
        elif diarizer == "sortformer":
            from diarization import SortformerDiarizer

            models.diarizer_model = SortformerDiarizer(device=device)
        else:
            raise ValueError(f"Unknown diarizer: {diarizer!r}")

    if load_punct:
        models.punct_model = PunctuationModel(model="kredor/punctuate-all")

    return models


def separate_vocals(audio_path: str, output_dir: str, device: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems=vocals",
            audio_path,
            "-o",
            output_dir,
            "--device",
            device,
        ],
        check=False,
    )
    if result.returncode != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        return audio_path
    return os.path.join(
        output_dir,
        "htdemucs",
        os.path.splitext(os.path.basename(audio_path))[0],
        "vocals.wav",
    )


def transcribe(
    whisper_model,
    whisper_pipeline,
    audio_waveform,
    *,
    language: Optional[str] = None,
    suppress_numerals: bool = False,
    batch_size: int = 8,
):
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer) if suppress_numerals else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    return full_transcript, info


def align(alignment_model, alignment_tokenizer, audio_waveform, full_transcript, language, batch_size=8):
    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device),
        batch_size=batch_size,
    )

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[language],
    )

    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    return postprocess_results(text_starred, spans, stride, scores)


def diarize(diarizer_model, audio_waveform):
    return diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))


def map_words(word_timestamps, speaker_ts):
    return get_words_speaker_mapping(word_timestamps, speaker_ts, "start")


def restore_punctuation(punct_model, wsm, language):
    if language not in punct_model_langs:
        logging.warning(
            f"Punctuation restoration is not available for {language} language."
            " Using the original punctuation."
        )
        return wsm

    words_list = [w["word"] for w in wsm]
    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    return wsm


def finalize_mappings(wsm, speaker_ts):
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    return wsm, ssm
```

- [ ] Create `pipeline.py` with the content above.

**Step 5: Refactor `diarize.py` to call the stage functions, preserving the load→use→del→empty_cache sequence**

Replace `diarize.py` lines 1–37 imports/setup and 96–244 body (keep argparse block, lines 38–96, as-is) with:

```python
import logging
import os

import faster_whisper
import torch

import pipeline
from helpers import cleanup, langs_to_iso, process_language_arg, punct_model_langs

# ... (argparse block unchanged) ...

args = parser.parse_args()
language = process_language_arg(args.language, args.model_name)

temp_path = os.path.join(os.getcwd(), f"temp_outputs_{os.getpid()}")
os.makedirs(temp_path, exist_ok=True)

if args.stemming:
    vocal_target = pipeline.separate_vocals(args.audio, temp_path, args.device)
else:
    vocal_target = args.audio

# Transcribe
models = pipeline.load_models(
    args.model_name, args.device, load_alignment=False, load_diarizer=False, load_punct=False
)
audio_waveform = faster_whisper.decode_audio(vocal_target)
full_transcript, info = pipeline.transcribe(
    models.whisper_model,
    models.whisper_pipeline,
    audio_waveform,
    language=language,
    suppress_numerals=args.suppress_numerals,
    batch_size=args.batch_size,
)
del models
torch.cuda.empty_cache()

# Forced Alignment
models = pipeline.load_models(
    args.model_name, args.device, load_whisper=False, load_diarizer=False, load_punct=False
)
word_timestamps = pipeline.align(
    models.alignment_model,
    models.alignment_tokenizer,
    audio_waveform,
    full_transcript,
    info.language,
    args.batch_size,
)
del models
torch.cuda.empty_cache()

# Diarization
models = pipeline.load_models(
    args.model_name, args.device, args.diarizer, load_whisper=False, load_alignment=False, load_punct=False
)
speaker_ts = pipeline.diarize(models.diarizer_model, audio_waveform)
del models
torch.cuda.empty_cache()

wsm = pipeline.map_words(word_timestamps, speaker_ts)

if info.language in punct_model_langs:
    models = pipeline.load_models(
        args.model_name, args.device, load_whisper=False, load_alignment=False, load_diarizer=False
    )
    wsm = pipeline.restore_punctuation(models.punct_model, wsm, info.language)
    del models
else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

wsm, ssm = pipeline.finalize_mappings(wsm, speaker_ts)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
```

Keep `get_speaker_aware_transcript` and `write_srt` imported from `helpers` as today. Remove the now-unused imports (`ctc_forced_aligner.*`, `deepmultilingualpunctuation.PunctuationModel`, `find_numeral_symbol_tokens`, `get_realigned_ws_mapping_with_punctuation`, `get_sentences_speaker_mapping`, `get_words_speaker_mapping`) from `diarize.py` since they now live in `pipeline.py`. Keep `get_speaker_aware_transcript`, `write_srt`, `cleanup`, `langs_to_iso` (no longer needed directly — remove if unused), `process_language_arg`, `punct_model_langs` imports from `helpers`.

- [ ] Apply the refactor to `diarize.py`.

**Step 6: Run the regression test again to confirm the refactor didn't change behavior**

Run: `python -m pytest tests/test_diarize_cli.py -v`
Expected: PASS

- [ ] Ran and confirmed PASS.

**Step 7: Run ruff to confirm no lint issues from the refactor**

Run: `ruff check diarize.py pipeline.py`
Expected: no errors (fix any unused-import errors by removing them from `diarize.py`).

- [ ] Ran and confirmed clean.

**Step 8: Commit**

```bash
git add pipeline.py diarize.py tests/test_diarize_cli.py
git commit -m "$(cat <<'EOF'
Extract diarization pipeline into reusable stage functions

diarize.py keeps its exact load/use/delete/empty_cache sequence and
output; the stage functions are what api_server.py's worker will
reuse without deleting models between calls.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018SMjqPYv94DjiiCHtk6pbL
EOF
)"
```

- [ ] Committed.

---

## Task 2: Job store (on-disk job state, atomic writes)

**Files:**
- Create: `jobstore.py`
- Test: `tests/test_jobstore.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure filesystem + `json` + `uuid`).
- Produces (used by Tasks 4 and 5):
  - `jobstore.ALLOWED_AUDIO_EXTENSIONS: frozenset[str]` = `{".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".webm"}`
  - `jobstore.JobStore(jobs_dir: str)` class with:
    - `create_job(self, audio_bytes: bytes, extension: str, params: dict) -> str` — generates a `uuid4` hex job id, creates `jobs_dir/{job_id}/`, writes `audio{extension}`, writes `params.json` (the dict), writes initial `status.json = {"status": "queued"}`, returns `job_id`. Raises `ValueError` if `extension not in ALLOWED_AUDIO_EXTENSIONS`.
    - `audio_path(self, job_id: str) -> str` — path to the saved audio file (globs for `audio.*` under the job dir), raises `FileNotFoundError` if the job dir doesn't exist.
    - `params(self, job_id: str) -> dict` — reads `params.json`.
    - `get_status(self, job_id: str) -> dict` — reads `status.json`; raises `FileNotFoundError` if job dir doesn't exist.
    - `set_status(self, job_id: str, status: dict) -> None` — atomic write (`.tmp` + `os.replace`).
    - `set_result(self, job_id: str, result: dict) -> None` — atomic write of `result.json`.
    - `get_result(self, job_id: str) -> dict` — reads `result.json`.
    - `delete_job(self, job_id: str) -> None` — `shutil.rmtree(job_dir)`; raises `FileNotFoundError` if missing.
    - `job_exists(self, job_id: str) -> bool`

**Step 1: Write the failing tests**

Create `tests/test_jobstore.py`:

```python
import json
import os

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
```

- [ ] Write the test file above.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_jobstore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jobstore'`

- [ ] Ran and confirmed FAIL.

**Step 3: Implement `jobstore.py`**

```python
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
```

- [ ] Implement `jobstore.py` as above.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_jobstore.py -v`
Expected: PASS (all tests)

- [ ] Ran and confirmed PASS.

**Step 5: Commit**

```bash
git add jobstore.py tests/test_jobstore.py
git commit -m "$(cat <<'EOF'
Add JobStore for on-disk job state with atomic writes

status.json/result.json are written via tmp-file + os.replace so a
GET poll landing mid-write never sees truncated JSON.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018SMjqPYv94DjiiCHtk6pbL
EOF
)"
```

- [ ] Committed.

---

## Task 3: Result schema builder (AssemblyAI-shaped JSON)

**Files:**
- Create: `schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure data transformation over `wsm`-shaped lists: `[{"word": str, "start_time": int, "end_time": int, "speaker": int}, ...]`).
- Produces (used by Task 4):
  - `schema.build_result(job_id: str, language: str, audio_duration_seconds: int, wsm: list[dict]) -> dict` returning the shape from spec §7: `{"id", "status": "completed", "language_code", "audio_duration", "text", "utterances": [{"speaker", "start", "end", "text", "words": [{"text", "start", "end", "speaker"}]}]}`.

**Step 1: Write the failing tests**

Create `tests/test_schema.py`:

```python
from schema import build_result


def _wsm(*rows):
    return [
        {"word": word, "start_time": start, "end_time": end, "speaker": speaker}
        for word, start, end, speaker in rows
    ]


def test_build_result_top_level_fields():
    wsm = _wsm(("Hello", 0, 200, 0), ("world.", 200, 500, 0))
    result = build_result("job-1", "en", 1, wsm)

    assert result["id"] == "job-1"
    assert result["status"] == "completed"
    assert result["language_code"] == "en"
    assert result["audio_duration"] == 1
    assert result["text"] == "Hello world."


def test_single_speaker_produces_one_utterance():
    wsm = _wsm(("Hello", 0, 200, 0), ("world.", 200, 500, 0))
    result = build_result("job-1", "en", 1, wsm)

    assert len(result["utterances"]) == 1
    utt = result["utterances"][0]
    assert utt["speaker"] == "A"
    assert utt["start"] == 0
    assert utt["end"] == 500
    assert utt["text"] == "Hello world."
    assert utt["words"] == [
        {"text": "Hello", "start": 0, "end": 200, "speaker": "A"},
        {"text": "world.", "start": 200, "end": 500, "speaker": "A"},
    ]


def test_speaker_change_creates_new_utterance_even_mid_sentence():
    # AssemblyAI groups by speaker turn only, not by sentence boundary.
    wsm = _wsm(
        ("Hello", 0, 200, 0),
        ("there,", 200, 400, 0),
        ("hi", 400, 600, 1),
        ("back", 600, 800, 1),
        ("and more", 800, 1000, 0),
    )
    result = build_result("job-1", "en", 1, wsm)

    assert len(result["utterances"]) == 3
    assert [u["speaker"] for u in result["utterances"]] == ["A", "B", "A"]
    assert result["utterances"][0]["text"] == "Hello there,"
    assert result["utterances"][1]["text"] == "hi back"
    assert result["utterances"][2]["text"] == "and more"


def test_speaker_letters_assigned_by_first_appearance_not_numeric_id():
    # Raw diarizer ids can start at a nonzero, non-contiguous value.
    wsm = _wsm(("Hi", 0, 100, 7), ("there", 100, 200, 3), ("again", 200, 300, 7))
    result = build_result("job-1", "en", 1, wsm)

    assert [u["speaker"] for u in result["utterances"]] == ["A", "B", "A"]


def test_empty_wsm_produces_empty_text_and_no_utterances():
    result = build_result("job-1", "en", 0, [])
    assert result["text"] == ""
    assert result["utterances"] == []
```

- [ ] Write the test file above.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'schema'`

- [ ] Ran and confirmed FAIL.

**Step 3: Implement `schema.py`**

```python
def _speaker_letters(wsm):
    letters = {}
    for word in wsm:
        speaker_id = word["speaker"]
        if speaker_id not in letters:
            letters[speaker_id] = chr(ord("A") + len(letters))
    return letters


def build_result(job_id: str, language: str, audio_duration_seconds: int, wsm: list) -> dict:
    letters = _speaker_letters(wsm)

    utterances = []
    current = None
    for word in wsm:
        speaker = letters[word["speaker"]]
        if current is None or speaker != current["speaker"]:
            if current is not None:
                utterances.append(current)
            current = {
                "speaker": speaker,
                "start": word["start_time"],
                "end": word["end_time"],
                "words": [],
            }
        current["end"] = word["end_time"]
        current["words"].append(
            {
                "text": word["word"],
                "start": word["start_time"],
                "end": word["end_time"],
                "speaker": speaker,
            }
        )
    if current is not None:
        utterances.append(current)

    for utt in utterances:
        utt["text"] = " ".join(w["text"] for w in utt["words"])

    return {
        "id": job_id,
        "status": "completed",
        "language_code": language,
        "audio_duration": audio_duration_seconds,
        "text": " ".join(w["word"] for w in wsm),
        "utterances": utterances,
    }
```

- [ ] Implement `schema.py` as above.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schema.py -v`
Expected: PASS (all tests)

- [ ] Ran and confirmed PASS.

**Step 5: Commit**

```bash
git add schema.py tests/test_schema.py
git commit -m "$(cat <<'EOF'
Add AssemblyAI-shaped result schema builder

Groups utterances by speaker turn (matching AssemblyAI's own
utterance/word-run relationship), not by sentence boundary.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018SMjqPYv94DjiiCHtk6pbL
EOF
)"
```

- [ ] Committed.

---

## Task 4: Worker process (loads models once, drains the job queue)

**Files:**
- Create: `worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes:
  - `pipeline.load_models(...)`, `pipeline.separate_vocals`, `pipeline.transcribe`, `pipeline.align`, `pipeline.diarize`, `pipeline.map_words`, `pipeline.restore_punctuation`, `pipeline.finalize_mappings` (Task 1)
  - `jobstore.JobStore` (Task 2)
  - `schema.build_result` (Task 3)
- Produces (used by Task 5):
  - `worker.process_one_job(models: pipeline.Models, store: jobstore.JobStore, job_id: str, *, temp_dir: str) -> None` — runs the full pipeline for one job and writes `status.json`/`result.json` via `store`. Catches all exceptions internally and writes a `{"status": "failed", "error": str(e)}` status instead of raising.
  - `worker.worker_loop(job_queue, jobs_dir: str, whisper_model_name: str, device: str, diarizer: str) -> None` — calls `pipeline.load_models(whisper_model_name, device, diarizer)` once (all four model groups), builds a `jobstore.JobStore(jobs_dir)`, then loops: `job_id = job_queue.get()`; if `job_id is None`, return (sentinel to stop the loop, used by tests and graceful shutdown); else call `process_one_job(...)`.

**Step 1: Write the failing tests**

Create `tests/test_worker.py`. These tests never load real models — `process_one_job` is tested with a fake `pipeline` module via monkeypatching, so they run in milliseconds:

```python
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
        lambda *a, **k: [{"word": "hello", "start_time": 0, "end_time": 250, "speaker": 0},
                          {"word": "world", "start_time": 250, "end_time": 500, "speaker": 0}],
    )
    monkeypatch.setattr(worker.pipeline, "restore_punctuation", lambda punct_model, wsm, lang: wsm)


def test_process_one_job_writes_completed_status_and_result(monkeypatch, store, tmp_path):
    _patch_happy_path(monkeypatch, tmp_path)
    job_id = store.create_job(b"fake-audio", ".wav", {"language": None, "no_stem": True,
                                                        "suppress_numerals": False, "batch_size": 8})

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
    job_id = store.create_job(b"fake-audio", ".wav", {"language": None, "no_stem": True,
                                                        "suppress_numerals": False, "batch_size": 8})

    worker.process_one_job(FakeModels(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    status = store.get_status(job_id)
    assert status["status"] == "failed"
    assert "transcription exploded" in status["error"]


def test_process_one_job_respects_no_stem_false(monkeypatch, store, tmp_path):
    _patch_happy_path(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        worker.pipeline, "separate_vocals",
        lambda audio, out, dev: calls.append(True) or audio,
    )
    job_id = store.create_job(b"fake-audio", ".wav", {"language": None, "no_stem": False,
                                                        "suppress_numerals": False, "batch_size": 8})

    worker.process_one_job(FakeModels(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    assert calls == [True]


def test_worker_loop_stops_on_none_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(worker.pipeline, "load_models", lambda *a, **k: FakeModels())
    q = queue.Queue()
    q.put(None)

    # Must return, not hang, when it sees the sentinel.
    worker.worker_loop(q, str(tmp_path), "tiny.en", "cpu", "sortformer")


def test_worker_loop_processes_queued_job_then_stops(monkeypatch, tmp_path):
    _patch_happy_path(monkeypatch, tmp_path)
    monkeypatch.setattr(worker.pipeline, "load_models", lambda *a, **k: FakeModels())

    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"fake-audio", ".wav", {"language": None, "no_stem": True,
                                                        "suppress_numerals": False, "batch_size": 8})

    q = queue.Queue()
    q.put(job_id)
    q.put(None)

    worker.worker_loop(q, str(tmp_path), "tiny.en", "cpu", "sortformer")

    assert store.get_status(job_id) == {"status": "completed"}
```

- [ ] Write the test file above.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'worker'`

- [ ] Ran and confirmed FAIL.

**Step 3: Implement `worker.py`**

```python
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


def worker_loop(job_queue, jobs_dir: str, whisper_model_name: str, device: str, diarizer: str) -> None:
    models = pipeline.load_models(whisper_model_name, device, diarizer)
    store = JobStore(jobs_dir)
    temp_dir = os.path.join(jobs_dir, f"_worker_temp_{os.getpid()}")

    while True:
        job_id = job_queue.get()
        if job_id is None:
            return
        process_one_job(models, store, job_id, temp_dir=temp_dir)
```

- [ ] Implement `worker.py` as above.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_worker.py -v`
Expected: PASS (all tests)

- [ ] Ran and confirmed PASS.

**Step 5: Commit**

```bash
git add worker.py tests/test_worker.py
git commit -m "$(cat <<'EOF'
Add worker loop that loads models once and drains the job queue

process_one_job never raises - all failures are captured into
status.json so one bad file can't kill the worker process.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018SMjqPYv94DjiiCHtk6pbL
EOF
)"
```

- [ ] Committed.

---

## Task 5: FastAPI app (`/jobs` endpoints) with an injectable queue for tests

**Files:**
- Create: `api_server.py`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Test: `tests/test_api_server.py`

**Interfaces:**
- Consumes: `jobstore.JobStore` (Task 2), `worker.worker_loop` (Task 4, used only in `main()`, not in the tests for this task).
- Produces (used by Task 6):
  - `api_server.create_app(store: jobstore.JobStore, job_queue) -> fastapi.FastAPI` — `job_queue` only needs a `.put(item)` method, so tests can pass a plain `queue.Queue()` instead of a real `multiprocessing.Queue`.
  - `api_server.main() -> None` — parses `--host`, `--port`, `--max-parallel` (default `1`), `--whisper-model`, `--device`, `--diarizer`, `--jobs-dir` (default `./jobs`); creates the jobs dir; creates a `multiprocessing.get_context("spawn").Queue()`; starts `--max-parallel` `worker.worker_loop` processes; builds the app via `create_app`; runs `uvicorn.run(app, host=..., port=...)`.

**Step 1: Write the failing tests**

Create `tests/test_api_server.py`. These use FastAPI's `TestClient` and a plain `queue.Queue()` stand-in — no real models or worker processes involved:

```python
import queue

from fastapi.testclient import TestClient

from api_server import create_app
from jobstore import JobStore


def _client(tmp_path):
    store = JobStore(str(tmp_path))
    job_queue = queue.Queue()
    app = create_app(store, job_queue)
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
```

- [ ] Write the test file above.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api_server'`

- [ ] Ran and confirmed FAIL.

**Step 3: Add dependencies**

Append to `requirements.txt`:

```
fastapi
uvicorn[standard]
python-multipart
```

Create `requirements-dev.txt`:

```
-r requirements.txt
pytest
httpx
```

Run: `pip install -r requirements-dev.txt`

- [ ] Updated `requirements.txt`, created `requirements-dev.txt`, installed.

**Step 4: Implement `api_server.py`**

```python
import argparse
import multiprocessing
import os

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from jobstore import ALLOWED_AUDIO_EXTENSIONS, JobStore
from worker import worker_loop


def create_app(store: JobStore, job_queue) -> FastAPI:
    app = FastAPI()

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
    args = parser.parse_args()

    store = JobStore(args.jobs_dir)
    ctx = multiprocessing.get_context("spawn")
    job_queue = ctx.Queue()

    workers = []
    for _ in range(args.max_parallel):
        p = ctx.Process(
            target=worker_loop,
            args=(job_queue, args.jobs_dir, args.whisper_model, args.device, args.diarizer),
            daemon=True,
        )
        p.start()
        workers.append(p)

    app = create_app(store, job_queue)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] Implement `api_server.py` as above.

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_server.py -v`
Expected: PASS (all tests)

- [ ] Ran and confirmed PASS.

**Step 6: Run ruff**

Run: `ruff check api_server.py worker.py jobstore.py schema.py pipeline.py`
Expected: no errors.

- [ ] Ran and confirmed clean.

**Step 7: Commit**

```bash
git add api_server.py requirements.txt requirements-dev.txt tests/test_api_server.py
git commit -m "$(cat <<'EOF'
Add FastAPI job submission/status/delete endpoints

create_app takes an injectable queue so route logic is tested without
spawning real worker processes or loading any models.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018SMjqPYv94DjiiCHtk6pbL
EOF
)"
```

- [ ] Committed.

---

## Task 6: End-to-end test, docs, CI wiring

**Files:**
- Create: `tests/test_api_e2e.py`
- Modify: `.github/workflows/test_run.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `api_server.py` (Task 5) run as a real subprocess (real models, real worker process, real HTTP).

**Step 1: Write the end-to-end test**

This is a genuine integration test — the first one that loads real models and runs the real server as a subprocess. No prior "failing" step applies (there is nothing to make pass incrementally; the whole stack must exist first), so write it directly:

```python
import json
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
```

Add `requests` to `requirements-dev.txt` (append `requests` and `pytest-timeout`, which the `@pytest.mark.timeout` decorators used across this plan's tests require):

```
-r requirements.txt
pytest
pytest-timeout
httpx
requests
```

- [ ] Write `tests/test_api_e2e.py` as above and update `requirements-dev.txt`.

**Step 2: Run it**

Run: `python -m pytest tests/test_api_e2e.py -v`
Expected: PASS (downloads `tiny.en` + alignment + sortformer models on first run, same as the existing CLI CI job; takes several minutes)

- [ ] Ran and confirmed PASS.

**Step 3: Wire into CI**

Modify `.github/workflows/test_run.yml`, in the `test-run` job, after the existing "Install dependencies" step and before the "Test with sortformer diarizer" step, add:

```yaml
    - name: Install dev dependencies
      run: |
        uv pip install --system -r requirements-dev.txt

    - name: Test API server (end-to-end)
      run: |
        python -m pytest tests/test_api_e2e.py tests/test_api_server.py tests/test_worker.py tests/test_schema.py tests/test_jobstore.py -v
```

- [ ] Edit the workflow file as above.

**Step 4: Update `README.md`**

Add a new `## API Server` section after the existing `## Command Line Options` section:

```markdown
## API Server

For programmatic / concurrent use, run the HTTP API server instead of the CLI:

```shell
python api_server.py --max-parallel 1 --whisper-model medium.en --device cuda --diarizer msdd
```

`--max-parallel` controls how many audio files are transcribed at the same time; each
unit of parallelism keeps its own copy of every model resident in memory, so raise it
only if you have the VRAM for N copies of your chosen Whisper model + the alignment,
diarization, and punctuation models. Extra requests beyond `--max-parallel` wait in a
FIFO queue and are picked up as capacity frees up.

Submit a file:

```shell
curl -X POST http://localhost:8000/jobs -F file=@audio.wav
# {"job_id": "...", "status": "queued"}
```

Poll for the result:

```shell
curl http://localhost:8000/jobs/<job_id>
# {"job_id": "...", "status": "completed", "result": {"text": "...", "utterances": [...]}}
```

Delete a job's files once you're done with it:

```shell
curl -X DELETE http://localhost:8000/jobs/<job_id>
```
```

- [ ] Update `README.md` as above.

**Step 5: Commit**

```bash
git add tests/test_api_e2e.py requirements-dev.txt .github/workflows/test_run.yml README.md
git commit -m "$(cat <<'EOF'
Add end-to-end API server test, wire into CI, document usage

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018SMjqPYv94DjiiCHtk6pbL
EOF
)"
```

- [ ] Committed.
