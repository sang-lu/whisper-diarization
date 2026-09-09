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


def align(
    alignment_model, alignment_tokenizer, audio_waveform, full_transcript, language, batch_size=8
):
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
    labeled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labeled_words):
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
