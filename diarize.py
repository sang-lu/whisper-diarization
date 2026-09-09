import argparse
import logging
import os

import faster_whisper
import torch

import pipeline
from helpers import (
    cleanup,
    get_speaker_aware_transcript,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--audio", help="name of the target audio file", required=True)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation.This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, "
    "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise use 'cpu'",
)

parser.add_argument(
    "--diarizer",
    default="msdd",
    choices=["msdd", "sortformer"],
    help="Choose the diarization model to use",
)

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
    args.model_name,
    args.device,
    load_alignment=False,
    load_diarizer=False,
    load_punct=False,
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
    args.model_name,
    args.device,
    load_whisper=False,
    load_diarizer=False,
    load_punct=False,
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
    args.model_name,
    args.device,
    args.diarizer,
    load_whisper=False,
    load_alignment=False,
    load_punct=False,
)
speaker_ts = pipeline.diarize(models.diarizer_model, audio_waveform)
del models
torch.cuda.empty_cache()

wsm = pipeline.map_words(word_timestamps, speaker_ts)

if info.language in punct_model_langs:
    models = pipeline.load_models(
        args.model_name,
        args.device,
        load_whisper=False,
        load_alignment=False,
        load_diarizer=False,
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
