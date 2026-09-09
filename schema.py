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
