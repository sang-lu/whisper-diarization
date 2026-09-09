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
    wsm = _wsm(("Hi", 0, 100, 7), ("there", 100, 200, 3), ("again", 200, 300, 7))
    result = build_result("job-1", "en", 1, wsm)

    assert [u["speaker"] for u in result["utterances"]] == ["A", "B", "A"]


def test_empty_wsm_produces_empty_text_and_no_utterances():
    result = build_result("job-1", "en", 0, [])
    assert result["text"] == ""
    assert result["utterances"] == []
