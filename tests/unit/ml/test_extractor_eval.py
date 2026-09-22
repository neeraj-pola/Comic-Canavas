"""The pure, network-free half of ml/evals/extractor.py — `load_goldens`
and `score_case`. `run_eval`/`main` need a real LLM_EXTRACTOR provider
and key (see that module's docstring); not exercised by `make test`.
"""

from __future__ import annotations

from contracts import Beat, BeatSheet
from ml.evals.extractor import GOLDENS_DIR, GoldenCase, load_goldens, score_case


def test_all_30_goldens_load_and_validate() -> None:
    goldens = load_goldens()
    assert len(goldens) == 30
    assert all(isinstance(g.expected, BeatSheet) for g in goldens)


def test_golden_ids_are_unique() -> None:
    goldens = load_goldens()
    ids = [g.id for g in goldens]
    assert len(ids) == len(set(ids))


def test_category_breakdown_matches_task_2_3() -> None:
    goldens = load_goldens()
    counts: dict[str, int] = {}
    for g in goldens:
        counts[g.category] = counts.get(g.category, 0) + 1
    assert counts == {"real": 10, "synthetic": 10, "quiet": 4, "sensitive": 3, "jargon": 3}


def test_sensitive_category_flags_sensitive_and_nothing_else_expects_it() -> None:
    goldens = load_goldens()
    sensitive_expected = {g.id for g in goldens if "sensitive" in g.expected.flags}
    sensitive_category = {g.id for g in goldens if g.category == "sensitive"}
    assert sensitive_expected == sensitive_category
    assert len(sensitive_category) == 3


def test_n8n_whisper_correction_golden_is_present() -> None:
    # A whisper-error tolerance example: "an eight n" -> "n8n".
    goldens = load_goldens()
    golden = next(g for g in goldens if g.id == "028_jargon_n8n_workflow")
    assert "n8n" in golden.known_vocab
    assert "an eight n" in golden.transcript
    assert any("n8n" in b.event for b in golden.expected.beats)


def _golden(**overrides: object) -> GoldenCase:
    defaults: dict[str, object] = {
        "id": "synthetic",
        "category": "synthetic",
        "known_people": [],
        "known_places": [],
        "known_vocab": [],
        "transcript": "Cooked dinner and watched the sunset from the kitchen window.",
        "expected": BeatSheet(
            date="2026-01-01",
            mood_arc=["content"],
            beats=[
                Beat(
                    id="b1",
                    time="evening",
                    place="kitchen",
                    event="Cooked dinner and watched the sunset",
                    emotion="content",
                    importance=0.5,
                    humor=0.1,
                )
            ],
            people_mentioned=[],
        ),
    }
    defaults.update(overrides)
    return GoldenCase(**defaults)  # type: ignore[arg-type]


def test_score_case_perfect_match_has_recall_1_and_no_hallucinations() -> None:
    golden = _golden()
    predicted = golden.expected.model_copy()
    score = score_case(golden, predicted)
    assert score.recall == 1.0
    assert score.hallucinations == 0


def test_score_case_paraphrased_event_still_counts_as_recalled() -> None:
    golden = _golden()
    predicted = golden.expected.model_copy(
        update={
            "beats": [
                golden.expected.beats[0].model_copy(
                    update={"event": "Cooked dinner and watched the sunset outside"}
                )
            ]
        }
    )
    score = score_case(golden, predicted)
    assert score.recall == 1.0


def test_score_case_missing_beat_lowers_recall() -> None:
    golden = _golden(
        expected=BeatSheet(
            date="2026-01-01",
            mood_arc=["content", "tired"],
            beats=[
                *_golden().expected.beats,
                Beat(
                    id="b2",
                    time="night",
                    place="bedroom",
                    event="Went to sleep early feeling tired",
                    emotion="tired",
                    importance=0.2,
                    humor=0.0,
                ),
            ],
            people_mentioned=[],
        )
    )
    predicted = BeatSheet(
        date="2026-01-01",
        mood_arc=["content"],
        beats=[golden.expected.beats[0]],
        people_mentioned=[],
    )
    score = score_case(golden, predicted)
    assert score.recall == 0.5


def test_score_case_unrelated_predicted_beat_is_a_hallucination() -> None:
    golden = _golden()
    predicted = golden.expected.model_copy(
        update={
            "beats": [
                *golden.expected.beats,
                Beat(
                    id="b2",
                    time="morning",
                    place="gym",
                    event="Ran a marathon before breakfast",
                    emotion="proud",
                    importance=0.9,
                    humor=0.0,
                ),
            ]
        }
    )
    score = score_case(golden, predicted)
    assert score.recall == 1.0
    assert score.hallucinations == 1


def test_score_case_splitting_a_merged_beat_is_not_a_hallucination() -> None:
    # A golden may deliberately merge several sub-events into one terse
    # label; a model that instead reports some of those sub-events as
    # their own beats, each naming details the terse label dropped but
    # the transcript still has, must not be penalized for it — see
    # extractor.py's score_case
    # docstring-comment for why hallucinations are checked against the
    # transcript, not the (possibly abbreviated) expected beats.
    golden = _golden(
        transcript=(
            "Dropped off dry cleaning, picked up a prescription, returned a "
            "package at the post office, and got gas. Just a day of errands."
        ),
        expected=BeatSheet(
            date="2026-01-01",
            mood_arc=["content"],
            beats=[
                Beat(
                    id="b1",
                    time="afternoon",
                    place="other",
                    event="Ran a day of small errands: dry cleaning, a prescription, package, gas",
                    emotion="content",
                    importance=0.1,
                    humor=0.0,
                )
            ],
            people_mentioned=[],
        ),
    )
    predicted = BeatSheet(
        date="2026-01-01",
        mood_arc=["content"],
        beats=[
            Beat(
                id="b1",
                time="afternoon",
                place="other",
                event="Returned a package at the post office",
                emotion="content",
                importance=0.1,
                humor=0.0,
            ),
            Beat(
                id="b2",
                time="afternoon",
                place="other",
                event="Got gas",
                emotion="content",
                importance=0.1,
                humor=0.0,
            ),
        ],
        people_mentioned=[],
    )

    score = score_case(golden, predicted)

    assert score.hallucinations == 0


def test_score_case_name_normalization() -> None:
    golden = _golden(
        known_people=["Rebecca"],
        expected=BeatSheet(
            date="2026-01-01",
            mood_arc=["content"],
            beats=[
                Beat(
                    id="b1",
                    time="evening",
                    place="kitchen",
                    event="Had dinner with Rebecca",
                    emotion="content",
                    people=["Rebecca"],
                    importance=0.5,
                    humor=0.0,
                )
            ],
            people_mentioned=["Rebecca"],
        ),
    )

    correctly_normalized = golden.expected.model_copy()
    score = score_case(golden, correctly_normalized)
    assert score.name_normalization_correct == 1
    assert score.name_normalization_total == 1

    not_normalized = golden.expected.model_copy(
        update={
            "beats": [golden.expected.beats[0].model_copy(update={"people": ["Bex"]})],
            "people_mentioned": ["Bex"],
        }
    )
    score = score_case(golden, not_normalized)
    assert score.name_normalization_correct == 0
    assert score.name_normalization_total == 1


def test_goldens_dir_points_at_the_real_directory() -> None:
    assert GOLDENS_DIR.name == "transcripts"
    assert GOLDENS_DIR.exists()
