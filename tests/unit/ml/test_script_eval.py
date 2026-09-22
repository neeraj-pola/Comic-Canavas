"""The pure, network-free half of ml/evals/script.py —
`load_script_goldens`, `constraint_compliance`, `mean_caption_length`.
`run_eval`/`main`/`judge_pairwise` need real `LLM_SCRIPT`/`LLM_JUDGE`
providers and keys; not exercised by `make test`.
"""

from __future__ import annotations

from contracts import Beat, BeatSheet, Panel, Script
from ml.evals.script import (
    GOLDENS_DIR,
    constraint_compliance,
    load_script_goldens,
    mean_caption_length,
)


def test_all_15_script_goldens_load_and_validate() -> None:
    goldens = load_script_goldens()
    assert len(goldens) == 15
    for g in goldens:
        assert isinstance(g.beat_sheet, BeatSheet)
        assert len(g.reference_scripts) == 2
        assert all(isinstance(s, Script) for s in g.reference_scripts)


def test_golden_ids_are_unique() -> None:
    goldens = load_script_goldens()
    ids = [g.id for g in goldens]
    assert len(ids) == len(set(ids))


def test_reference_scripts_satisfy_constraint_compliance() -> None:
    # The frozen references are meant to exemplify a fully-compliant
    # script — if they don't score 1.0 here, either a golden or the
    # compliance check itself has drifted.
    for golden in load_script_goldens():
        for script in golden.reference_scripts:
            check = constraint_compliance(golden.beat_sheet, script)
            assert check.score == 1.0, (golden.id, check)


def _beat_sheet(**beat_overrides: object) -> BeatSheet:
    defaults: dict[str, object] = {
        "id": "b1",
        "time": "evening",
        "place": "kitchen",
        "event": "Cooked dinner",
        "emotion": "content",
        "people": ["Sam"],
        "importance": 0.5,
        "humor": 0.1,
    }
    defaults.update(beat_overrides)
    return BeatSheet(
        date="2026-01-01", mood_arc=["content"], beats=[Beat(**defaults)], people_mentioned=["Sam"]
    )


def _panel(**overrides: object) -> Panel:
    defaults: dict[str, object] = {
        "id": 1,
        "beat_id": "b1",
        "place": "kitchen",
        "time_of_day": "evening",
        "expression": "content",
        "action": "Cooking at the stove",
        "framing": "wide",
        "caption_a": "Dinner's on",
        "caption_b": "Cooking tonight",
        "cast": ["Sam"],
    }
    defaults.update(overrides)
    return Panel(**defaults)


def test_constraint_compliance_flags_bad_beat_id() -> None:
    beat_sheet = _beat_sheet()
    script = Script(
        mood="content",
        quiet_day=False,
        panels=[
            _panel(beat_id="nonexistent", framing="wide"),
            _panel(id=2, beat_id="nonexistent", framing="close"),
        ],
    )
    check = constraint_compliance(beat_sheet, script)
    assert check.beat_ids_valid is False


def test_constraint_compliance_flags_cast_mismatch() -> None:
    beat_sheet = _beat_sheet(people=["Sam"])
    script = Script(
        mood="content",
        quiet_day=False,
        panels=[
            _panel(cast=["Someone Else"], framing="wide"),
            _panel(id=2, framing="close", cast=["Someone Else"]),
        ],
    )
    check = constraint_compliance(beat_sheet, script)
    assert check.cast_matches_beats is False


def test_constraint_compliance_flags_missing_framing_diversity() -> None:
    beat_sheet = _beat_sheet()
    script = Script(
        mood="content",
        quiet_day=False,
        panels=[
            _panel(framing="medium"),
            _panel(id=2, framing="medium"),
        ],
    )
    check = constraint_compliance(beat_sheet, script)
    assert check.framing_diverse is False


def test_mean_caption_length() -> None:
    script = Script(
        mood="content",
        quiet_day=False,
        panels=[
            _panel(caption_a="12345", caption_b="1234567890", framing="wide"),
            _panel(id=2, caption_a="12", caption_b="1234", framing="close"),
        ],
    )
    # (5 + 10 + 2 + 4) / 4 = 5.25
    assert mean_caption_length(script) == 5.25


def test_goldens_dir_points_at_the_real_directory() -> None:
    assert GOLDENS_DIR.name == "scripts"
    assert GOLDENS_DIR.exists()
