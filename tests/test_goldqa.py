from collections import defaultdict

import pytest

from goldqa.drift import track
from goldqa.score import error_rate, rubric_violations, score
from goldqa.simulate import ANNOTATORS, generate


@pytest.fixture(scope="module")
def reports():
    sc = defaultdict(list)
    for r in generate():
        sc[(r["annotator"], r["week"])].append(score(r["item"], r["locale"], r["gold"], r["hyp"], r["duration"]))
    return {r.annotator: r for r in track(sc, {a: v[0] for a, v in ANNOTATORS.items()})}


def tok(*ws, start=0, step=300):
    return [{"w": w, "s": start + i * step, "e": start + i * step + 200} for i, w in enumerate(ws)]


@pytest.mark.parametrize("ann,metric,week", [
    ("A03", "filler", 5), ("A05", "ts", 3), ("A06", "rubric", 4), ("A06", "tags", 4), ("A10", "err", 6)])
def test_planted_drift_flagged_in_right_week(reports, ann, metric, week):
    assert (week, metric) in [(w, k) for w, k, _ in reports[ann].flags]


def test_clean_annotators_not_flagged(reports):
    for a, (_, beh) in ANNOTATORS.items():
        if beh in ("clean", "cjk_segmentation"):
            assert reports[a].flags == [], (a, reports[a].flags)


def test_cjk_scored_on_characters(reports):
    assert reports["A07"].wer_only_would_flag and reports["A07"].status == "OK"


def test_error_rate():
    assert error_rate(list("abcd"), list("abxd")) == 0.25
    assert error_rate(["a", "b"], ["a", "b"]) == 0


def test_rubric_catches_variants_case_and_timestamps():
    v = rubric_violations([{"w": "(pause)", "s": 0, "e": 10}, {"w": "Hola", "s": 20, "e": 30},
                           {"w": "x", "s": 5, "e": 9}, {"w": "y", "s": None, "e": None}], 1000)
    assert any("tag spelled" in x for x in v) and any("uppercase" in x for x in v)
    assert any("bad timestamp" in x for x in v) and any("no timestamp" in x for x in v)


def test_timestamp_offset_rejects_item():
    gold = tok("quiero", "cambiar", "mi", "pedido")
    hyp = tok("quiero", "cambiar", "mi", "pedido", start=120)
    s = score("i", "es", gold, hyp, 5000)
    assert s.ts_median_ms == 120 and not s.approved and "timestamps" in s.why_rejected[0]


def test_filler_dropping_rejects_item():
    gold = tok("eh", "quiero", "este", "cambiar", "mi", "pedido", "pues", "ya")
    hyp = tok("quiero", "cambiar", "mi", "pedido", "ya")
    s = score("i", "es", gold, hyp, 5000)
    assert s.filler_recall == 0 and not s.approved


def test_perfect_item_approved():
    g = tok("ich", "möchte", "[pause]", "ändern")
    assert score("i", "de", g, [dict(t) for t in g], 5000).approved
