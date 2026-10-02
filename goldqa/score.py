"""Rubric-aware scoring of one transcript against its gold version.

A transcript is a list of tokens {"w": str, "s": start_ms, "e": end_ms}. Tags are tokens like "[pause]".
Rubric (from perit.ai): lowercase, fillers kept, one written convention for every tag, a timestamp on every word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TAGS = {"[pause]", "[overlap]", "[keyboard]", "[noise]"}
CJK = {"zh", "ja", "ko"}               # score with CER: these scripts don't put spaces between words
FILLERS = {
    "de": {"äh", "ähm", "also"}, "fr": {"euh", "ben", "bah"}, "it": {"ehm", "cioè", "allora"},
    "es": {"eh", "este", "pues"}, "pt": {"é", "tipo", "né"}, "hi": {"मतलब", "अच्छा", "हाँ"},
    "zh": {"那个", "嗯", "就是"}, "ja": {"えーと", "あの", "まあ"}, "ko": {"음", "그", "어"},
}
TAG_LIKE = re.compile(r"^[\[\(<{].*[\]\)>}]$")
TS_OK_MS = 50


def edit_align(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment -> list of (i, j) pairs; None marks insertion/deletion."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    pairs = edit_align(ref, hyp)
    errs = sum(1 for i, j in pairs if i is None or j is None or ref[i] != hyp[j])
    return errs / len(ref)


def rubric_violations(tokens: list[dict], duration_ms: int) -> list[str]:
    v, last = [], -1
    for t in tokens:
        w = t["w"]
        if TAG_LIKE.match(w) and w not in TAGS:
            v.append(f"tag spelled '{w}' (allowed: {', '.join(sorted(TAGS))})")
        elif w != w.lower():
            v.append(f"uppercase '{w}'")
        if t.get("s") is None or t.get("e") is None:
            v.append(f"no timestamp on '{w}'")
            continue
        if t["s"] < last or t["e"] < t["s"] or t["e"] > duration_ms:
            v.append(f"bad timestamp on '{w}' ({t['s']}–{t['e']} ms)")
        last = t["s"]
    return v


@dataclass
class ItemScore:
    item: str
    locale: str
    err: float                 # CER for zh/ja/ko, WER otherwise
    wer: float                 # always reported, to show the CJK trap
    filler_recall: float | None
    tag_f1: float | None
    ts_median_ms: float | None
    ts_within: float | None    # share of matched words within 50 ms
    violations: list[str] = field(default_factory=list)
    approved: bool = True
    why_rejected: list[str] = field(default_factory=list)


def _words(tokens):
    return [t for t in tokens if not TAG_LIKE.match(t["w"])]


def score(item_id: str, locale: str, gold: list[dict], hyp: list[dict], duration_ms: int,
          max_err: float = 0.05, max_ts_ms: float = TS_OK_MS, min_filler: float = 0.9) -> ItemScore:
    gw, hw = _words(gold), _words(hyp)
    g, h = [t["w"].lower() for t in gw], [t["w"].lower() for t in hw]
    wer = error_rate(g, h)
    err = error_rate(list("".join(g)), list("".join(h))) if locale in CJK else wer

    # timestamps on words both sides agree on
    diffs = [abs(gw[i]["s"] - hw[j]["s"]) for i, j in edit_align(g, h)
             if i is not None and j is not None and g[i] == h[j] and hw[j].get("s") is not None]
    diffs.sort()
    ts_med = diffs[len(diffs) // 2] if diffs else None
    ts_within = sum(d <= TS_OK_MS for d in diffs) / len(diffs) if diffs else None

    fill = FILLERS.get(locale, set())
    gf = sum(1 for w in g if w in fill)
    filler_recall = None
    if gf and locale in CJK:
        # no reliable word boundaries: count filler occurrences in the character stream instead
        gs, hs = "".join(g), "".join(h)
        filler_recall = min(1.0, sum(hs.count(f) for f in fill) / max(1, sum(gs.count(f) for f in fill)))
    elif gf:
        hf = {}
        for w in h:
            hf[w] = hf.get(w, 0) + 1
        got = 0
        for w in g:
            if w in fill and hf.get(w, 0) > 0:
                got += 1
                hf[w] -= 1
        filler_recall = got / gf

    gt = [t["w"] for t in gold if t["w"] in TAGS]
    ht = [t["w"] for t in hyp if t["w"] in TAGS]
    tag_f1 = None
    if gt or ht:
        common = sum(min(gt.count(x), ht.count(x)) for x in TAGS)
        p = common / len(ht) if ht else 0.0
        r = common / len(gt) if gt else 0.0
        tag_f1 = 2 * p * r / (p + r) if p + r else 0.0

    s = ItemScore(item_id, locale, round(err, 4), round(wer, 4), filler_recall, tag_f1, ts_med,
                  ts_within, rubric_violations(hyp, duration_ms))
    if s.err > max_err:
        s.why_rejected.append(f"{'CER' if locale in CJK else 'WER'} {s.err:.1%} > {max_err:.0%}")
    if s.violations:
        s.why_rejected.append(f"{len(s.violations)} rubric violation(s): {s.violations[0]}")
    if ts_med is not None and ts_med > max_ts_ms:
        s.why_rejected.append(f"word timestamps off by {ts_med:.0f} ms (median) > {max_ts_ms:.0f} ms")
    if filler_recall is not None and filler_recall < min_filler:
        s.why_rejected.append(f"fillers kept {filler_recall:.0%} (rubric: keep all)")
    s.approved = not s.why_rejected
    return s
