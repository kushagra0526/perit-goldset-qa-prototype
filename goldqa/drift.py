"""Weekly per-annotator drift: EWMA of each metric against its limit, first week it crosses."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

from .score import CJK, ItemScore

LAMBDA = 0.5
# metric -> (how to read it from a week of items, limit, "high" = bad when above / "low" = bad when below, label)
METRICS = {
    "err": (lambda xs: median(x.err for x in xs), 0.05, "high", "error rate"),
    "filler": (lambda xs: _mean([x.filler_recall for x in xs]), 0.9, "low", "fillers kept"),
    "tags": (lambda xs: _mean([x.tag_f1 for x in xs]), 0.85, "low", "tag F1"),
    "ts": (lambda xs: _mean([x.ts_median_ms for x in xs]), 50, "high", "timestamp error (ms)"),
    "rubric": (lambda xs: sum(len(x.violations) for x in xs) / len(xs), 0.5, "high", "rubric violations / item"),
}


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


@dataclass
class AnnotatorReport:
    annotator: str
    locale: str
    weekly: dict = field(default_factory=dict)      # metric -> [values per week]
    ewma: dict = field(default_factory=dict)
    flags: list[tuple[int, str, str]] = field(default_factory=list)   # (week, metric, message)
    approved_items: int = 0
    total_items: int = 0
    wer_only_would_flag: bool = False

    @property
    def status(self) -> str:
        return "RECALIBRATE" if self.flags else "OK"


def track(scores: dict[tuple[str, int], list[ItemScore]], locales: dict[str, str]) -> list[AnnotatorReport]:
    by_ann = defaultdict(dict)
    for (ann, week), items in scores.items():
        by_ann[ann][week] = items
    out = []
    for ann, weeks in sorted(by_ann.items()):
        r = AnnotatorReport(ann, locales[ann])
        for key, (get, limit, bad, label) in METRICS.items():
            vals, ew, e = [], [], None
            flagged = False
            for w in sorted(weeks):
                v = get(weeks[w])
                vals.append(v)
                if v is None:
                    ew.append(e)
                    continue
                e = v if e is None else LAMBDA * v + (1 - LAMBDA) * e
                ew.append(e)
                crossed = e > limit if bad == "high" else e < limit
                if crossed and not flagged:
                    flagged = True
                    shown = f"{e:.0f}" if key == "ts" else f"{e:.1%}" if key == "err" else f"{e:.2f}"
                    lim = f"{limit:.0%}" if key == "err" else limit
                    r.flags.append((w, key, f"{label} {shown} vs limit {lim} — from week {w}"))
            r.weekly[key], r.ewma[key] = vals, ew
        items = [x for w in weeks.values() for x in w]
        r.total_items, r.approved_items = len(items), sum(x.approved for x in items)
        if r.locale in CJK:
            r.wer_only_would_flag = median(x.wer for x in items) > 0.05 and median(x.err for x in items) <= 0.05
        out.append(r)
    return out
