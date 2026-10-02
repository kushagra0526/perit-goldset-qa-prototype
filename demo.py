"""python demo.py -> scores 960 synthetic gold items (12 annotators x 8 weeks), prints the recalibration list, writes out/dashboard.html"""
from __future__ import annotations

import time
from collections import defaultdict
from html import escape
from pathlib import Path

from goldqa.drift import METRICS, track
from goldqa.score import score
from goldqa.simulate import ANNOTATORS, WEEKS, generate

HERE = Path(__file__).parent
CSS = """
:root{--bg:#f8f8f6;--fg:#1b1b19;--mut:#6c6a64;--line:#e3e1db;--card:#fff;--ok:#1d7a4f;--bad:#c2412d;--acc:#2f62c9}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--fg:#ecebe6;--mut:#9a978f;--line:#2b2a27;--card:#1a1a18;--ok:#58c08d;--bad:#ef7a66;--acc:#7fa6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1220px;margin:0 auto;padding:30px 16px}h1{font-size:25px;margin:0}h2{font-size:17px;margin:26px 0 10px}.mut{color:var(--mut)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px}.kpi b{display:block;font-size:26px}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
th,td{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:middle}th{font-size:11px;text-transform:uppercase;color:var(--mut)}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}.num{font-variant-numeric:tabular-nums}
"""


def spark(vals, limit, bad, w=110, h=26):
    v = [x for x in vals if x is not None]
    if not v:
        return ""
    lo, hi = min(v + [limit]), max(v + [limit])
    hi = hi if hi > lo else lo + 1
    y = lambda x: h - 3 - (x - lo) / (hi - lo) * (h - 6)
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{y(x):.1f}" for i, x in enumerate(vals) if x is not None)
    crossed = any((x > limit if bad == "high" else x < limit) for x in v)
    col = "var(--bad)" if crossed else "var(--ok)"
    return (f'<svg width="{w}" height="{h}"><line x1="0" x2="{w}" y1="{y(limit):.1f}" y2="{y(limit):.1f}" stroke="currentColor" '
            f'stroke-dasharray="3 3" opacity=".35"/><polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/></svg>')


def main():
    t0 = time.perf_counter()
    rows = generate()
    scores = defaultdict(list)
    for r in rows:
        scores[(r["annotator"], r["week"])].append(score(r["item"], r["locale"], r["gold"], r["hyp"], r["duration"]))
    reports = track(scores, {a: v[0] for a, v in ANNOTATORS.items()})
    secs = time.perf_counter() - t0
    flagged = [r for r in reports if r.flags]
    all_items = [x for v in scores.values() for x in v]
    zh = [x for x in all_items if x.locale == "zh"]
    zh_wer_reject = sum(1 for x in zh if x.wer > 0.05)

    print(f"\n{len(all_items)} gold items · {len(reports)} annotators · {WEEKS} weeks · scored in {secs:.1f}s\n")
    print("RECALIBRATE")
    for r in flagged:
        for w, k, msg in r.flags:
            print(f"  {r.annotator} ({r.locale})  {msg}")
    print(f"\nOK: {', '.join(r.annotator for r in reports if not r.flags)}")
    print(f"\nCJK trap: zh annotator A07 — WER would reject {zh_wer_reject}/{len(zh)} items; CER rejects "
          f"{sum(1 for x in zh if not x.approved)}/{len(zh)} (same characters, different word spaces)")

    trs = ""
    for r in reports:
        sp = "".join(f"<td>{spark(r.weekly[k], METRICS[k][1], METRICS[k][2])}</td>" for k in METRICS)
        why = "<br>".join(escape(m) for _, _, m in r.flags) or ("<span class=mut>clean</span>" if not r.wer_only_would_flag else
                                                                "<span class=mut>clean — would have failed on WER; scored on CER</span>")
        trs += (f"<tr><td><b>{r.annotator}</b> <span class=mut>{r.locale}</span></td><td class={'bad' if r.flags else 'ok'}>{r.status}</td>"
                f"<td class=num>{r.approved_items}/{r.total_items}</td>{sp}<td>{why}</td></tr>")
    heads = "".join(f"<th>{escape(METRICS[k][3])}</th>" for k in METRICS)
    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Gold-Set QA Dashboard</title><style>{CSS}</style></head><body><main>
<h1>Weekly gold-set QA — annotator drift</h1>
<div class=mut>Synthetic gold sets in 9 locales · 12 simulated annotators · {WEEKS} weeks · rubric: lowercase, fillers kept, one tag convention, timestamp on every word · unofficial prototype by Kushagra Pratap Singh</div>
<div class=kpis><div class=kpi><span class=mut>Gold items scored</span><b>{len(all_items)}</b></div>
<div class=kpi><span class=mut>Approved (pay gate)</span><b>{sum(x.approved for x in all_items)}</b></div>
<div class=kpi><span class=mut>Recalibrate</span><b class=bad>{len(flagged)}</b></div>
<div class=kpi><span class=mut>zh items WER would wrongly reject</span><b>{zh_wer_reject}/{len(zh)}</b></div></div>
<h2>Annotators — weekly trend per metric (dashed = limit)</h2><div class=scroll><table>
<tr><th>Annotator</th><th>Status</th><th>Approved</th>{heads}<th>Why</th></tr>{trs}</table></div>
</main></body></html>"""
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"\ndashboard: {HERE / 'out' / 'dashboard.html'}")


if __name__ == "__main__":
    main()
