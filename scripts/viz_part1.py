"""Render a Part 1 results JSON as a self-contained HTML report (no plotting deps).

Shows: per-layer AUROC, refusal-mention rates by probe polarity, an SVG strip plot of
held-out L53 probe scores (threshold marked, colored by harmful/harmless), and a table of
every prompt with its score, fires flag, refusal-keyword hits, and the full NLA explanation.

Run:  python scripts/viz_part1.py [results.json]   # default: latest part1_refusal_nla_*.json
"""
import glob
import html
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")

HARMFUL, HARMLESS = "#d1495b", "#3a6ea5"


def latest_results():
    files = glob.glob(os.path.join(RESULTS_DIR, "part1_refusal_nla_*.json"))
    files = [f for f in files if not f.endswith(".html")]
    if not files:
        sys.exit("No part1_refusal_nla_*.json found in data/results/")
    return max(files, key=os.path.getmtime)


def strip_plot_svg(examples, threshold, w=860, h=170, pad=40):
    scores = [e["score_l53"] for e in examples]
    lo, hi = min(scores + [threshold]), max(scores + [threshold])
    span = (hi - lo) or 1.0
    lo -= 0.06 * span
    hi += 0.06 * span

    def x(s):
        return pad + (s - lo) / (hi - lo) * (w - 2 * pad)

    mid = h / 2
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    # axis
    parts.append(f'<line x1="{pad}" y1="{mid}" x2="{w-pad}" y2="{mid}" stroke="#ccc"/>')
    for s in _ticks(lo, hi):
        parts.append(f'<line x1="{x(s):.1f}" y1="{mid-4}" x2="{x(s):.1f}" y2="{mid+4}" stroke="#999"/>')
        parts.append(f'<text x="{x(s):.1f}" y="{mid+20}" font-size="11" text-anchor="middle" fill="#666">{s:g}</text>')
    # threshold
    tx = x(threshold)
    parts.append(f'<line x1="{tx:.1f}" y1="20" x2="{tx:.1f}" y2="{h-20}" stroke="#e09f3e" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{tx:.1f}" y="14" font-size="11" text-anchor="middle" fill="#b07d29">threshold {threshold:.2f}</text>')
    # jittered dots
    seen = {}
    for e in examples:
        s = e["score_l53"]
        key = round(x(s))
        seen[key] = seen.get(key, 0) + 1
        dy = (seen[key] - 1) * 9 * (1 if seen[key] % 2 else -1) / 2
        color = HARMFUL if e["label"] == 1 else HARMLESS
        stroke = "#222" if e["fires"] else "none"
        parts.append(
            f'<circle cx="{x(s):.1f}" cy="{mid-26+dy:.1f}" r="6" fill="{color}" '
            f'stroke="{stroke}" stroke-width="1.5"><title>{html.escape(e["instruction"][:120])}\n'
            f'score={s:.2f} fires={e["fires"]}</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _ticks(lo, hi, n=8):
    step = (hi - lo) / n
    mag = 10 ** (len(str(int(abs(step)))) - 1) if step >= 1 else 1
    step = max(round(step / mag) * mag, 1)
    start = int(lo // step) * step
    return [start + i * step for i in range(n + 3) if lo <= start + i * step <= hi]


def render(path):
    d = json.load(open(path))
    ex = sorted(d["examples"], key=lambda e: -e["score_l53"])
    thr = d["layer_report"][str(d["nla_layer"])]["threshold"]
    s = d["summary"]

    auroc_cells = "".join(
        f'<td>L{L}<br><b>{a:.3f}</b></td>' for L, a in s["layer_auroc"].items()
    )
    rows = []
    for e in ex:
        color = HARMFUL if e["label"] == 1 else HARMLESS
        hits = ", ".join(e["refusal_keyword_hits"]) or "<i>none</i>"
        rows.append(
            f'<tr><td style="color:{color};font-weight:600">{"harmful" if e["label"] else "harmless"}</td>'
            f'<td style="text-align:right">{e["score_l53"]:.2f}</td>'
            f'<td style="text-align:center">{"●" if e["fires"] else "○"}</td>'
            f'<td>{html.escape(e["instruction"][:140])}</td>'
            f'<td style="font-size:12px;color:#555">{hits}</td>'
            f'<td><details><summary>show</summary><div class="expl">{html.escape(e["explanation"])}</div></details></td></tr>'
        )

    return f"""<!doctype html><meta charset="utf-8">
<title>Part 1 — {html.escape(d['run_id'])}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:30px auto;padding:0 16px;color:#222}}
 h1{{font-size:20px}} .sub{{color:#777;font-size:13px;margin-bottom:20px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
 .card{{background:#f6f7f9;border-radius:8px;padding:12px 16px;flex:1;min-width:170px}}
 .card .big{{font-size:26px;font-weight:700}}
 table{{border-collapse:collapse;width:100%;margin-top:10px}}
 th,td{{border-bottom:1px solid #eee;padding:6px 8px;vertical-align:top;text-align:left}}
 th{{font-size:12px;text-transform:uppercase;color:#888}}
 .expl{{white-space:pre-wrap;background:#fafafa;border-left:3px solid #ddd;padding:8px;margin-top:6px;font-size:13px}}
 .legend span{{margin-right:16px}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
 table.auroc td{{text-align:center;border:none;padding:4px 14px}}
</style>
<h1>Part 1 — Does the NLA verbalize refusal?</h1>
<div class="sub">{html.escape(d['model'])} · NLA {html.escape(d['nla_checkpoint'])} · run <code>{html.escape(d['run_id'])}</code>
 · {len(ex)} held-out prompts</div>

<div class="cards">
 <div class="card">refusal mention rate<br><span class="big" style="color:{HARMFUL}">{s['refusal_mention_rate_probe_positive']*100:.0f}%</span><br>probe-positive (fires)</div>
 <div class="card">refusal mention rate<br><span class="big" style="color:{HARMLESS}">{s['refusal_mention_rate_probe_negative']*100:.0f}%</span><br>probe-negative</div>
 <div class="card">held-out AUROC<br><table class="auroc"><tr>{auroc_cells}</tr></table><span style="font-size:12px;color:#777">L{d['nla_layer']} committed for NLA</span></div>
</div>

<h3>Held-out L{d['nla_layer']} probe scores</h3>
<div class="legend"><span><span class="dot" style="background:{HARMFUL}"></span>harmful</span>
 <span><span class="dot" style="background:{HARMLESS}"></span>harmless</span>
 <span>◯ outline = fires the probe</span></div>
{strip_plot_svg(ex, thr)}

<h3>Per-prompt explanations</h3>
<table><tr><th>class</th><th>L{d['nla_layer']} score</th><th>fires</th><th>instruction</th><th>refusal keywords</th><th>NLA explanation</th></tr>
{''.join(rows)}
</table>
"""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_results()
    out = path.rsplit(".", 1)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(path))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
