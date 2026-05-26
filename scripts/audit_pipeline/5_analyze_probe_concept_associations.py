"""Audit pipeline — stage 5: analyze and visualize probe–concept associations.

For each probe, ranks concepts by

    association = freq(concept | top-third projection) − freq(concept | bottom-third)

A concept equally common in both tails (e.g. the NLA's generic "Q&A format" tic) cancels to
~0 and drops out; the top-ranked concepts are what the probe actually keys on — no
pre-specified taxonomy.

Renders ONE interactive, self-contained HTML for the run: a probe toggle switches the chart;
clicking a concept bar shows the activations tagged with it, each traced end-to-end
(prompt → NLA readout → probe firing → concepts). Offline (no GPU/API).

Run:    python scripts/audit_pipeline/5_analyze_probe_concept_associations.py [--run-name v1] [--grade-tag <t>] [--top-frac 0.33] [--min-count 4]
Input:  data/results/part3_concepts_<run>[_<grade-tag>].json + part3_probe_<run>.json
        + part3_verbalize_<run>.json + part3_prompts_<run>.json (fallback: part3_audit_<run>.json)
Output: data/results/part3_association_<run>[_<grade-tag>].json + .html (interactive explorer)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/   -> _cli
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root  -> src

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "results")


def association(concepts, proj, top_frac, min_count):
    """Rank concepts by freq(concept | high projection) − freq(concept | low projection).

    high/low are the top/bottom `top_frac` of activations by projection. Concepts seen fewer
    than `min_count` times overall are dropped. Returns (table sorted by descending
    association, per-tail size k).
    """
    n = len(proj)
    k = max(1, int(round(n * top_frac)))
    order = np.argsort(proj)
    low_idx, high_idx = set(order[:k].tolist()), set(order[-k:].tolist())
    rows = {}
    for i, tags in enumerate(concepts):
        for t in (tags or []):
            r = rows.setdefault(t, {"high": 0, "low": 0, "count": 0})
            r["count"] += 1
            if i in high_idx:
                r["high"] += 1
            elif i in low_idx:
                r["low"] += 1
    out = []
    for t, r in rows.items():
        if r["count"] < min_count:
            continue
        fh, fl = r["high"] / k, r["low"] / k
        out.append({"concept": t, "freq_high": fh, "freq_low": fl,
                    "association": fh - fl, "count": r["count"]})
    out.sort(key=lambda x: -x["association"])
    return out, k


def tails(proj, top_frac):
    """Per-activation tail label: 'high'/'low'/'mid' by projection rank (matches association)."""
    n = len(proj)
    k = max(1, int(round(n * top_frac)))
    order = np.argsort(proj)
    low_idx, high_idx = set(order[:k].tolist()), set(order[-k:].tolist())
    return ["high" if i in high_idx else "low" if i in low_idx else "mid" for i in range(n)]


def load_instructions(rn, n):
    """Prompt instructions per activation: prompts file → legacy audit examples → blank."""
    p = os.path.join(RESULTS_DIR, f"part3_prompts_{rn}.json")
    if os.path.exists(p):
        pr = json.load(open(p))["prompts"]
        return [(pr[i].get("instruction", "") if i < len(pr) else "") for i in range(n)]
    a = os.path.join(RESULTS_DIR, f"part3_audit_{rn}.json")
    if os.path.exists(a):
        ex = json.load(open(a)).get("examples", [])
        return [(ex[i].get("instruction", "") if i < len(ex) else "") for i in range(n)]
    return [""] * n


def render(data, title):
    data_js = json.dumps(data).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_js)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--grade-tag", default=None, help="concepts grade tag (default: ungraded)")
    ap.add_argument("--top-frac", type=float, default=0.33)
    ap.add_argument("--min-count", type=int, default=4)
    args = ap.parse_args()

    import _cli
    if _cli.interactive():
        token = _cli.choose_run("part3_concepts_*.json", r"part3_concepts_(.+)\.json", "concept set")
        args.top_frac = _cli.ask_float("Top/bottom fraction per tail", args.top_frac)
        args.min_count = _cli.ask_int("Drop concepts seen fewer than N times", args.min_count)
    else:
        token = f"{args.run_name}_{args.grade_tag}" if args.grade_tag else args.run_name

    cj = json.load(open(os.path.join(RESULTS_DIR, f"part3_concepts_{token}.json")))
    concepts = [c or [] for c in cj["concepts"]]
    rn, grade_tag = cj["run_name"], cj.get("grade_tag")
    vj = json.load(open(os.path.join(RESULTS_DIR, f"part3_verbalize_{rn}.json")))
    pj = json.load(open(os.path.join(RESULTS_DIR, f"part3_probe_{rn}.json")))

    n = len(concepts)
    cells, expls = vj["cells"], vj["explanations"]
    instrs = load_instructions(rn, n)
    acts = [{"cell": cells[i], "instr": instrs[i], "expl": expls[i], "tags": concepts[i]}
            for i in range(n)]

    probes, assoc_tables = {}, {}
    for name, blk in pj["probes"].items():
        proj = np.asarray(blk["projection"], dtype=np.float64)
        table, k = association(concepts, proj, args.top_frac, args.min_count)
        assoc_tables[name] = table
        probes[name] = {
            "src": blk.get("src"),
            "proj": [round(float(x), 4) for x in proj],
            "fires": blk.get("fires"),
            "tail": tails(proj, args.top_frac),
            "assoc": table,
        }
        print(f"\n=== {name} probe — top associated concepts (k={k}) ===")
        for r in table[:8]:
            print(f"  {r['association']:+.2f}  {r['concept']}  "
                  f"(high {r['freq_high']:.2f} / low {r['freq_low']:.2f}, n={r['count']})")

    config = {"top_frac": args.top_frac, "min_count": args.min_count, "n": n}
    suffix = f"{rn}_{grade_tag}" if grade_tag else rn

    out = {"run_name": rn, "grade_tag": grade_tag, "model": cj.get("model"),
           "config": config, "association": assoc_tables}
    json.dump(out, open(os.path.join(RESULTS_DIR, f"part3_association_{suffix}.json"), "w"), indent=2)

    data = {"run": rn, "grade_tag": grade_tag, "model": cj.get("model"),
            "config": config, "acts": acts, "probes": probes}
    html_path = os.path.join(RESULTS_DIR, f"part3_association_{suffix}.html")
    with open(html_path, "w") as f:
        f.write(render(data, f"{rn}{(' / ' + grade_tag) if grade_tag else ''}"))
    print(f"\nSaved part3_association_{suffix}.json\nSaved {html_path}")


_TEMPLATE = r"""<!doctype html><meta charset="utf-8">
<title>Concept association — __TITLE__</title>
<style>
 body{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:28px auto;padding:0 16px;color:#222}
 h1{font-size:20px;margin-bottom:2px} .sub{color:#777;font-size:12.5px;margin-bottom:6px}
 .controls{margin:14px 0 6px} select{font-size:13px;padding:3px 6px}
 .wrap{display:flex;gap:28px;align-items:flex-start}
 .chart{flex:0 0 470px} .detail{flex:1;min-width:0}
 h3{font-size:14px;margin:0 0 8px}
 .row{display:grid;grid-template-columns:190px 1fr 44px;align-items:center;gap:8px;margin:2px 0;
   background:none;border:0;padding:0;width:100%;cursor:pointer;text-align:inherit;font:inherit}
 .row:hover .lbl{color:#000} .row.sel .lbl{font-weight:700;color:#000}
 .lbl{font-size:12px;text-align:right;color:#333;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .track{position:relative;height:17px;background:#f4f4f6;border-radius:3px}
 .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#ccc}
 .bar{position:absolute;top:2px;bottom:2px;border-radius:2px}
 .val{font-size:12px;color:#555}
 .dhead{font-size:14px;margin:0 0 10px} .dhead b{font-size:15px}
 .card{border:1px solid #e8e8ea;border-radius:7px;padding:10px 12px;margin:0 0 12px;background:#fcfcfd}
 .stage{margin:6px 0} .stage .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:#999}
 .badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;color:#fff;margin-right:6px}
 .b-harmful{background:#d1495b} .b-harmless{background:#3f8e5b}
 .instr{font-weight:600}
 .verdict{font-size:12.5px} .v-yes{color:#d1495b;font-weight:700} .v-no{color:#3f8e5b;font-weight:700}
 .tail{font-size:11px;color:#777;margin-left:6px}
 .expl{font-size:12.5px;color:#333;background:#fff;border:1px solid #eee;border-radius:5px;
   padding:7px 9px;max-height:150px;overflow:auto;white-space:pre-wrap}
 .chip{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px;background:#eef;color:#334;margin:2px 3px 0 0}
 .chip.on{background:#2563eb;color:#fff}
 .arrow{color:#bbb;margin:0 4px}
 .empty{color:#999}
</style>
<h1>Concept association — what does the probe fire on?</h1>
<div class="sub" id="sub"></div>
<div class="controls">probe:
 <select id="probe"></select>
 <span class="sub" id="psrc"></span></div>
<p class="sub">Click a concept bar to trace the activations tagged with it
 (prompt → NLA readout → probe firing → concepts). Bars right = the probe fires more when the
 concept is present; baseline tics appear in both tails and cancel to ~0.</p>
<div class="wrap">
 <div class="chart"><h3 id="ctitle"></h3><div id="bars"></div></div>
 <div class="detail" id="detail"></div>
</div>
<script>
const D = __DATA__;
const POS = "#d1495b", NEG = "#9aa0a6";
const esc = s => (s==null?"":String(s)).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let probe = Object.keys(D.probes)[0];
let concept = null;

document.getElementById("sub").textContent =
  `run ${D.run}${D.grade_tag ? " / "+D.grade_tag : ""} · ${D.model||""} · `
  + `association = freq(concept | top ${Math.round(D.config.top_frac*100)}% proj) − freq(bottom ${Math.round(D.config.top_frac*100)}%) · `
  + `n=${D.config.n}, min-count ${D.config.min_count}`;

const sel = document.getElementById("probe");
Object.keys(D.probes).forEach(name => {
  const o = document.createElement("option"); o.value = name; o.textContent = name; sel.appendChild(o);
});
sel.value = probe;
sel.addEventListener("change", () => { probe = sel.value; concept = null; renderChart(); });

function renderChart() {
  const P = D.probes[probe];
  document.getElementById("ctitle").textContent = `${probe} probe — ${P.assoc.length} concepts`;
  document.getElementById("psrc").textContent = P.src ? `(data/probes/${P.src})` : "";
  const mx = Math.max(1e-9, ...P.assoc.map(r => Math.abs(r.association)));
  const bars = document.getElementById("bars");
  bars.innerHTML = "";
  if (!concept || !P.assoc.some(r => r.concept === concept)) concept = P.assoc.length ? P.assoc[0].concept : null;
  P.assoc.forEach(r => {
    const e = r.association, w = Math.abs(e)/mx*46, left = e>=0 ? 50 : 50-w, color = e>=0 ? POS : NEG;
    const btn = document.createElement("button");
    btn.className = "row" + (r.concept===concept ? " sel" : "");
    btn.title = `high ${r.freq_high.toFixed(2)} / low ${r.freq_low.toFixed(2)}, n=${r.count}`;
    btn.innerHTML = `<div class="lbl">${esc(r.concept)}</div>`
      + `<div class="track"><div class="mid"></div>`
      + `<div class="bar" style="left:${left.toFixed(1)}%;width:${w.toFixed(1)}%;background:${color}"></div></div>`
      + `<div class="val">${e>=0?"+":""}${e.toFixed(2)}</div>`;
    btn.addEventListener("click", () => { concept = r.concept; renderChart(); });
    bars.appendChild(btn);
  });
  renderDetail();
}

function renderDetail() {
  const det = document.getElementById("detail");
  const P = D.probes[probe];
  if (!concept) { det.innerHTML = '<p class="empty">No concepts pass the min-count filter.</p>'; return; }
  const idxs = D.acts.map((a,i)=>i).filter(i => D.acts[i].tags.includes(concept));
  idxs.sort((a,b) => P.proj[b] - P.proj[a]);
  const hi = idxs.filter(i => P.tail[i]==="high").length;
  const lo = idxs.filter(i => P.tail[i]==="low").length;
  let h = `<div class="dhead"><b>${esc(concept)}</b> — ${idxs.length} activations `
        + `(${hi} in high third, ${lo} in low third)</div>`;
  h += idxs.map(i => card(i, P)).join("");
  det.innerHTML = h;
}

function card(i, P) {
  const a = D.acts[i];
  const harm = a.cell && a.cell.indexOf("harmless")<0 && a.cell.indexOf("harmful")===0;
  const fires = P.fires ? P.fires[i] : null;
  const verdict = fires===null ? "" :
    `<span class="${fires?"v-yes":"v-no"}">${fires?"FIRES":"quiet"}</span> `;
  const chips = a.tags.map(t => `<span class="chip${t===concept?" on":""}">${esc(t)}</span>`).join("");
  return `<div class="card">
    <div class="stage"><span class="k">prompt</span><br>
      <span class="badge ${harm?"b-harmful":"b-harmless"}">${esc(a.cell)}</span>
      <span class="instr">${esc(a.instr)||"<em>(instruction unavailable)</em>"}</span></div>
    <div class="stage"><span class="k">probe</span> <span class="arrow">→</span>
      <span class="verdict">${verdict}proj ${P.proj[i].toFixed(2)}<span class="tail">(${P.tail[i]} third)</span></span></div>
    <div class="stage"><span class="k">NLA readout</span><div class="expl">${esc(a.expl)}</div></div>
    <div class="stage"><span class="k">concepts</span><br>${chips}</div>
  </div>`;
}

renderChart();
</script>
"""


if __name__ == "__main__":
    main()
