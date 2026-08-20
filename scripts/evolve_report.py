import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_generations(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    generations = result.get("generations")
    if isinstance(generations, list) and generations:
        return [g for g in generations if isinstance(g, dict)]
    candidates = result.get("candidates", [])
    promoted = result.get("promoted", [])
    return [{"generation": 1, "parents": [], "candidates": candidates, "promoted": promoted}]


def _flatten_report(data: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(data.get("run_id", "unknown"))
    created_at = str(data.get("created_at", ""))
    results = data.get("results", [])

    all_candidates: List[Dict[str, Any]] = []
    generation_stats: List[Dict[str, Any]] = []
    metric_names = set()
    target_summaries: List[Dict[str, Any]] = []

    for target_idx, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        target = result.get("target", {}) if isinstance(result.get("target"), dict) else {}
        target_path = str(target.get("path", f"target_{target_idx+1}"))
        target_summaries.append(
            {
                "target_index": target_idx,
                "path": target_path,
                "start_line": int(target.get("start_line", 0) or 0),
                "end_line": int(target.get("end_line", 0) or 0),
            }
        )
        for gen in _iter_generations(result):
            generation = int(gen.get("generation", 1) or 1)
            candidates = gen.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
            scores = []
            promoted_count = 0
            status_counts: Dict[str, int] = {}
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                score = _safe_float(cand.get("score", 0.0))
                scores.append(score)
                promoted = bool(cand.get("promoted", False))
                if promoted:
                    promoted_count += 1
                status = str(cand.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1

                metrics = cand.get("metrics", [])
                metric_map: Dict[str, Any] = {}
                if isinstance(metrics, list):
                    for metric in metrics:
                        if not isinstance(metric, dict):
                            continue
                        name = str(metric.get("name", "metric"))
                        metric_names.add(name)
                        metric_map[name] = {
                            "score": metric.get("score"),
                            "passed": metric.get("passed"),
                            "details": str(metric.get("details", "") or ""),
                        }

                payload = cand.get("candidate_payload", {})
                edits = payload.get("edits", []) if isinstance(payload, dict) else []
                compact_edits = []
                if isinstance(edits, list):
                    for edit in edits[:10]:
                        if not isinstance(edit, dict):
                            continue
                        compact_edits.append(
                            {
                                "path": str(edit.get("path", "")),
                                "before": str(edit.get("before", "") or "")[:400],
                                "after": str(edit.get("after", "") or "")[:400],
                                "occurrence": edit.get("occurrence"),
                            }
                        )

                all_candidates.append(
                    {
                        "target_path": target_path,
                        "target_index": target_idx,
                        "generation": generation,
                        "id": str(cand.get("id", "")),
                        "run_candidate_id": str(cand.get("run_candidate_id", cand.get("id", ""))),
                        "summary": str(cand.get("summary", "") or ""),
                        "status": status,
                        "score": score,
                        "promoted": promoted,
                        "parents": [str(p) for p in cand.get("parents", []) if p],
                        "metrics": metric_map,
                        "edits": compact_edits,
                    }
                )
            if scores:
                best = max(scores)
                worst = min(scores)
                avg = sum(scores) / len(scores)
            else:
                best = 0.0
                worst = 0.0
                avg = 0.0
            generation_stats.append(
                {
                    "target_path": target_path,
                    "target_index": target_idx,
                    "generation": generation,
                    "count": len(scores),
                    "promoted_count": promoted_count,
                    "best_score": best,
                    "avg_score": avg,
                    "worst_score": worst,
                    "status_counts": status_counts,
                }
            )

    all_scores = [c["score"] for c in all_candidates]
    promoted_total = sum(1 for c in all_candidates if c.get("promoted"))
    apply_failed_total = sum(1 for c in all_candidates if c.get("status") == "apply_failed")
    rejected_total = sum(1 for c in all_candidates if c.get("status") == "rejected")
    scored_total = sum(1 for c in all_candidates if c.get("status") == "scored")

    overview = {
        "run_id": run_id,
        "created_at": created_at,
        "targets": len(target_summaries),
        "generations": len({(g["target_index"], g["generation"]) for g in generation_stats}),
        "candidates": len(all_candidates),
        "promoted": promoted_total,
        "best_score": max(all_scores) if all_scores else 0.0,
        "avg_score": (sum(all_scores) / len(all_scores)) if all_scores else 0.0,
        "apply_failed": apply_failed_total,
        "rejected": rejected_total,
        "scored": scored_total,
    }

    return {
        "overview": overview,
        "targets": target_summaries,
        "generation_stats": sorted(generation_stats, key=lambda x: (x["target_index"], x["generation"])),
        "metric_names": sorted(metric_names),
        "candidates": sorted(all_candidates, key=lambda x: x.get("score", 0.0), reverse=True),
    }


def _render_html(flat: Dict[str, Any], report_path: str) -> str:
    payload = json.dumps(flat, ensure_ascii=True).replace("</", "<\\/")
    title = f"Evolution Dashboard - {flat.get('overview', {}).get('run_id', '')}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --ink: #e5e7eb;
      --muted: #9ca3af;
      --good: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
      --accent: #06b6d4;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 500px at 0% 0%, #164e63 0%, transparent 65%),
        radial-gradient(1000px 450px at 100% 100%, #312e81 0%, transparent 60%),
        var(--bg);
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .hero {{
      display: flex; justify-content: space-between; align-items: end;
      gap: 16px; margin-bottom: 16px;
    }}
    .hero h1 {{ margin: 0; font-size: 28px; letter-spacing: 0.4px; }}
    .hero .meta {{ color: var(--muted); font-size: 13px; }}
    .cards {{
      display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px; margin-bottom: 16px;
    }}
    .card {{
      background: color-mix(in oklab, var(--panel) 88%, black 12%);
      border: 1px solid var(--border); border-radius: 10px; padding: 12px;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .6px; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .grid {{
      display: grid; grid-template-columns: 1.2fr .8fr; gap: 16px; margin-bottom: 16px;
    }}
    .panel {{
      background: color-mix(in oklab, var(--panel-soft) 90%, black 10%);
      border: 1px solid var(--border);
      border-radius: 12px; padding: 14px;
    }}
    .panel h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .chart-wrap {{ height: 220px; }}
    svg {{ width: 100%; height: 100%; overflow: visible; }}
    .axis text {{ fill: var(--muted); font-size: 11px; }}
    .axis line {{ stroke: #475569; stroke-width: 1; }}
    .line-best {{ fill: none; stroke: var(--good); stroke-width: 2.5; }}
    .line-avg {{ fill: none; stroke: var(--accent); stroke-width: 2; stroke-dasharray: 4 4; }}
    .legend {{ display: flex; gap: 12px; color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 99px; display: inline-block; margin-right: 5px; vertical-align: middle; }}
    .lineage {{
      max-height: 250px; overflow: auto; border: 1px solid var(--border);
      border-radius: 8px; padding: 8px; background: #0b1220;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    }}
    .lineage-row {{ padding: 3px 0; border-bottom: 1px dashed #233247; }}
    .lineage-row:last-child {{ border-bottom: 0; }}
    .table-wrap {{
      border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
      background: #0b1220;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead th {{
      text-align: left; background: #111827; color: var(--muted);
      padding: 10px; border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 1;
    }}
    tbody td {{ padding: 9px 10px; border-bottom: 1px solid #1f2c40; vertical-align: top; }}
    tbody tr:hover {{ background: #122033; cursor: pointer; }}
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 99px;
      font-size: 11px; font-weight: 700;
    }}
    .b-promoted {{ background: color-mix(in oklab, var(--good) 25%, transparent); color: #86efac; }}
    .b-rejected {{ background: color-mix(in oklab, var(--warn) 28%, transparent); color: #fcd34d; }}
    .b-apply_failed {{ background: color-mix(in oklab, var(--bad) 25%, transparent); color: #fca5a5; }}
    .b-scored {{ background: color-mix(in oklab, var(--accent) 22%, transparent); color: #67e8f9; }}
    .details {{
      margin-top: 12px; border: 1px solid var(--border); border-radius: 10px;
      background: #0b1220; padding: 12px;
    }}
    .details h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .metric-grid {{
      display: grid; grid-template-columns: repeat(3, minmax(140px, 1fr));
      gap: 8px; margin-bottom: 10px;
    }}
    .metric-cell {{
      border: 1px solid #2a3a54; border-radius: 8px; padding: 8px; background: #111a2b;
    }}
    .muted {{ color: var(--muted); }}
    .edits pre {{
      white-space: pre-wrap; word-break: break-word; margin: 6px 0 10px;
      background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 8px;
      max-height: 140px; overflow: auto;
    }}
    .filters {{ display: flex; gap: 8px; margin-bottom: 10px; }}
    .filters input, .filters select {{
      background: #111827; color: var(--ink); border: 1px solid #334155;
      border-radius: 8px; padding: 8px 10px; font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>Evolution Dashboard</h1>
        <div class="meta">Run: <code>{flat.get("overview", {}).get("run_id", "")}</code> | Source: <code>{report_path}</code></div>
      </div>
      <div class="meta">Generated by <code>scripts/evolve_report.py</code></div>
    </div>

    <div id="cards" class="cards"></div>

    <div class="grid">
      <div class="panel">
        <h2>Generation Score Trend</h2>
        <div class="chart-wrap"><svg id="trendChart" viewBox="0 0 900 220" preserveAspectRatio="none"></svg></div>
        <div class="legend">
          <span><i class="dot" style="background: var(--good)"></i>Best</span>
          <span><i class="dot" style="background: var(--accent)"></i>Average</span>
        </div>
      </div>
      <div class="panel">
        <h2>Lineage</h2>
        <div id="lineage" class="lineage"></div>
      </div>
    </div>

    <div class="panel">
      <h2>Candidate Leaderboard</h2>
      <div class="filters">
        <input id="search" placeholder="Search summary, id, path..." />
        <select id="statusFilter">
          <option value="">All statuses</option>
          <option value="scored">scored</option>
          <option value="rejected">rejected</option>
          <option value="apply_failed">apply_failed</option>
        </select>
        <select id="targetFilter"><option value="">All targets</option></select>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Target</th><th>Gen</th><th>Candidate</th><th>Status</th><th>Score</th><th>Promoted</th><th>Summary</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
      <div id="details" class="details"><h3>Select a candidate</h3><div class="muted">Click a row for metrics and edits.</div></div>
    </div>
  </div>

  <script id="reportData" type="application/json">{payload}</script>
  <script>
  (() => {{
    const data = JSON.parse(document.getElementById('reportData').textContent);
    const overview = data.overview || {{}};
    const candidates = data.candidates || [];
    const generationStats = data.generation_stats || [];
    const metricNames = data.metric_names || [];

    const cards = [
      ['Targets', overview.targets || 0],
      ['Candidates', overview.candidates || 0],
      ['Promoted', overview.promoted || 0],
      ['Best Score', (overview.best_score || 0).toFixed(3)],
      ['Avg Score', (overview.avg_score || 0).toFixed(3)],
      ['Failures', overview.apply_failed || 0],
    ];
    document.getElementById('cards').innerHTML = cards.map(([k,v]) =>
      `<div class="card"><div class="label">${{k}}</div><div class="value">${{v}}</div></div>`
    ).join('');

    function buildTrend() {{
      const svg = document.getElementById('trendChart');
      const width = 900, height = 220;
      const pad = {{ l: 40, r: 10, t: 10, b: 26 }};
      const points = generationStats.slice().sort((a,b) => (a.generation - b.generation) || (a.target_index - b.target_index));
      if (!points.length) {{
        svg.innerHTML = '<text x="20" y="40" fill="#9ca3af">No generation data</text>';
        return;
      }}
      const xs = points.map((_,i)=>i);
      const xMax = Math.max(1, xs.length - 1);
      const scaleX = i => pad.l + (i / xMax) * (width - pad.l - pad.r);
      const scaleY = v => pad.t + ((1 - v) * (height - pad.t - pad.b));
      const yTicks = [0,0.25,0.5,0.75,1];
      const axis = yTicks.map(v =>
        `<line x1="${{pad.l}}" y1="${{scaleY(v)}}" x2="${{width-pad.r}}" y2="${{scaleY(v)}}" stroke="#1f2c40"/><text x="4" y="${{scaleY(v)+4}}" fill="#9ca3af" font-size="10">${{v.toFixed(2)}}</text>`
      ).join('');
      const xAxis = points.map((p,i) =>
        `<text x="${{scaleX(i)-8}}" y="${{height-8}}" fill="#9ca3af" font-size="10">g${{p.generation}}</text>`
      ).join('');
      const pathFor = key => points.map((p,i) => `${{i?'L':'M'}} ${{scaleX(i)}} ${{scaleY(p[key] || 0)}}`).join(' ');
      svg.innerHTML = `
        ${{axis}}
        <line x1="${{pad.l}}" y1="${{height-pad.b}}" x2="${{width-pad.r}}" y2="${{height-pad.b}}" stroke="#475569"/>
        <path class="line-best" d="${{pathFor('best_score')}}"></path>
        <path class="line-avg" d="${{pathFor('avg_score')}}"></path>
        ${{xAxis}}
      `;
    }}

    function statusBadge(status, promoted) {{
      if (promoted) return '<span class="badge b-promoted">PROMOTED</span>';
      const cls = 'b-' + status;
      return `<span class="badge ${{cls}}">${{status}}</span>`;
    }}

    const rowsEl = document.getElementById('rows');
    const detailsEl = document.getElementById('details');
    const targetFilter = document.getElementById('targetFilter');
    const lineageEl = document.getElementById('lineage');
    const search = document.getElementById('search');
    const statusFilter = document.getElementById('statusFilter');

    const targetSet = Array.from(new Set(candidates.map(c => c.target_path))).sort();
    targetFilter.innerHTML += targetSet.map(t => `<option value="${{t}}">${{t}}</option>`).join('');

    function renderLineage(list) {{
      const rows = list
        .filter(c => c.parents && c.parents.length)
        .map(c => `<div class="lineage-row">${{c.parents.join(' + ')}} -> ${{c.run_candidate_id}}</div>`);
      lineageEl.innerHTML = rows.length ? rows.join('') : '<div class="muted">No parent-child links in this run.</div>';
    }}

    function openDetails(candidate) {{
      const metricRows = metricNames.map(name => {{
        const m = (candidate.metrics || {{}})[name] || {{}};
        const s = (m.score === null || m.score === undefined) ? 'n/a' : Number(m.score).toFixed(3);
        return `<div class="metric-cell"><div class="label">${{name}}</div><div class="value" style="font-size:18px">${{s}}</div><div class="muted">${{m.details || ''}}</div></div>`;
      }}).join('');

      const edits = (candidate.edits || []).map((e, idx) => `
        <div class="edits">
          <div><strong>Edit ${{idx+1}}</strong> <span class="muted">${{e.path}}</span></div>
          <div class="muted">Before</div><pre>${{(e.before || '').replace(/[<>&]/g, m => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}}[m]))}}</pre>
          <div class="muted">After</div><pre>${{(e.after || '').replace(/[<>&]/g, m => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}}[m]))}}</pre>
        </div>
      `).join('');

      detailsEl.innerHTML = `
        <h3>${{candidate.run_candidate_id}} <span class="muted">(${{candidate.target_path}})</span></h3>
        <div class="muted">Generation ${{candidate.generation}} | Score ${{Number(candidate.score || 0).toFixed(4)}} | Status: ${{candidate.status}}</div>
        <div class="muted">Parents: ${{(candidate.parents || []).length ? candidate.parents.join(', ') : 'none'}}</div>
        <p>${{candidate.summary || ''}}</p>
        <div class="metric-grid">${{metricRows || '<div class="muted">No metrics</div>'}}</div>
        ${{edits || '<div class="muted">No edits captured.</div>'}}
      `;
    }}

    function renderRows() {{
      const q = (search.value || '').toLowerCase().trim();
      const s = statusFilter.value;
      const t = targetFilter.value;
      const filtered = candidates.filter(c => {{
        if (s && c.status !== s) return false;
        if (t && c.target_path !== t) return false;
        if (!q) return true;
        const hay = `${{c.run_candidate_id}} ${{c.summary}} ${{c.target_path}}`.toLowerCase();
        return hay.includes(q);
      }});
      rowsEl.innerHTML = filtered.map((c, i) => `
        <tr data-id="${{c.run_candidate_id}}">
          <td>${{i + 1}}</td>
          <td><code>${{c.target_path}}</code></td>
          <td>${{c.generation}}</td>
          <td><code>${{c.run_candidate_id}}</code></td>
          <td>${{statusBadge(c.status, c.promoted)}}</td>
          <td>${{Number(c.score || 0).toFixed(4)}}</td>
          <td>${{c.promoted ? 'yes' : 'no'}}</td>
          <td>${{c.summary || ''}}</td>
        </tr>
      `).join('');
      for (const row of rowsEl.querySelectorAll('tr')) {{
        row.addEventListener('click', () => {{
          const id = row.getAttribute('data-id');
          const cand = filtered.find(x => x.run_candidate_id === id);
          if (cand) openDetails(cand);
        }});
      }}
      if (filtered.length && !detailsEl.dataset.locked) {{
        openDetails(filtered[0]);
      }}
    }}

    [search, statusFilter, targetFilter].forEach(el => el.addEventListener('input', renderRows));
    buildTrend();
    renderLineage(candidates);
    renderRows();
  }})();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from evolve report.json.")
    parser.add_argument("--report", required=True, help="Path to report.json")
    parser.add_argument("--output", required=True, help="Path to output .html")
    args = parser.parse_args()

    report_path = os.path.abspath(args.report)
    output_path = os.path.abspath(args.output)
    data = _load_json(report_path)
    flat = _flatten_report(data)
    html = _render_html(flat, report_path=report_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[ok] wrote dashboard: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
