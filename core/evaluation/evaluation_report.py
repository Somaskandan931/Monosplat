"""
core/evaluation/evaluation_report.py
--------------------------------------
EvaluationReport — aggregates PSNR, SSIM, LPIPS, FPS, training duration,
and model size into a structured report, then serialises it to:

    evaluation_report.json
    evaluation_report.md
    evaluation_report.html

Also provides run-comparison support via EvaluationReport.compare().

Design
------
  - Completely decoupled from SQLAlchemy / FastAPI — works standalone and
    from the CLI.
  - All render/metric numbers are pre-computed by the individual evaluators;
    this class only assembles, formats, and persists them.
  - JSON is the authoritative store; MD and HTML are generated from it.
  - compare() accepts a list of report dicts (loaded from JSON) and returns
    a side-by-side comparison dict + writes comparison_{timestamp}.html.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("monosplat.evaluation.report")

# ── Version stamp (bumped with evaluation framework changes) ──────────────────
REPORT_VERSION = "1.0.0"


# ═════════════════════════════════════════════════════════════════════════════
# EvaluationReport
# ═════════════════════════════════════════════════════════════════════════════

class EvaluationReport:
    """
    Assemble and persist a complete evaluation report for one training run.

    Usage
    -----
        report = EvaluationReport(
            run_id="run_20260603_161200",
            run_dir="/experiments/run_20260603_161200",
        )
        report.set_psnr({"mean": 28.4, "min": 24.1, "max": 32.7, "std": 1.9, "count": 50})
        report.set_ssim({"mean": 0.89, ...})
        report.set_lpips({"mean": 0.08, ...})
        report.set_fps({"primary_fps": 42.3, "n_gaussians": 210000, "device": "cuda"})
        report.set_training(duration_seconds=3612.4, iterations=15000,
                            final_loss=0.0412, n_gaussians=210000,
                            model_path="/experiments/.../final.ply")
        paths = report.save()
        # → {"json": ..., "md": ..., "html": ...}
    """

    def __init__(
        self,
        run_id: str,
        run_dir: str | Path,
        dataset_path: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id          = run_id
        self.run_dir         = Path(run_dir)
        self.dataset_path    = dataset_path
        self.config_snapshot = config_snapshot or {}

        self._generated_at = datetime.now(timezone.utc).isoformat()
        self._data: Dict[str, Any] = {}

    # ── Metric setters ─────────────────────────────────────────────────────

    def set_psnr(self, result: Dict[str, Any]) -> "EvaluationReport":
        self._data["psnr"] = result
        return self

    def set_ssim(self, result: Dict[str, Any]) -> "EvaluationReport":
        self._data["ssim"] = result
        return self

    def set_lpips(self, result: Dict[str, Any]) -> "EvaluationReport":
        self._data["lpips"] = result
        return self

    def set_fps(self, result: Dict[str, Any]) -> "EvaluationReport":
        self._data["fps"] = result
        return self

    def set_training(
        self,
        *,
        duration_seconds: Optional[float] = None,
        iterations: Optional[int] = None,
        final_loss: Optional[float] = None,
        n_gaussians: Optional[int] = None,
        model_path: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
    ) -> "EvaluationReport":
        model_size_mb = None
        if model_path:
            p = Path(model_path)
            if p.exists():
                model_size_mb = round(p.stat().st_size / 1e6, 2)

        ckpt_size_mb = None
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.exists():
                ckpt_size_mb = round(p.stat().st_size / 1e6, 2)

        self._data["training"] = {
            "duration_seconds": round(duration_seconds, 1) if duration_seconds else None,
            "duration_human":   _fmt_duration(duration_seconds),
            "iterations":       iterations,
            "final_loss":       round(final_loss, 6) if final_loss else None,
            "n_gaussians":      n_gaussians,
            "model_path":       str(model_path) if model_path else None,
            "model_size_mb":    model_size_mb,
            "checkpoint_path":  str(checkpoint_path) if checkpoint_path else None,
            "ckpt_size_mb":     ckpt_size_mb,
        }
        return self

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": REPORT_VERSION,
            "run_id":         self.run_id,
            "generated_at":   self._generated_at,
            "dataset_path":   self.dataset_path,
            "config":         self.config_snapshot,
            "metrics":        self._data,
            "summary":        self._build_summary(),
        }

    def save(self) -> Dict[str, str]:
        """
        Write all three report formats to run_dir.

        Returns
        -------
        dict with keys "json", "md", "html" — absolute paths.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()

        json_path = self._save_json(payload)
        md_path   = self._save_md(payload)
        html_path = self._save_html(payload)

        log.info("Evaluation report saved → %s", self.run_dir)
        return {
            "json": str(json_path),
            "md":   str(md_path),
            "html": str(html_path),
        }

    # ── Run comparison ─────────────────────────────────────────────────────

    @staticmethod
    def compare(
        report_dicts: List[Dict[str, Any]],
        output_dir: Optional[str | Path] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple evaluation report dicts side by side.

        Parameters
        ----------
        report_dicts : list of dicts loaded from evaluation_report.json files
        output_dir   : directory to write comparison HTML (optional)
        labels       : human-readable run labels (defaults to run_id)

        Returns
        -------
        dict with keys: labels, metrics_table, winner, html_path
        """
        if not report_dicts:
            raise ValueError("Need at least one report to compare")

        if labels is None:
            labels = [r.get("run_id", f"run_{i}") for i, r in enumerate(report_dicts)]

        # Build flat metrics table
        table: List[Dict] = []
        for label, rpt in zip(labels, report_dicts):
            m = rpt.get("metrics", {})
            row = {
                "label":            label,
                "run_id":           rpt.get("run_id", ""),
                "generated_at":     rpt.get("generated_at", ""),
                "psnr_mean":        _safe(m, "psnr", "mean"),
                "ssim_mean":        _safe(m, "ssim", "mean"),
                "lpips_mean":       _safe(m, "lpips", "mean"),
                "fps":              _safe(m, "fps", "primary_fps"),
                "n_gaussians":      _safe(m, "fps", "n_gaussians")
                                    or _safe(m, "training", "n_gaussians"),
                "duration_seconds": _safe(m, "training", "duration_seconds"),
                "duration_human":   _safe(m, "training", "duration_human"),
                "final_loss":       _safe(m, "training", "final_loss"),
                "model_size_mb":    _safe(m, "training", "model_size_mb"),
            }
            table.append(row)

        winner = _find_winner(table)

        result: Dict[str, Any] = {
            "labels":        labels,
            "metrics_table": table,
            "winner":        winner,
            "compared_at":   datetime.now(timezone.utc).isoformat(),
        }

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = out / f"comparison_{ts}.html"
            html_path.write_text(_render_comparison_html(result, table), encoding="utf-8")
            result["html_path"] = str(html_path)
            log.info("Comparison report → %s", html_path)

        return result

    # ── Private serialisers ────────────────────────────────────────────────

    def _save_json(self, payload: Dict) -> Path:
        path = self.run_dir / "evaluation_report.json"
        tmp  = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    def _save_md(self, payload: Dict) -> Path:
        path = self.run_dir / "evaluation_report.md"
        path.write_text(_render_md(payload), encoding="utf-8")
        return path

    def _save_html(self, payload: Dict) -> Path:
        path = self.run_dir / "evaluation_report.html"
        path.write_text(_render_html(payload), encoding="utf-8")
        return path

    def _build_summary(self) -> Dict[str, Any]:
        m = self._data
        return {
            "psnr_mean":        _safe(m, "psnr",     "mean"),
            "ssim_mean":        _safe(m, "ssim",     "mean"),
            "lpips_mean":       _safe(m, "lpips",    "mean"),
            "primary_fps":      _safe(m, "fps",      "primary_fps"),
            "n_gaussians":      _safe(m, "fps",      "n_gaussians")
                                or _safe(m, "training", "n_gaussians"),
            "training_seconds": _safe(m, "training", "duration_seconds"),
            "model_size_mb":    _safe(m, "training", "model_size_mb"),
            "final_loss":       _safe(m, "training", "final_loss"),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Markdown renderer
# ═════════════════════════════════════════════════════════════════════════════

def _render_md(p: Dict) -> str:
    s   = p.get("summary", {})
    m   = p.get("metrics", {})
    cfg = p.get("config", {})

    lines = [
        f"# MonoSplat Evaluation Report",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Run ID | `{p.get('run_id', 'N/A')}` |",
        f"| Generated | {p.get('generated_at', 'N/A')} |",
        f"| Report version | {p.get('report_version', 'N/A')} |",
        f"| Dataset | `{p.get('dataset_path') or 'N/A'}` |",
        f"",
        f"---",
        f"",
        f"## Image Quality Metrics",
        f"",
        f"| Metric | Mean | Min | Max | Std | Count |",
        f"|---|---|---|---|---|---|",
    ]

    for key, label in [("psnr", "PSNR (dB ↑)"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓")]:
        d = m.get(key, {})
        lines.append(
            f"| {label} | {_fmt(d.get('mean'))} | {_fmt(d.get('min'))} | "
            f"{_fmt(d.get('max'))} | {_fmt(d.get('std'))} | {d.get('count', 'N/A')} |"
        )

    lines += [
        f"",
        f"## Render Performance",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
    ]
    fps = m.get("fps", {})
    lines += [
        f"| Primary FPS (1280×720) | {_fmt(fps.get('primary_fps'))} |",
        f"| Gaussians | {fps.get('n_gaussians', 'N/A'):,}" if fps.get('n_gaussians') else f"| Gaussians | N/A |",
        f"| Device | {fps.get('device', 'N/A')} |",
        f"| VRAM (GB) | {_fmt(fps.get('vram_gb'))} |",
    ]

    if "resolutions" in fps:
        lines += [f"", f"### Per-Resolution FPS", f"", f"| Resolution | FPS | Std |", f"|---|---|---|"]
        for r in fps["resolutions"]:
            lines.append(f"| {r['width']}×{r['height']} | {_fmt(r.get('fps_mean'))} | {_fmt(r.get('fps_std'))} |")

    tr = m.get("training", {})
    lines += [
        f"",
        f"## Training Stats",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Duration | {tr.get('duration_human', 'N/A')} |",
        f"| Iterations | {tr.get('iterations', 'N/A')} |",
        f"| Final Loss | {_fmt(tr.get('final_loss'))} |",
        f"| Gaussians | {tr.get('n_gaussians', 'N/A')} |",
        f"| Model size | {_fmt(tr.get('model_size_mb'))} MB |",
        f"| Checkpoint size | {_fmt(tr.get('ckpt_size_mb'))} MB |",
        f"",
        f"---",
        f"*Generated by MonoSplat Evaluation Framework v{p.get('report_version', '?')}*",
    ]
    return "\n".join(lines) + "\n"


# ═════════════════════════════════════════════════════════════════════════════
# HTML renderer — single-file, no external deps
# ═════════════════════════════════════════════════════════════════════════════

def _render_html(p: Dict) -> str:
    s  = p.get("summary", {})
    m  = p.get("metrics", {})
    tr = m.get("training", {})
    fps_d = m.get("fps", {})

    psnr_series  = _series(m, "psnr")
    ssim_series  = _series(m, "ssim")
    lpips_series = _series(m, "lpips")
    chart_labels = [x["name"] for x in psnr_series] if psnr_series else []

    def _card(title: str, value, unit: str = "", note: str = "", arrow: str = "") -> str:
        disp = f"{value}" if value is not None else "N/A"
        col  = "#00d2a8" if arrow == "↑" else ("#ff6b6b" if arrow == "↓" else "#7ba7f7")
        return f"""
        <div class="card">
          <div class="card-label">{title} <span class="arrow">{arrow}</span></div>
          <div class="card-value" style="color:{col}">{disp}</div>
          <div class="card-unit">{unit} {note}</div>
        </div>"""

    def _res_rows() -> str:
        rows = ""
        for r in fps_d.get("resolutions", []):
            rows += f"<tr><td>{r['width']}×{r['height']}</td><td>{_fmt(r.get('fps_mean'))}</td><td>{_fmt(r.get('fps_std'))}</td></tr>"
        return rows or "<tr><td colspan='3'>N/A</td></tr>"

    def _js_array(vals) -> str:
        if not vals:
            return "[]"
        return "[" + ", ".join(f"{v['value']}" for v in vals) + "]"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MonoSplat Evaluation — {p.get('run_id','')}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2d3148;
    --text: #e8eaf6; --muted: #8892b0; --accent: #00d2a8;
    --red: #ff6b6b; --blue: #7ba7f7; --yellow: #ffd166;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); margin-bottom: .25rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: var(--blue); margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: .5rem; }}
  .meta {{ color: var(--muted); font-size: .8rem; margin-bottom: 2rem; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem 1.6rem; min-width: 160px; }}
  .card-label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
  .card-value {{ font-size: 2rem; font-weight: 700; line-height: 1.1; margin: .3rem 0; }}
  .card-unit {{ font-size: .75rem; color: var(--muted); }}
  .arrow {{ color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
  th, td {{ padding: .6rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: .75rem; text-transform: uppercase; }}
  tr:hover td {{ background: rgba(255,255,255,.03); }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; }}
  .chart-box h3 {{ font-size: .85rem; color: var(--muted); margin-bottom: .75rem; text-transform: uppercase; letter-spacing: .05em; }}
  canvas {{ max-height: 220px; }}
  footer {{ margin-top: 3rem; font-size: .75rem; color: var(--muted); text-align: center; }}
</style>
</head>
<body>

<h1>MonoSplat — Evaluation Report</h1>
<div class="meta">
  Run: <strong>{p.get('run_id','N/A')}</strong> &nbsp;|&nbsp;
  Generated: {p.get('generated_at','N/A')} &nbsp;|&nbsp;
  v{p.get('report_version','?')}
  {f"&nbsp;|&nbsp; Dataset: <code>{p.get('dataset_path')}</code>" if p.get('dataset_path') else ""}
</div>

<h2>Quality Summary</h2>
<div class="cards">
  {_card("PSNR", _fmt(s.get('psnr_mean')), "dB", "higher=better", "↑")}
  {_card("SSIM", _fmt(s.get('ssim_mean')), "", "higher=better", "↑")}
  {_card("LPIPS", _fmt(s.get('lpips_mean')), "", "lower=better", "↓")}
  {_card("Render FPS", _fmt(s.get('primary_fps')), "fps @ 1280×720", "", "")}
  {_card("Gaussians", f"{s.get('n_gaussians') or 'N/A':,}" if s.get('n_gaussians') else "N/A", "", "", "")}
  {_card("Train Time", tr.get('duration_human','N/A'), "", "", "")}
  {_card("Model Size", _fmt(s.get('model_size_mb')), "MB", "", "")}
  {_card("Final Loss", _fmt(s.get('final_loss')), "", "", "")}
</div>

<h2>Per-Image Metrics</h2>
<div class="chart-grid">
  <div class="chart-box"><h3>PSNR per frame (dB ↑)</h3><canvas id="psnrChart"></canvas></div>
  <div class="chart-box"><h3>SSIM per frame ↑</h3><canvas id="ssimChart"></canvas></div>
  <div class="chart-box"><h3>LPIPS per frame ↓</h3><canvas id="lpipsChart"></canvas></div>
  <div class="chart-box"><h3>Render FPS by resolution</h3><canvas id="fpsChart"></canvas></div>
</div>

<h2>Image Quality Detail</h2>
<table>
  <tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Std</th><th>Count</th></tr>
  {_metric_row(m, "psnr",  "PSNR (dB ↑)")}
  {_metric_row(m, "ssim",  "SSIM ↑")}
  {_metric_row(m, "lpips", "LPIPS ↓")}
</table>

<h2>Render Performance</h2>
<table>
  <tr><th>Resolution</th><th>FPS (mean)</th><th>FPS (std)</th></tr>
  {_res_rows()}
</table>
<p style="margin-top:.75rem;font-size:.8rem;color:var(--muted)">
  Device: {fps_d.get('device','N/A')} &nbsp;|&nbsp;
  VRAM: {_fmt(fps_d.get('vram_gb'))} GB
</p>

<h2>Training Details</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Duration</td><td>{tr.get('duration_human','N/A')}</td></tr>
  <tr><td>Iterations</td><td>{tr.get('iterations','N/A')}</td></tr>
  <tr><td>Final Loss</td><td>{_fmt(tr.get('final_loss'))}</td></tr>
  <tr><td>Gaussians</td><td>{tr.get('n_gaussians','N/A')}</td></tr>
  <tr><td>Model size</td><td>{_fmt(tr.get('model_size_mb'))} MB</td></tr>
  <tr><td>Checkpoint size</td><td>{_fmt(tr.get('ckpt_size_mb'))} MB</td></tr>
  <tr><td>Model path</td><td><code>{tr.get('model_path') or 'N/A'}</code></td></tr>
</table>

<footer>Generated by MonoSplat Evaluation Framework v{p.get('report_version','?')}</footer>

<script>
const LABELS = {json.dumps(chart_labels)};
const psnrData  = {_js_array(psnr_series)};
const ssimData  = {_js_array(ssim_series)};
const lpipsData = {_js_array(lpips_series)};

const fpsLabels = {json.dumps([f"{r['width']}x{r['height']}" for r in fps_d.get('resolutions',[])])};
const fpsData   = {json.dumps([r.get('fps_mean') for r in fps_d.get('resolutions',[])])};

function mkLine(ctx, data, label, color) {{
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels: LABELS, datasets: [{{ label, data, borderColor: color,
      backgroundColor: color+'22', borderWidth:1.5, pointRadius:0, tension:0.3 }}] }},
    options: {{ responsive:true, plugins:{{ legend:{{display:false}} }},
      scales:{{ x:{{display:false}}, y:{{grid:{{color:'#2d3148'}},
        ticks:{{color:'#8892b0',font:{{size:10}}}} }} }} }}
  }});
}}
function mkBar(ctx, labels, data, label, color) {{
  new Chart(ctx, {{
    type: 'bar',
    data: {{ labels, datasets: [{{ label, data, backgroundColor: color+'99',
      borderColor: color, borderWidth:1 }}] }},
    options: {{ responsive:true, plugins:{{ legend:{{display:false}} }},
      scales:{{ x:{{ticks:{{color:'#8892b0',font:{{size:10}}}},grid:{{display:false}}}},
        y:{{grid:{{color:'#2d3148'}},ticks:{{color:'#8892b0',font:{{size:10}}}}}} }} }}
  }});
}}
if(psnrData.length)  mkLine(document.getElementById('psnrChart').getContext('2d'), psnrData, 'PSNR', '#00d2a8');
if(ssimData.length)  mkLine(document.getElementById('ssimChart').getContext('2d'), ssimData, 'SSIM', '#7ba7f7');
if(lpipsData.length) mkLine(document.getElementById('lpipsChart').getContext('2d'), lpipsData, 'LPIPS', '#ff6b6b');
if(fpsData.length)   mkBar(document.getElementById('fpsChart').getContext('2d'), fpsLabels, fpsData, 'FPS', '#ffd166');
</script>
</body></html>"""


# ═════════════════════════════════════════════════════════════════════════════
# Comparison HTML renderer
# ═════════════════════════════════════════════════════════════════════════════

def _render_comparison_html(result: Dict, table: List[Dict]) -> str:
    labels   = result.get("labels", [])
    winner   = result.get("winner", {})
    compared = result.get("compared_at", "")

    def _col(row, key, winner_label, is_higher_better: bool) -> str:
        val = row.get(key)
        disp = _fmt(val) if val is not None else "N/A"
        is_best = winner.get(key) == row["label"]
        color = "#00d2a8" if is_best else "inherit"
        badge = " 🏆" if is_best else ""
        return f'<td style="color:{color}">{disp}{badge}</td>'

    header_cells = "".join(f"<th>{l}</th>" for l in labels)
    rows_html = ""
    metrics = [
        ("psnr_mean",        "PSNR mean (dB ↑)",         True),
        ("ssim_mean",        "SSIM mean ↑",               True),
        ("lpips_mean",       "LPIPS mean ↓",              False),
        ("fps",              "Primary FPS ↑",             True),
        ("n_gaussians",      "Gaussians",                 False),
        ("duration_seconds", "Training time (s)",         False),
        ("final_loss",       "Final loss ↓",              False),
        ("model_size_mb",    "Model size (MB)",           False),
    ]
    for key, label, higher in metrics:
        cells = "".join(_col(row, key, winner.get(key), higher) for row in table)
        rows_html += f"<tr><td><strong>{label}</strong></td>{cells}</tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MonoSplat Run Comparison</title>
<style>
  body {{ background:#0f1117; color:#e8eaf6; font-family:'Segoe UI',system-ui,sans-serif; padding:2rem; }}
  h1 {{ color:#00d2a8; margin-bottom:.25rem; }}
  .meta {{ color:#8892b0; font-size:.8rem; margin-bottom:2rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:.875rem; }}
  th,td {{ padding:.65rem 1.1rem; border-bottom:1px solid #2d3148; text-align:left; }}
  th {{ color:#8892b0; font-size:.75rem; text-transform:uppercase; }}
  tr:hover td {{ background:rgba(255,255,255,.03); }}
</style>
</head>
<body>
<h1>MonoSplat — Run Comparison</h1>
<div class="meta">Compared at: {compared} &nbsp;|&nbsp; {len(labels)} runs</div>
<table>
  <tr><th>Metric</th>{header_cells}</tr>
  {rows_html}
</table>
<p style="margin-top:2rem;font-size:.75rem;color:#8892b0">
  Generated by MonoSplat Evaluation Framework v{REPORT_VERSION}
</p>
</body></html>"""


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _safe(d: Dict, *keys) -> Any:
    node = d
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _metric_row(m: Dict, key: str, label: str) -> str:
    d = m.get(key, {})
    return (
        f"<tr><td>{label}</td>"
        f"<td>{_fmt(d.get('mean'))}</td>"
        f"<td>{_fmt(d.get('min'))}</td>"
        f"<td>{_fmt(d.get('max'))}</td>"
        f"<td>{_fmt(d.get('std'))}</td>"
        f"<td>{d.get('count','N/A')}</td></tr>"
    )


def _series(m: Dict, key: str) -> List[Dict]:
    """Extract per_image list as [{name, value}] for charting."""
    per = m.get(key, {}).get("per_image", [])
    return [{"name": row.get("name", ""), "value": row.get(key)} for row in per]


def _find_winner(table: List[Dict]) -> Dict[str, str]:
    """For each metric, return the label of the best-performing run."""
    winner: Dict[str, str] = {}
    higher_better = {"psnr_mean", "ssim_mean", "fps"}
    lower_better  = {"lpips_mean", "duration_seconds", "final_loss", "model_size_mb"}

    for key in higher_better | lower_better:
        best_label = None
        best_val   = None
        for row in table:
            val = row.get(key)
            if val is None:
                continue
            if best_val is None:
                best_val   = val
                best_label = row["label"]
            elif key in higher_better and val > best_val:
                best_val   = val
                best_label = row["label"]
            elif key in lower_better and val < best_val:
                best_val   = val
                best_label = row["label"]
        if best_label:
            winner[key] = best_label

    return winner
