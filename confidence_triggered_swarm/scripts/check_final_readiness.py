#!/usr/bin/env python3
"""Check whether final-report and presentation artifacts are present.

This is a lightweight submission readiness check. It does not judge scientific
quality; it verifies that the canonical files used by the report and slides are
available and flags known manual tasks.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = ROOT / "final_submission"
CANONICAL_DECK = FINAL_DIR / "confidence_triggered_swarm_premium.pptx"
EXPECTED_DECK_SLIDES = 10


def _status(label: str, ok: bool, detail: str, warn: bool = False) -> Tuple[str, str]:
    if ok:
        return "OK", f"[OK]   {label}: {detail}"
    if warn:
        return "WARN", f"[WARN] {label}: {detail}"
    return "FAIL", f"[FAIL] {label}: {detail}"


def _exists(path: Path, label: str, warn: bool = False) -> Tuple[str, str]:
    rel = path.relative_to(ROOT)
    return _status(label, path.exists(), str(rel), warn=warn)


def _all_exist(paths: Iterable[Path], label: str) -> Tuple[str, str]:
    missing = [p.relative_to(ROOT) for p in paths if not p.exists()]
    if missing:
        return "FAIL", f"[FAIL] {label}: missing {', '.join(map(str, missing))}"
    return "OK", f"[OK]   {label}: all present"


def _style_available() -> Tuple[str, str]:
    local_style = FINAL_DIR / "neurips_2026.sty"
    kpsewhich = shutil.which("kpsewhich")
    tex_path = False
    if kpsewhich:
        result = subprocess.run(
            [kpsewhich, "neurips_2026.sty"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        tex_path = result.returncode == 0 and bool(result.stdout.strip())
    ok = local_style.exists() or tex_path
    detail = "found" if ok else "place official neurips_2026.sty in final_submission/"
    return _status("NeurIPS style", ok, detail, warn=True)


def _author_contrib_done() -> Tuple[str, str]:
    report = FINAL_DIR / "final_report.tex"
    if not report.exists():
        return "FAIL", "[FAIL] author contributions: final_report.tex missing"
    text = report.read_text(encoding="utf-8")
    todo_tokens = ("TODO_AUTHOR_CONTRIBUTIONS", r"TODO\_AUTHOR\_CONTRIBUTIONS")
    done = not any(token in text for token in todo_tokens)
    detail = "filled" if done else "replace TODO_AUTHOR_CONTRIBUTIONS before submission"
    return _status("author contributions", done, detail, warn=True)


def _pptx_slide_count() -> Tuple[str, str]:
    deck = CANONICAL_DECK
    if not deck.exists():
        rel = deck.relative_to(ROOT)
        return "FAIL", f"[FAIL] presentation deck: pptx missing ({rel})"
    try:
        with zipfile.ZipFile(deck) as zf:
            slides = [
                name
                for name in zf.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ]
    except zipfile.BadZipFile:
        return "FAIL", "[FAIL] presentation deck: pptx is not a valid zip archive"
    ok = len(slides) == EXPECTED_DECK_SLIDES
    detail = (
        f"{len(slides)} slides; expected {EXPECTED_DECK_SLIDES} "
        "for the premium 10-12 minute talk"
    )
    return _status("presentation deck", ok, detail)


def _pdf_page_check() -> Tuple[str, str]:
    pdf = FINAL_DIR / "final_report.pdf"
    if not pdf.exists():
        return _status(
            "PDF page count",
            False,
            "final_report.pdf not built yet; compile and check 5-9 content pages",
            warn=True,
        )
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return _status(
            "PDF page count",
            False,
            "pdfinfo not installed; check 5-9 content pages manually",
            warn=True,
        )
    result = subprocess.run(
        [pdfinfo, str(pdf)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return _status("PDF page count", False, "pdfinfo failed", warn=True)
    pages = None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
            break
    if pages is None:
        return _status("PDF page count", False, "could not read page count", warn=True)
    ok = 5 <= pages <= 9
    return _status("PDF page count", ok, f"{pages} pages", warn=True)


def main() -> int:
    checks: List[Tuple[str, str]] = []

    required_docs = [
        ROOT / "README.md",
        ROOT / "REPORT.md",
        ROOT / "docs" / "artifact_guide.md",
        ROOT / "docs" / "professor_brief.md",
        ROOT / "docs" / "professor_demo_notes.md",
        ROOT / "docs" / "final_submission_audit.md",
        FINAL_DIR / "README.md",
        FINAL_DIR / "final_report.tex",
        FINAL_DIR / "references.bib",
        FINAL_DIR / "slides_10_12min_outline.md",
        FINAL_DIR / "slide_speaker_notes.md",
        CANONICAL_DECK,
    ]
    required_results = [
        ROOT / "runs" / "baseline" / "best_model.pt",
        ROOT / "runs" / "full_eval" / "evaluation_results.json",
        ROOT / "runs" / "continual_run" / "continual_results.json",
        ROOT / "runs" / "ablations" / "ablation_results.json",
        ROOT / "runs" / "professor_ready" / "README.md",
    ]
    figure_paths = [
        ROOT / "runs" / "professor_ready" / "fig1_frozen_vs_lifelong.png",
        ROOT / "runs" / "professor_ready" / "fig1_frozen_vs_lifelong.pdf",
        ROOT / "runs" / "professor_ready" / "fig2_degradation.png",
        ROOT / "runs" / "professor_ready" / "fig2_degradation.pdf",
        ROOT / "runs" / "professor_ready" / "fig3_ablations.png",
        ROOT / "runs" / "professor_ready" / "fig3_ablations.pdf",
        ROOT / "runs" / "professor_ready" / "fig4_forgetting.png",
        ROOT / "runs" / "professor_ready" / "fig4_forgetting.pdf",
        ROOT / "runs" / "professor_ready" / "fig5_training_over_time.png",
        ROOT / "runs" / "professor_ready" / "fig5_training_over_time.pdf",
        ROOT / "runs" / "professor_ready" / "fig6_continual_matrix.png",
        ROOT / "runs" / "professor_ready" / "fig6_continual_matrix.pdf",
        ROOT / "runs" / "professor_ready" / "fig7_clean_retention.png",
        ROOT / "runs" / "professor_ready" / "fig7_clean_retention.pdf",
        ROOT / "runs" / "professor_ready" / "fig8_cl_metrics.png",
        ROOT / "runs" / "professor_ready" / "fig8_cl_metrics.pdf",
    ]

    checks.append(_all_exist(required_docs, "submission docs"))
    checks.append(_all_exist(required_results, "canonical result artifacts"))
    checks.append(_all_exist(figure_paths, "presentation/report figures"))
    checks.append(_pptx_slide_count())
    checks.append(_style_available())
    checks.append(_author_contrib_done())
    checks.append(_pdf_page_check())

    for _status_code, line in checks:
        print(line)

    failures = [code for code, _ in checks if code == "FAIL"]
    warnings = [code for code, _ in checks if code == "WARN"]
    print()
    if failures:
        print(f"Final readiness: FAIL ({len(failures)} blocking issue(s))")
        return 1
    if warnings:
        print(f"Final readiness: WARN ({len(warnings)} manual item(s) remain)")
        return 0
    print("Final readiness: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
