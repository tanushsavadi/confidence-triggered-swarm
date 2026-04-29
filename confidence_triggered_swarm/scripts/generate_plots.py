#!/usr/bin/env python3
"""Generate plots from saved evaluation JSON artifacts.

This script intentionally reads the stored experiment outputs instead of using
hardcoded report numbers. By default it writes professor-ready figures to
`runs/professor_ready/` so existing draft figures are left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_RESULTS = BASE_DIR / "runs" / "full_eval" / "evaluation_results.json"
DEFAULT_ABLATION_RESULTS = BASE_DIR / "runs" / "ablations" / "ablation_results.json"
DEFAULT_OUT_DIR = BASE_DIR / "runs" / "professor_ready"
DEFAULT_SEVERITIES = ["clean", "mild", "moderate", "severe"]

C_FROZEN = "#4878CF"
C_ADAPTED = "#E8833A"
C_GREEN = "#6ACC65"
C_RED = "#D65F5F"
C_PURPLE = "#956CB4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data-driven plots from saved evaluation JSON"
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=DEFAULT_EVAL_RESULTS,
        help="Path to evaluation_results.json",
    )
    parser.add_argument(
        "--ablation-results",
        type=Path,
        default=DEFAULT_ABLATION_RESULTS,
        help="Path to ablation_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated PNG/PDF figures",
    )
    parser.add_argument(
        "--continual-results",
        type=Path,
        default=None,
        help="Optional continual_results.json from train_continual.py (fig5–fig8)",
    )
    return parser.parse_args()


def _configure_matplotlib() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    with path.open() as f:
        return json.load(f)


def _extract_comparison(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    comparison = eval_results.get("comparison")
    if comparison:
        return comparison

    baseline = eval_results.get("baseline", {})
    lifelong = eval_results.get("lifelong", {})
    fallback: Dict[str, Any] = {}
    for severity in DEFAULT_SEVERITIES:
        if severity not in baseline or severity not in lifelong:
            continue
        bl = baseline[severity]
        ll = lifelong[severity]
        bl_reward = float(bl.get("mean_reward", 0.0))
        ll_reward = float(ll.get("mean_reward", 0.0))
        fallback[severity] = {
            "baseline_reward": bl_reward,
            "lifelong_reward": ll_reward,
            "reward_improvement_pct": (
                (ll_reward - bl_reward) / abs(bl_reward) * 100.0
                if abs(bl_reward) > 1e-8
                else 0.0
            ),
            "baseline_waypoints": float(bl.get("mean_waypoints_reached", 0.0)),
            "lifelong_waypoints": float(ll.get("mean_waypoints_reached", 0.0)),
            "adaptation_rate": float(ll.get("adaptation_rate", 0.0)),
            "total_adaptations": int(ll.get("total_adaptations", 0)),
            "mean_confidence": float(ll.get("mean_confidence", 0.0)),
        }
    return fallback


def _severity_order(comparison: Dict[str, Any]) -> List[str]:
    available = [sev for sev in DEFAULT_SEVERITIES if sev in comparison]
    return available or list(comparison.keys())


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  saved {path}")
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars: Any, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        offset = 4 if height >= 0 else -12
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
        )


def fig1_frozen_vs_lifelong(
    comparison: Dict[str, Any], severities: List[str], out_dir: Path
) -> None:
    print("Fig 1: Frozen vs Lifelong...")
    frozen = [comparison[s]["baseline_reward"] for s in severities]
    lifelong = [comparison[s]["lifelong_reward"] for s in severities]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(severities))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        frozen,
        width,
        label="Frozen Policy",
        color=C_FROZEN,
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        lifelong,
        width,
        label="Lifelong Policy",
        color=C_ADAPTED,
        edgecolor="white",
        linewidth=0.5,
    )

    _annotate_bars(ax, bars1, "{:.1f}")
    _annotate_bars(ax, bars2, "{:.1f}")

    ax.set_xlabel("Surprise Severity")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Saved Evaluation Results: Frozen vs Lifelong Reward")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in severities])
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_frozen_vs_lifelong")


def fig2_degradation(
    comparison: Dict[str, Any], severities: List[str], out_dir: Path
) -> None:
    print("Fig 2: Degradation...")
    frozen = [comparison[s]["baseline_reward"] for s in severities]
    lifelong = [comparison[s]["lifelong_reward"] for s in severities]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(severities))
    ax.plot(
        x,
        frozen,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=C_FROZEN,
        label="Frozen Policy",
        zorder=3,
    )
    ax.plot(
        x,
        lifelong,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color=C_ADAPTED,
        label="Lifelong Policy",
        zorder=3,
    )
    ax.fill_between(x, frozen, lifelong, alpha=0.12, color=C_ADAPTED)

    for idx, severity in enumerate(severities):
        base = frozen[idx]
        if abs(base) <= 1e-8:
            continue
        delta_pct = (lifelong[idx] - base) / abs(base) * 100.0
        mid_y = (frozen[idx] + lifelong[idx]) / 2
        ax.annotate(
            f"{delta_pct:+.0f}%",
            xy=(x[idx] + 0.05, mid_y),
            fontsize=10,
            fontweight="bold",
            color=C_ADAPTED if delta_pct >= 0 else C_RED,
        )

    ax.set_xlabel("Surprise Severity")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Saved Evaluation Results: Reward by Severity")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in severities])
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_degradation")


def fig3_ablations(ablations: Dict[str, Any], out_dir: Path) -> None:
    print("Fig 3: Ablations...")
    order = [
        ("frozen_severe", "Frozen (no adapt)", C_FROZEN),
        ("full_method", "Full Method", C_ADAPTED),
        ("no_kl_anchor", "No KL Anchoring", C_GREEN),
        ("no_clean_replay", "No Clean Replay", C_RED),
        ("no_ewc", "No EWC", C_PURPLE),
    ]

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    colors: List[str] = []
    for key, label, color in order:
        if key not in ablations:
            continue
        labels.append(label)
        means.append(float(ablations[key].get("mean_reward", 0.0)))
        stds.append(float(ablations[key].get("std_reward", 0.0)))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        means,
        xerr=stds,
        height=0.6,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )

    for bar, mean, std in zip(bars, means, stds):
        width = bar.get_width()
        ax.text(
            width + std + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f} ± {std:.1f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Episode Reward (Severe Surprise)")
    ax.set_title("Saved Ablation Results")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out_dir, "fig3_ablations")


def _rolling_mean(x: List[float], window: int) -> np.ndarray:
    """Causal rolling mean (same length as x)."""
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    w = max(1, min(window, n))
    out = np.empty(n, dtype=float)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    for i in range(n):
        start = max(0, i - w + 1)
        out[i] = (csum[i + 1] - csum[start]) / (i + 1 - start)
    return out


def fig5_training_over_time(continual: Dict[str, Any], out_dir: Path) -> None:
    print("Fig 5: Training over time (per-episode reward)...")
    per = continual.get("per_episode_rewards", {})
    ll = per.get("lifelong", [])
    fr = per.get("frozen", [])
    boundaries = continual.get("phase_boundaries", [])
    phases = continual.get("phases", DEFAULT_SEVERITIES)

    fig, ax = plt.subplots(figsize=(10, 5))
    ep_idx = np.arange(1, len(ll) + 1)
    win = max(5, len(ll) // 25) if ll else 5

    if ll:
        ax.plot(ep_idx, ll, color=C_ADAPTED, alpha=0.35, linewidth=0.8, label="Lifelong (raw)")
        ax.plot(ep_idx, _rolling_mean(ll, win), color=C_ADAPTED, linewidth=2.2, label="Lifelong (rolling mean)")
    if fr:
        ax.plot(np.arange(1, len(fr) + 1), fr, color=C_FROZEN, alpha=0.35, linewidth=0.8, label="Frozen (raw)")
        ax.plot(
            np.arange(1, len(fr) + 1),
            _rolling_mean(fr, win),
            color=C_FROZEN,
            linewidth=2.2,
            linestyle="--",
            label="Frozen (rolling mean)",
        )

    ax.relim()
    ax.autoscale_view()
    ymax = float(max(ax.get_ylim()[1], 1.0))
    prev = 0
    for bi, b in enumerate(boundaries):
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.6)
        mid = (prev + b) / 2 if b > prev else b / 2
        label_txt = phases[bi] if bi < len(phases) else str(bi)
        ax.text(mid, ymax * 0.92, label_txt, ha="center", fontsize=10, fontweight="bold")
        prev = b

    ax.set_xlabel("Episode (sequential phases)")
    ax.set_ylabel("Episode reward")
    ax.set_title("Continual Learning: Reward Over Time (Sequential Phases)")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_training_over_time")


def fig6_continual_matrix(continual: Dict[str, Any], out_dir: Path) -> None:
    print("Fig 6: Continual matrix (heatmaps)...")
    R_ll = np.asarray(continual.get("R_lifelong", []), dtype=float)
    R_fr = np.asarray(continual.get("R_frozen", []), dtype=float)
    phases = continual.get("phases", DEFAULT_SEVERITIES)
    if R_ll.size == 0:
        print("  [skip] No R_lifelong in continual JSON")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(float(np.nanmax(R_ll)), float(np.nanmax(R_fr)), 1.0)
    vmin = min(float(np.nanmin(R_ll)), float(np.nanmin(R_fr)), 0.0)

    for ax, R, title in zip(
        axes,
        (R_ll, R_fr),
        ("Lifelong (after sequential adaptation)", "Frozen baseline (same eval schedule)"),
    ):
        im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels([p.capitalize() for p in phases])
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels([f"After {p}" for p in phases])
        ax.set_xlabel("Eval task")
        ax.set_ylabel("Training phase completed")
        ax.set_title(title)
        mid = (vmin + vmax) / 2
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                val = R[i, j]
                tc = "w" if val < mid else "k"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=tc, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Retroactive reward matrix R[i, j]", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_continual_matrix")


def fig7_clean_retention(continual: Dict[str, Any], out_dir: Path) -> None:
    print("Fig 7: Clean retention...")
    R_ll = np.asarray(continual.get("R_lifelong", []), dtype=float)
    R_fr = np.asarray(continual.get("R_frozen", []), dtype=float)
    phases = continual.get("phases", DEFAULT_SEVERITIES)
    if R_ll.size == 0:
        print("  [skip]")
        return
    clean_col = 0
    x = np.arange(len(phases))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, R_ll[:, clean_col], marker="o", linewidth=2, color=C_ADAPTED, label="Lifelong")
    fr_clean = continual.get("metrics", {}).get("frozen", {}).get("clean_retention")
    if fr_clean is not None and len(fr_clean) == len(phases):
        ax.plot(x, fr_clean, marker="s", linewidth=2, color=C_FROZEN, linestyle="--", label="Frozen (no adaptation)")
    else:
        ax.plot(
            x,
            R_fr[:, clean_col],
            marker="s",
            linewidth=2,
            color=C_FROZEN,
            linestyle="--",
            label="Frozen",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"After {p}" for p in phases])
    ax.set_ylabel("Mean reward on clean")
    ax.set_xlabel("Phase completed")
    ax.set_title("Clean-task performance after each phase (forgetting check)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig7_clean_retention")


def fig8_cl_metrics(continual: Dict[str, Any], out_dir: Path) -> None:
    print("Fig 8: CL metrics bars...")
    m = continual.get("metrics", {})
    ll = m.get("lifelong", {})
    fr = m.get("frozen", {})
    keys = [
        ("average_reward", "Avg reward"),
        ("backward_transfer", "BWT"),
        ("forward_transfer", "FWT"),
        ("remembering", "Remembering"),
    ]
    labels = [lbl for _, lbl in keys]
    v_ll = [float(ll.get(k, 0.0)) for k, _ in keys]
    v_fr = [float(fr.get(k, 0.0)) for k, _ in keys]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, v_fr, width, label="Frozen", color=C_FROZEN, edgecolor="white")
    ax.bar(x + width / 2, v_ll, width, label="Lifelong", color=C_ADAPTED, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.set_title("Continual learning diagnostics (from reward matrix)")
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_cl_metrics")


def fig4_forgetting(eval_results: Dict[str, Any], out_dir: Path) -> None:
    print("Fig 4: Forgetting...")
    forgetting = eval_results.get("forgetting", {})
    baseline_clean = forgetting.get("baseline_clean", {})
    adapted_clean = forgetting.get("adapted_clean", {})

    rewards = [
        float(baseline_clean.get("mean_reward", 0.0)),
        float(adapted_clean.get("mean_reward", 0.0)),
    ]
    waypoints = [
        float(baseline_clean.get("mean_waypoints_reached", 0.0)),
        float(adapted_clean.get("mean_waypoints_reached", 0.0)),
    ]
    conditions = ["Before\nAdaptation", "After\nAdaptation"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    ax = axes[0]
    bars = ax.bar(
        conditions,
        rewards,
        width=0.5,
        color=[C_FROZEN, C_ADAPTED],
        edgecolor="white",
        linewidth=0.5,
    )
    _annotate_bars(ax, bars, "{:.1f}")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Clean Reward")

    ax = axes[1]
    bars = ax.bar(
        conditions,
        waypoints,
        width=0.5,
        color=[C_FROZEN, C_ADAPTED],
        edgecolor="white",
        linewidth=0.5,
    )
    _annotate_bars(ax, bars, "{:.2f}")
    ax.set_ylabel("Mean Waypoints Reached")
    ax.set_title("Clean Waypoints")

    fig.suptitle("Saved Forgetting Check", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_forgetting")


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    eval_results = _load_json(args.evaluation_results)
    ablation_results = _load_json(args.ablation_results)
    comparison = _extract_comparison(eval_results)
    severities = _severity_order(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation JSON: {args.evaluation_results}")
    print(f"Ablation JSON:   {args.ablation_results}")
    print(f"Output dir:      {args.output_dir}")
    if args.continual_results:
        print(f"Continual JSON:  {args.continual_results}")
    print()

    fig1_frozen_vs_lifelong(comparison, severities, args.output_dir)
    fig2_degradation(comparison, severities, args.output_dir)
    fig3_ablations(ablation_results, args.output_dir)
    fig4_forgetting(eval_results, args.output_dir)

    if args.continual_results and args.continual_results.exists():
        continual = _load_json(args.continual_results)
        fig5_training_over_time(continual, args.output_dir)
        fig6_continual_matrix(continual, args.output_dir)
        fig7_clean_retention(continual, args.output_dir)
        fig8_cl_metrics(continual, args.output_dir)
    elif args.continual_results:
        print(
            f"[WARNING] Continual results not found: {args.continual_results} — skipping fig5–fig8"
        )

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
