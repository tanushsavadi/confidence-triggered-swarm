"""Run and summarize the extension experiment matrix.

This script is intentionally orchestration-only: it calls the existing training
and evaluation entry points, stores each run in a predictable folder, and builds
aggregate summaries from the JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "confidence_triggered_swarm" / "configs"
DEFAULT_BASELINE = "runs/baseline/best_model.pt"
DEFAULT_SEEDS = [42, 123, 456]
SEVERITIES = ["clean", "mild", "moderate", "severe"]

METHODS: Dict[str, Dict[str, Any]] = {
    "frozen": {"mode": "frozen", "config": None},
    "current_default": {"mode": "lifelong", "config": None},
    "improved_ppo": {"mode": "lifelong", "config": CONFIG_DIR / "improved_ppo.yaml"},
    "improved_ppo_moderate": {
        "mode": "lifelong",
        "config": CONFIG_DIR / "improved_ppo_moderate.yaml",
    },
    "reward_weighted_rescue": {
        "mode": "lifelong",
        "config": CONFIG_DIR / "improved_reward_weighted_rescue.yaml",
    },
    "always_adapt": {"mode": "lifelong", "config": CONFIG_DIR / "always_adapt.yaml"},
}

SCREENING_METHODS = [
    "frozen",
    "current_default",
    "improved_ppo",
    "improved_ppo_moderate",
    "reward_weighted_rescue",
    "always_adapt",
]
VALIDATION_BASE_METHODS = ["frozen", "current_default", "always_adapt"]
TUNED_CANDIDATES = [
    "current_default",
    "improved_ppo",
    "improved_ppo_moderate",
    "reward_weighted_rescue",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extension experiment stages")
    parser.add_argument(
        "action",
        choices=[
            "screening",
            "validation",
            "continual",
            "domain-randomized",
            "summarize",
            "all",
        ],
    )
    parser.add_argument("--save-root", type=Path, default=Path("runs/extension"))
    parser.add_argument("--baseline-path", type=str, default=DEFAULT_BASELINE)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--screening-episodes", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=75)
    parser.add_argument("--continual-adapt-episodes", type=int, default=50)
    parser.add_argument("--continual-eval-episodes", type=int, default=30)
    parser.add_argument("--tuned-method", type=str, default="auto")
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--domain-timesteps", type=int, default=1_000_000)
    parser.add_argument("--domain-save-dir", type=Path, default=None)
    parser.add_argument("--domain-baseline-path", type=str, default=None)
    return parser.parse_args()


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str) -> List[int]:
    return [int(item) for item in _split_csv(value)]


def _result_filename(method: str) -> str:
    return "frozen_results.json" if METHODS[method]["mode"] == "frozen" else "lifelong_results.json"


def _run(cmd: Sequence[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(printable)
    if not dry_run:
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _method_command(
    method: str,
    seed: int,
    episodes: int,
    out_dir: Path,
    baseline_path: str,
) -> List[str]:
    spec = METHODS[method]
    cmd = [
        sys.executable,
        "-m",
        "confidence_triggered_swarm.scripts.evaluate",
        "--baseline-path",
        baseline_path,
        "--mode",
        spec["mode"],
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--save-dir",
        str(out_dir),
    ]
    if spec["config"] is not None:
        cmd.extend(["--config", str(spec["config"])])
    return cmd


def run_eval_stage(
    stage: str,
    methods: Iterable[str],
    seeds: Iterable[int],
    episodes: int,
    save_root: Path,
    baseline_path: str,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"Unknown method: {method}")
        for seed in seeds:
            out_dir = save_root / stage / method / f"seed_{seed}"
            result_path = out_dir / _result_filename(method)
            if skip_existing and result_path.exists():
                print(f"skip existing: {result_path}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            _run(_method_command(method, seed, episodes, out_dir, baseline_path), dry_run)


def run_continual_stage(
    methods: Iterable[str],
    seeds: Iterable[int],
    save_root: Path,
    baseline_path: str,
    n_adapt: int,
    n_eval: int,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    for method in methods:
        if method not in METHODS or METHODS[method]["mode"] == "frozen":
            raise SystemExit(f"Continual stage needs a lifelong method, got: {method}")
        for seed in seeds:
            out_dir = save_root / "continual" / method / f"seed_{seed}"
            result_path = out_dir / "continual_results.json"
            if skip_existing and result_path.exists():
                print(f"skip existing: {result_path}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "confidence_triggered_swarm.scripts.train_continual",
                "--baseline-path",
                baseline_path,
                "--n-adapt-episodes",
                str(n_adapt),
                "--n-eval-episodes",
                str(n_eval),
                "--seed",
                str(seed),
                "--save-dir",
                str(out_dir),
            ]
            config = METHODS[method]["config"]
            if config is not None:
                cmd.extend(["--config", str(config)])
            _run(cmd, dry_run)


def run_domain_randomized(args: argparse.Namespace) -> None:
    domain_dir = args.domain_save_dir or args.save_root / "domain_randomized" / "seed_42"
    model_path = args.domain_baseline_path
    if model_path is None:
        model_path = str(domain_dir / "best_model.pt")
        if not (args.skip_existing and Path(model_path).exists()):
            domain_dir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    sys.executable,
                    "-m",
                    "confidence_triggered_swarm.scripts.train_domain_randomized",
                    "--timesteps",
                    str(args.domain_timesteps),
                    "--seed",
                    "42",
                    "--save-dir",
                    str(domain_dir),
                ],
                args.dry_run,
            )

    eval_dir = args.save_root / "domain_randomized_eval" / "frozen" / "seed_42"
    result_path = eval_dir / "frozen_results.json"
    if args.skip_existing and result_path.exists():
        print(f"skip existing: {result_path}")
        return
    eval_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "confidence_triggered_swarm.scripts.evaluate",
            "--baseline-path",
            model_path,
            "--mode",
            "frozen",
            "--episodes",
            str(args.episodes or args.validation_episodes),
            "--seed",
            "42",
            "--save-dir",
            str(eval_dir),
            "--config",
            str(CONFIG_DIR / "domain_randomized.yaml"),
        ],
        args.dry_run,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_eval_payload(path: Path) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    payload = _load_json(path)
    if "baseline" in payload:
        return "frozen", payload["baseline"], payload.get("summary", {})
    if "lifelong" in payload:
        return "lifelong", payload["lifelong"], payload.get("summary", {})
    raise ValueError(f"Unrecognized eval JSON: {path}")


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(statistics.stdev(values))


def _sem(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float((_std(values) or 0.0) / math.sqrt(len(values)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def collect_eval_rows(stage_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(stage_dir.glob("*/*/*_results.json")):
        method = path.parents[1].name
        seed_name = path.parent.name
        if not seed_name.startswith("seed_"):
            continue
        seed = int(seed_name.replace("seed_", ""))
        mode, by_severity, summary = _extract_eval_payload(path)
        for severity, result in by_severity.items():
            all_rewards = [float(v) for v in result.get("all_rewards", [])]
            all_conf = [float(v) for v in result.get("all_confidences", [])]
            rows.append(
                {
                    "stage": stage_dir.name,
                    "method": method,
                    "mode": mode,
                    "seed": seed,
                    "severity": severity,
                    "mean_reward": float(result.get("mean_reward", 0.0)),
                    "std_reward": float(result.get("std_reward", 0.0)),
                    "sem_episode_reward": (
                        float(result.get("std_reward", 0.0))
                        / math.sqrt(max(int(result.get("n_episodes", len(all_rewards) or 1)), 1))
                    ),
                    "mean_waypoints_reached": float(
                        result.get("mean_waypoints_reached", 0.0)
                    ),
                    "success_rate": float(result.get("success_rate", 0.0)),
                    "mean_confidence": (
                        float(result["mean_confidence"])
                        if result.get("mean_confidence") is not None
                        else None
                    ),
                    "adaptation_rate": float(result.get("adaptation_rate", 0.0)),
                    "total_adaptations": int(result.get("total_adaptations", 0)),
                    "rejected_episodes": int(result.get("rejected_episodes", 0)),
                    "skipped_adaptations": int(result.get("skipped_adaptations", 0)),
                    "n_episodes": int(result.get("n_episodes", len(all_rewards) or 0)),
                    "all_rewards": all_rewards,
                    "all_confidences": all_conf,
                    "config_path": summary.get("config_path"),
                    "device": summary.get("device"),
                }
            )
    return rows


def aggregate_eval_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], {}).setdefault(row["severity"], []).append(row)

    aggregates: Dict[str, Dict[str, Any]] = {}
    for method, by_severity in grouped.items():
        aggregates[method] = {}
        for severity, severity_rows in by_severity.items():
            rewards = [r["mean_reward"] for r in severity_rows]
            waypoints = [r["mean_waypoints_reached"] for r in severity_rows]
            success = [r["success_rate"] for r in severity_rows]
            adaptation = [r["adaptation_rate"] for r in severity_rows]
            rejected = [r["rejected_episodes"] for r in severity_rows]
            skipped = [r["skipped_adaptations"] for r in severity_rows]
            std_rewards = [r["std_reward"] for r in severity_rows]
            confs = [
                r["mean_confidence"]
                for r in severity_rows
                if r.get("mean_confidence") is not None
            ]
            all_episode_rewards: List[float] = []
            all_episode_conf: List[float] = []
            for r in severity_rows:
                all_episode_rewards.extend(r.get("all_rewards", []))
                all_episode_conf.extend(r.get("all_confidences", []))

            aggregates[method][severity] = {
                "n_seeds": len(severity_rows),
                "seeds": sorted({r["seed"] for r in severity_rows}),
                "reward_mean": _mean(rewards),
                "reward_std_across_seeds": _std(rewards),
                "reward_sem_across_seeds": _sem(rewards),
                "episode_std_reward_mean": _mean(std_rewards),
                "waypoints_mean": _mean(waypoints),
                "success_rate_mean": _mean(success),
                "adaptation_rate_mean": _mean(adaptation),
                "rejected_episodes_mean": _mean(rejected),
                "skipped_adaptations_mean": _mean(skipped),
                "confidence_mean": _mean(confs),
                "confidence_reward_corr": _pearson(all_episode_conf, all_episode_rewards),
            }

    paired = _paired_deltas(rows)
    return {"rows": list(rows), "aggregates": aggregates, "paired_deltas": paired}


def _paired_deltas(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    frozen: Dict[tuple[int, str], Dict[str, Any]] = {}
    for row in rows:
        if row["method"] == "frozen":
            frozen[(row["seed"], row["severity"])] = row

    deltas: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if row["method"] == "frozen":
            continue
        key = (row["seed"], row["severity"])
        if key not in frozen:
            continue
        base = frozen[key]["mean_reward"]
        delta = row["mean_reward"] - base
        delta_pct = (delta / abs(base) * 100.0) if abs(base) > 1e-8 else 0.0
        by_sev = deltas.setdefault(row["method"], {}).setdefault(
            row["severity"], {"reward_delta": [], "reward_delta_pct": []}
        )
        by_sev["reward_delta"].append(float(delta))
        by_sev["reward_delta_pct"].append(float(delta_pct))

    out: Dict[str, Dict[str, Any]] = {}
    for method, by_sev in deltas.items():
        out[method] = {}
        for severity, vals in by_sev.items():
            out[method][severity] = {
                "reward_delta_mean": _mean(vals["reward_delta"]),
                "reward_delta_std": _std(vals["reward_delta"]),
                "reward_delta_sem": _sem(vals["reward_delta"]),
                "reward_delta_pct_mean": _mean(vals["reward_delta_pct"]),
                "reward_delta_pct_std": _std(vals["reward_delta_pct"]),
                "reward_delta_pct_sem": _sem(vals["reward_delta_pct"]),
            }
    return out


def select_tuned_method(summary: Dict[str, Any]) -> Dict[str, Any]:
    aggs = summary.get("aggregates", {})
    paired = summary.get("paired_deltas", {})
    always_severe_std = (
        aggs.get("always_adapt", {})
        .get("severe", {})
        .get("episode_std_reward_mean")
    )
    if always_severe_std is None:
        always_severe_std = float("inf")

    candidates: List[Dict[str, Any]] = []
    for method in TUNED_CANDIDATES:
        if method not in aggs:
            continue
        clean_delta = (
            paired.get(method, {})
            .get("clean", {})
            .get("reward_delta_pct_mean")
        )
        severe_std = aggs[method].get("severe", {}).get("episode_std_reward_mean")
        surprise_rewards = [
            aggs[method].get(sev, {}).get("reward_mean")
            for sev in ("mild", "moderate", "severe")
        ]
        if clean_delta is None or severe_std is None or any(v is None for v in surprise_rewards):
            continue
        passes = clean_delta >= -5.0 and severe_std <= always_severe_std
        candidates.append(
            {
                "method": method,
                "passes_rule": passes,
                "clean_delta_pct_mean": clean_delta,
                "severe_episode_std_mean": severe_std,
                "mean_surprise_reward": float(np.mean(surprise_rewards)),
            }
        )

    passing = [c for c in candidates if c["passes_rule"]]
    selected_pool = passing or candidates
    if not selected_pool:
        return {
            "selected_method": "current_default",
            "reason": "No complete candidate metrics found; defaulting to current_default.",
            "candidates": candidates,
        }

    selected = max(selected_pool, key=lambda c: c["mean_surprise_reward"])
    return {
        "selected_method": selected["method"],
        "reason": (
            "Highest mean surprise reward among candidates satisfying clean-retention "
            "and severe-variance constraints."
            if selected["passes_rule"]
            else "No candidate satisfied all constraints; chose highest mean surprise reward."
        ),
        "candidates": candidates,
    }


def write_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    aggs = summary.get("aggregates", {})
    paired = summary.get("paired_deltas", {})
    lines = ["# Extension Experiment Summary", ""]

    for method in sorted(aggs):
        lines.append(f"## {method}")
        lines.append("")
        lines.append(
            "| severity | reward mean | reward SEM | waypoints | success | adapt rate | confidence | delta vs frozen |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for severity in SEVERITIES:
            if severity not in aggs[method]:
                continue
            row = aggs[method][severity]
            delta = paired.get(method, {}).get(severity, {}).get("reward_delta_pct_mean")
            lines.append(
                f"| {severity} | {_fmt(row.get('reward_mean'))} | "
                f"{_fmt(row.get('reward_sem_across_seeds'))} | "
                f"{_fmt(row.get('waypoints_mean'))} | "
                f"{_fmt(row.get('success_rate_mean'))} | "
                f"{_fmt(row.get('adaptation_rate_mean'))} | "
                f"{_fmt(row.get('confidence_mean'))} | "
                f"{_fmt(delta, suffix='%')} |"
            )
        lines.append("")

    if "selection" in summary:
        sel = summary["selection"]
        lines.append("## Tuned Method Selection")
        lines.append("")
        lines.append(f"Selected: `{sel['selected_method']}`")
        lines.append("")
        lines.append(sel["reason"])
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}{suffix}"


def summarize_stage(stage_dir: Path) -> Dict[str, Any]:
    rows = collect_eval_rows(stage_dir)
    summary = aggregate_eval_rows(rows)
    if stage_dir.name == "screening":
        summary["selection"] = select_tuned_method(summary)
    stage_dir.mkdir(parents=True, exist_ok=True)
    json_path = stage_dir / "aggregate_summary.json"
    md_path = stage_dir / "aggregate_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    write_diagnostic_plot(summary, stage_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if "selection" in summary:
        selection_path = stage_dir / "selection.json"
        selection_path.write_text(
            json.dumps(summary["selection"], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {selection_path}")
    return summary


def write_diagnostic_plot(summary: Dict[str, Any], out_dir: Path) -> None:
    """Plot mean surprise reward against clean retention for quick diagnosis."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Skipping diagnostic plot: {exc}")
        return

    aggs = summary.get("aggregates", {})
    paired = summary.get("paired_deltas", {})
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    for method, by_sev in aggs.items():
        clean_delta = paired.get(method, {}).get("clean", {}).get("reward_delta_pct_mean")
        if method == "frozen":
            clean_delta = 0.0
        surprise = [
            by_sev.get(sev, {}).get("reward_mean")
            for sev in ("mild", "moderate", "severe")
        ]
        if clean_delta is None or any(v is None for v in surprise):
            continue
        xs.append(float(clean_delta))
        ys.append(float(np.mean(surprise)))
        labels.append(method)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axvline(-5.0, color="#D65F5F", linestyle="--", linewidth=1.2, label="-5% clean limit")
    ax.scatter(xs, ys, s=90, color="#4878CF", edgecolor="white", linewidth=0.8, zorder=3)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Clean reward delta vs frozen (%)")
    ax.set_ylabel("Mean surprise reward (mild/moderate/severe)")
    ax.set_title("Extension Diagnostic: Plasticity vs Clean Retention")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"diagnostic_reward_retention.{ext}"
        fig.savefig(path, dpi=300)
        print(f"Wrote {path}")
    plt.close(fig)


def resolve_tuned_method(save_root: Path, tuned_method: str) -> str:
    if tuned_method != "auto":
        if tuned_method not in METHODS:
            raise SystemExit(f"Unknown tuned method: {tuned_method}")
        return tuned_method

    selection_path = save_root / "screening" / "selection.json"
    if not selection_path.exists():
        summarize_stage(save_root / "screening")
    if selection_path.exists():
        selection = _load_json(selection_path)
        selected = selection.get("selected_method", "current_default")
        if selected in METHODS:
            return selected
    return "current_default"


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)

    if args.action in ("screening", "all"):
        methods = _split_csv(args.methods) or SCREENING_METHODS
        episodes = args.episodes or args.screening_episodes
        run_eval_stage(
            "screening",
            methods,
            seeds,
            episodes,
            args.save_root,
            args.baseline_path,
            args.skip_existing,
            args.dry_run,
        )
        if not args.dry_run:
            summarize_stage(args.save_root / "screening")

    if args.action in ("validation", "all"):
        tuned = resolve_tuned_method(args.save_root, args.tuned_method)
        methods = _split_csv(args.methods) or VALIDATION_BASE_METHODS + [tuned]
        methods = list(dict.fromkeys(methods))
        episodes = args.episodes or args.validation_episodes
        run_eval_stage(
            "validation",
            methods,
            seeds,
            episodes,
            args.save_root,
            args.baseline_path,
            args.skip_existing,
            args.dry_run,
        )
        if not args.dry_run:
            summarize_stage(args.save_root / "validation")

    if args.action in ("continual", "all"):
        tuned = resolve_tuned_method(args.save_root, args.tuned_method)
        methods = _split_csv(args.methods) or list(dict.fromkeys(["current_default", tuned]))
        run_continual_stage(
            methods,
            seeds,
            args.save_root,
            args.baseline_path,
            args.continual_adapt_episodes,
            args.continual_eval_episodes,
            args.skip_existing,
            args.dry_run,
        )

    if args.action == "domain-randomized":
        run_domain_randomized(args)

    if args.action == "summarize":
        stages = _split_csv(args.methods) or ["screening", "validation", "domain_randomized_eval"]
        for stage in stages:
            stage_dir = args.save_root / stage
            if stage_dir.exists():
                summarize_stage(stage_dir)
            else:
                print(f"missing stage: {stage_dir}")


if __name__ == "__main__":
    main()
