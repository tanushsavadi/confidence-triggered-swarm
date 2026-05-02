#!/usr/bin/env python3
"""Run short comparison sweeps for improved lifelong adaptation configs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PACKAGE_ROOT / "configs"

CONFIGS: Dict[str, Optional[Path]] = {
    "current_default": None,
    "deterministic_only": CONFIG_DIR / "improved_deterministic.yaml",
    "ppo_episode": CONFIG_DIR / "improved_ppo.yaml",
    "ppo_conservative": CONFIG_DIR / "improved_ppo_conservative.yaml",
    "ppo_moderate": CONFIG_DIR / "improved_ppo_moderate.yaml",
    "reward_weighted_rescue": CONFIG_DIR / "improved_reward_weighted_rescue.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep improved lifelong adaptation configs without replacing canonical runs."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per severity for each candidate config.",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="runs/baseline/best_model.pt",
        help="Path to trained baseline checkpoint.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="runs/improved_sweep",
        help="Output root for candidate run folders and summary files.",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="clean,mild,moderate,severe",
        help="Comma-separated severities to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed passed through to evaluation.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of config names to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Reuse existing lifelong_results.json files if present.",
    )
    return parser.parse_args()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_result(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("lifelong", payload)


def _summarize_config(
    name: str,
    run_dir: Path,
    severities: Iterable[str],
) -> Dict[str, Any]:
    result_path = run_dir / "lifelong_results.json"
    data = _load_result(result_path)

    rows: Dict[str, Dict[str, Any]] = {}
    for severity in severities:
        result = data.get(severity, {})
        rows[severity] = {
            "mean_reward": _to_float(result.get("mean_reward")),
            "std_reward": _to_float(result.get("std_reward")),
            "mean_waypoints_reached": _to_float(
                result.get("mean_waypoints_reached")
            ),
            "mean_episode_length": _to_float(result.get("mean_episode_length")),
            "adaptation_rate": _to_float(result.get("adaptation_rate")),
            "total_adaptations": int(result.get("total_adaptations", 0)),
            "skipped_adaptations": int(result.get("skipped_adaptations", 0)),
            "rejected_episodes": int(result.get("rejected_episodes", 0)),
        }

    surprise_rewards = [
        rows[s]["mean_reward"] for s in rows if s != "clean"
    ]
    return {
        "name": name,
        "run_dir": str(run_dir),
        "by_severity": rows,
        "mean_surprise_reward": float(np.mean(surprise_rewards))
        if surprise_rewards
        else 0.0,
        "clean_reward": rows.get("clean", {}).get("mean_reward", 0.0),
    }


def _write_markdown(summary: List[Dict[str, Any]], out_path: Path) -> None:
    severities = [
        s for s in (list(summary[0]["by_severity"].keys()) if summary else [])
        if s != "clean"
    ]
    lines = [
        "# Improved Adaptation Sweep",
        "",
        "| config | clean reward | surprise mean | "
        + " | ".join(f"{s} reward" for s in severities)
        + " | adaptations | skipped | rejected |",
        "|---|---:|---:|"
        + "|".join("---:" for _ in severities)
        + "|---:|---:|---:|",
    ]
    for row in summary:
        by_sev = row["by_severity"]
        rewards = " | ".join(
            f"{by_sev[s]['mean_reward']:.2f}" for s in severities
        )
        adaptations = sum(v["total_adaptations"] for v in by_sev.values())
        skipped = sum(v["skipped_adaptations"] for v in by_sev.values())
        rejected = sum(v["rejected_episodes"] for v in by_sev.values())
        lines.append(
            f"| {row['name']} | {row['clean_reward']:.2f} | "
            f"{row['mean_surprise_reward']:.2f} | {rewards} | "
            f"{adaptations} | {skipped} | {rejected} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    severities = [s.strip() for s in args.severities.split(",") if s.strip()]
    selected = list(CONFIGS.keys())
    if args.only:
        requested = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = sorted(set(requested) - set(CONFIGS))
        if unknown:
            raise SystemExit(f"Unknown config name(s): {', '.join(unknown)}")
        selected = requested

    summary: List[Dict[str, Any]] = []
    for name in selected:
        config_path = CONFIGS[name]
        run_dir = save_root / name
        result_path = run_dir / "lifelong_results.json"

        if not (args.skip_existing and result_path.exists()):
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "confidence_triggered_swarm.scripts.evaluate",
                "--baseline-path",
                args.baseline_path,
                "--mode",
                "lifelong",
                "--severity",
                ",".join(severities),
                "--episodes",
                str(args.episodes),
                "--save-dir",
                str(run_dir),
            ]
            if config_path is not None:
                cmd.extend(["--config", str(config_path)])
            if args.seed is not None:
                cmd.extend(["--seed", str(args.seed)])

            print(f"\n=== Running {name} ===")
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        summary.append(_summarize_config(name, run_dir, severities))

    summary_path = save_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, save_root / "summary.md")

    print(f"\nWrote {summary_path}")
    print(f"Wrote {save_root / 'summary.md'}")


if __name__ == "__main__":
    main()
