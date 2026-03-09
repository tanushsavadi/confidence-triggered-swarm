#!/usr/bin/env python3
# Generate final report figures.
#
# Outputs 4 figures to runs/final_plots/:
#   fig1 - frozen vs adapted bar chart
#   fig2 - degradation line plot
#   fig3 - ablation horizontal bars
#   fig4 - forgetting analysis
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.generate_plots

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # project root
OUT_DIR = BASE_DIR / "runs" / "final_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# matplotlib style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

plt.rcParams.update({
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
})

# colors
C_FROZEN  = "#4878CF"
C_ADAPTED = "#E8833A"
C_GREEN   = "#6ACC65"
C_RED     = "#D65F5F"
C_PURPLE  = "#956CB4"

# ---------------------------------------------------------------------------
# hardcoded from our final experiment results
# (20 episodes per severity for main eval, 15 for ablations)
# ---------------------------------------------------------------------------

SEVERITIES = ["clean", "mild", "moderate", "severe"]
FROZEN_REWARD  = [1337.7, 160.3, 31.6, 24.0]
ADAPTED_REWARD = [1459.3, 158.7, 75.4, 60.4]
FROZEN_WP  = [0.75, 0.60, 0.25, 0.25]
ADAPTED_WP = [0.75, 0.50, 0.40, 0.25]

# ablation (15 episodes, severe)
ABL_LABELS  = ["Frozen (no adapt)", "Full Method", "No KL Anchoring",
               "No Clean Replay", "No EWC"]
ABL_MEAN    = [29.18, 24.30, 53.13, 46.21, 45.89]
ABL_STD     = [62.04, 21.58, 77.84, 59.91, 62.81]
ABL_WP      = [0.00, 0.13, 0.07, 0.27, 0.27]
ABL_ADAPT   = [0, 0, 1, 1, 1]

# forgetting
FORGET_BASELINE_REWARD = 1293.1
FORGET_ADAPTED_REWARD  = 1433.3
FORGET_BASELINE_WP     = 0.85
FORGET_ADAPTED_WP      = 1.05


def _save(fig, name):
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(str(path))
        print(f"  saved {path}")
    plt.close(fig)


# --- Figure 1: Frozen vs Adapted ---

def fig1_frozen_vs_adapted():
    print("Fig 1: Frozen vs Adapted...")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(SEVERITIES))
    width = 0.35

    bars1 = ax.bar(x - width / 2, FROZEN_REWARD, width, label="Frozen Policy",
                   color=C_FROZEN, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ADAPTED_REWARD, width, label="Adapted Policy",
                   color=C_ADAPTED, edgecolor="white", linewidth=0.5)

    # value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Surprise Severity")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Frozen vs. Adapted Policy Performance Under Surprise")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SEVERITIES])
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, "fig1_frozen_vs_adapted")


# --- Figure 2: Degradation ---

def fig2_degradation():
    print("Fig 2: Degradation...")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(SEVERITIES))
    ax.plot(x, FROZEN_REWARD, marker="o", linewidth=2.5, markersize=8,
            color=C_FROZEN, label="Frozen Policy", zorder=3)
    ax.plot(x, ADAPTED_REWARD, marker="s", linewidth=2.5, markersize=8,
            color=C_ADAPTED, label="Adapted Policy", zorder=3)

    ax.fill_between(x, FROZEN_REWARD, ADAPTED_REWARD, alpha=0.12, color=C_ADAPTED)

    # annotate improvement at moderate and severe
    for i in [2, 3]:
        delta = ((ADAPTED_REWARD[i] - FROZEN_REWARD[i]) / max(FROZEN_REWARD[i], 1)) * 100
        mid_y = (FROZEN_REWARD[i] + ADAPTED_REWARD[i]) / 2
        ax.annotate(f"+{delta:.0f}%", xy=(x[i] + 0.08, mid_y),
                    fontsize=11, fontweight="bold", color=C_ADAPTED)

    ax.set_xlabel("Surprise Severity")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Performance Degradation Under Increasing Surprise Severity")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SEVERITIES])
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, "fig2_degradation")


# --- Figure 3: Ablation Study ---

def fig3_ablations():
    print("Fig 3: Ablation...")
    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(ABL_LABELS))
    colors = [C_FROZEN, C_ADAPTED, C_GREEN, C_RED, C_PURPLE]

    bars = ax.barh(y, ABL_MEAN, xerr=ABL_STD, height=0.6,
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2})

    for bar, mean, std in zip(bars, ABL_MEAN, ABL_STD):
        w = bar.get_width()
        ax.text(w + std + 3, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1f} ± {std:.1f}", va="center", ha="left", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(ABL_LABELS)
    ax.set_xlabel("Mean Episode Reward (Severe Surprise)")
    ax.set_title("Ablation Study: Component Contribution Under Severe Surprise")
    ax.invert_yaxis()
    ax.set_xlim(right=max(m + s for m, s in zip(ABL_MEAN, ABL_STD)) + 40)

    fig.tight_layout()
    _save(fig, "fig3_ablations")


# --- Figure 4: Forgetting ---

def fig4_forgetting():
    print("Fig 4: Forgetting...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    conditions = ["Before\nAdaptation", "After\nAdaptation"]
    bar_colors = [C_FROZEN, C_ADAPTED]

    # reward subplot
    ax = axes[0]
    rewards = [FORGET_BASELINE_REWARD, FORGET_ADAPTED_REWARD]
    bars = ax.bar(conditions, rewards, width=0.5, color=bar_colors,
                  edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Reward on Clean Environment")
    ax.set_ylim(0, max(rewards) * 1.18)

    delta_r = (FORGET_ADAPTED_REWARD - FORGET_BASELINE_REWARD) / FORGET_BASELINE_REWARD * 100
    ax.annotate(f"+{delta_r:.1f}%", xy=(1, FORGET_ADAPTED_REWARD),
                xytext=(0.15, FORGET_ADAPTED_REWARD * 0.85),
                fontsize=12, fontweight="bold", color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.5))

    # waypoints subplot
    ax = axes[1]
    wps = [FORGET_BASELINE_WP, FORGET_ADAPTED_WP]
    bars = ax.bar(conditions, wps, width=0.5, color=bar_colors,
                  edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Waypoints Reached")
    ax.set_title("Waypoints on Clean Environment")
    ax.set_ylim(0, max(wps) * 1.25)

    fig.suptitle(
        "Catastrophic Forgetting Analysis:\nClean Performance Before and After Adaptation",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig4_forgetting")


def main():
    print(f"Output: {OUT_DIR}\n")
    fig1_frozen_vs_adapted()
    fig2_degradation()
    fig3_ablations()
    fig4_forgetting()
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
