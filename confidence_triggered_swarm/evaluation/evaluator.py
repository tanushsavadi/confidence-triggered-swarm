"""Evaluation suite: frozen baseline vs lifelong adaptation across surprise levels.

Measures frozen baseline performance, lifelong adaptation gains, and
forgetting on clean env after adaptation.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from confidence_triggered_swarm.algorithms.ppo import PPOAgent
from confidence_triggered_swarm.adaptation.confidence import ConfidenceMonitor
from confidence_triggered_swarm.adaptation.ewc import EWCRegularizer
from confidence_triggered_swarm.adaptation.lifelong_trainer import LifelongTrainer
from confidence_triggered_swarm.envs.formation_aviary import FormationAviary
from confidence_triggered_swarm.envs.surprise_wrapper import SurpriseConfig, SurpriseWrapper


class Evaluator:
    """Compares frozen IPPO baseline vs lifelong adaptation across surprise levels.

    Produces comparison metrics, tables, and plots.
    """
    # TODO: could use utils.factory for _create_env/_create_agent but the
    # refactor would be pretty invasive since Evaluator caches obs_dim/act_dim
    # and has its own device resolution. Not worth it right now.

    def __init__(
        self,
        config: dict,
        baseline_model_path: str,
        save_dir: str = "results",
        device: str = "auto",
    ) -> None:
        self.config = config
        self.baseline_model_path = baseline_model_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # cache obs_dim/act_dim from a temp env
        tmp_env = self._create_env("clean")
        obs_sample, _ = tmp_env.reset()
        self.obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
        self.act_dim = tmp_env.action_space.shape[-1]
        tmp_env.close()

        self.n_eval_episodes = config.get("evaluation", {}).get("n_eval_episodes", 50)
        self.default_severity_levels = ["clean", "mild", "moderate", "severe"]

    # -- public eval methods --

    def evaluate_frozen_baseline(
        self, severity_levels: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Eval frozen baseline (no adaptation) across surprise levels."""
        if severity_levels is None:
            severity_levels = self.default_severity_levels

        all_results: Dict[str, Dict[str, Any]] = {}

        for severity in severity_levels:
            print(f"\n{'='*60}")
            print(f"  Frozen Baseline — severity: {severity}")
            print(f"{'='*60}")

            env = self._create_env(severity)
            agent = self._create_agent()
            agent.load(self.baseline_model_path)

            results = self._run_frozen_episodes(
                agent, env, self.n_eval_episodes, label=f"frozen/{severity}"
            )
            results["severity"] = severity
            results["mode"] = "frozen"
            all_results[severity] = results

            env.close()

        return all_results

    def evaluate_lifelong(
        self, severity_levels: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Eval lifelong adaptation across surprise levels."""
        if severity_levels is None:
            severity_levels = self.default_severity_levels

        adapt_cfg = self.config.get("adaptation", {})
        policy_cfg = self.config.get("policy", {})
        all_results: Dict[str, Dict[str, Any]] = {}

        for severity in severity_levels:
            print(f"\n{'='*60}")
            print(f"  Lifelong Adaptation — severity: {severity}")
            print(f"{'='*60}")

            agent = self._create_agent()
            agent.load(self.baseline_model_path)

            clean_env = self._create_env("clean")
            surprise_env = self._create_env(severity)

            monitor = ConfidenceMonitor(
                policy=agent.policy,
                confidence_threshold=adapt_cfg.get("confidence_threshold", 0.5),
                window_size=adapt_cfg.get("confidence_window", 50),
                use_mc_dropout=True,
                mc_samples=policy_cfg.get("mc_samples", 10),
                device=self.device,
            )
            ewc = EWCRegularizer(
                policy=agent.policy,
                ewc_lambda=adapt_cfg.get("ewc_lambda", 1000),
                device=self.device,
            )
            trainer = LifelongTrainer(
                agent=agent,
                confidence_monitor=monitor,
                ewc_regularizer=ewc,
                adapt_epochs=adapt_cfg.get("adapt_epochs", 5),
                adapt_lr=adapt_cfg.get("adapt_lr", 1e-4),
                adapt_batch_size=adapt_cfg.get("adapt_batch_size", 32),
                replay_buffer_size=adapt_cfg.get("replay_buffer_size", 10000),
                config=self.config,
                device=self.device,
            )

            trainer.setup(clean_env, n_calibration_episodes=5)

            results = trainer.run_evaluation(
                surprise_env,
                n_episodes=self.n_eval_episodes,
                label=f"lifelong/{severity}",
            )
            results["severity"] = severity
            results["mode"] = "lifelong"
            results["success_rate"] = self._compute_success_rate_from_results(results)

            all_results[severity] = results

            clean_env.close()
            surprise_env.close()

        return all_results

    def evaluate_forgetting(
        self, lifelong_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check if adaptation causes forgetting on clean env.

        Compares original frozen baseline vs adapted policy on clean env.
        If no adapted model given, adapts on severe first.
        """
        print(f"\n{'='*60}")
        print("  Forgetting Evaluation")
        print(f"{'='*60}")

        # frozen baseline on clean
        print("\n--- Frozen baseline on clean env ---")
        clean_env_1 = self._create_env("clean")
        baseline_agent = self._create_agent()
        baseline_agent.load(self.baseline_model_path)
        baseline_clean = self._run_frozen_episodes(
            baseline_agent, clean_env_1, self.n_eval_episodes, label="forgetting/baseline"
        )
        clean_env_1.close()

        adapt_cfg = self.config.get("adaptation", {})
        policy_cfg = self.config.get("policy", {})

        if lifelong_model_path is not None:
            adapted_agent = self._create_agent()
            adapted_agent.load(lifelong_model_path)
        else:
            # adapt on severe, then test on clean
            print("\n--- Adapting on severe environment ---")
            adapted_agent = self._create_agent()
            adapted_agent.load(self.baseline_model_path)

            clean_env_cal = self._create_env("clean")
            severe_env = self._create_env("severe")

            monitor = ConfidenceMonitor(
                policy=adapted_agent.policy,
                confidence_threshold=adapt_cfg.get("confidence_threshold", 0.5),
                window_size=adapt_cfg.get("confidence_window", 50),
                use_mc_dropout=True,
                mc_samples=policy_cfg.get("mc_samples", 10),
                device=self.device,
            )
            ewc = EWCRegularizer(
                policy=adapted_agent.policy,
                ewc_lambda=adapt_cfg.get("ewc_lambda", 1000),
                device=self.device,
            )
            trainer = LifelongTrainer(
                agent=adapted_agent,
                confidence_monitor=monitor,
                ewc_regularizer=ewc,
                adapt_epochs=adapt_cfg.get("adapt_epochs", 5),
                adapt_lr=adapt_cfg.get("adapt_lr", 1e-4),
                adapt_batch_size=adapt_cfg.get("adapt_batch_size", 32),
                replay_buffer_size=adapt_cfg.get("replay_buffer_size", 10000),
                config=self.config,
                device=self.device,
            )
            trainer.setup(clean_env_cal, n_calibration_episodes=5)

            _ = trainer.run_evaluation(
                severe_env, n_episodes=self.n_eval_episodes, label="forgetting/adapt_severe"
            )

            clean_env_cal.close()
            severe_env.close()

        # eval adapted policy on clean
        print("\n--- Adapted policy on clean env ---")
        clean_env_2 = self._create_env("clean")
        adapted_clean = self._run_frozen_episodes(
            adapted_agent, clean_env_2, self.n_eval_episodes, label="forgetting/adapted"
        )
        clean_env_2.close()

        reward_drop = baseline_clean["mean_reward"] - adapted_clean["mean_reward"]
        reward_drop_pct = (
            (reward_drop / abs(baseline_clean["mean_reward"]) * 100.0)
            if abs(baseline_clean["mean_reward"]) > 1e-8
            else 0.0
        )

        return {
            "baseline_clean": baseline_clean,
            "adapted_clean": adapted_clean,
            "reward_drop": float(reward_drop),
            "reward_drop_pct": float(reward_drop_pct),
            "waypoint_drop": float(
                baseline_clean["mean_waypoints_reached"]
                - adapted_clean["mean_waypoints_reached"]
            ),
            "forgetting_detected": reward_drop_pct > 5.0,
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run the full eval suite: frozen, lifelong, forgetting, comparison."""
        start_time = time.time()

        print("\n" + "=" * 70)
        print("  FULL EVALUATION SUITE")
        print("=" * 70)

        baseline_results = self.evaluate_frozen_baseline()
        lifelong_results = self.evaluate_lifelong()
        forgetting_results = self.evaluate_forgetting()
        comparison = self._compare_results(baseline_results, lifelong_results)

        elapsed = time.time() - start_time

        results = {
            "baseline": baseline_results,
            "lifelong": lifelong_results,
            "forgetting": forgetting_results,
            "comparison": comparison,
            "summary": {
                "total_time_sec": float(elapsed),
                "n_eval_episodes": self.n_eval_episodes,
                "severity_levels": self.default_severity_levels,
                "baseline_model": self.baseline_model_path,
                "device": self.device,
            },
        }

        self.save_results(results)
        self.print_summary(results)
        self.plot_results(results)

        return results

    # -- env / agent helpers --

    def _create_env(self, severity: str = "clean") -> Any:
        """Create env with given surprise severity."""
        env_cfg = self.config.get("env", {})

        from gym_pybullet_drones.utils.enums import (
            ActionType,
            DroneModel,
            ObservationType,
            Physics,
        )

        env = FormationAviary(
            num_drones=env_cfg.get("num_drones", 2),
            gui=False,
            freq=env_cfg.get("freq", 240),
            ctrl_freq=env_cfg.get("ctrl_freq", 30),
            episode_len_sec=env_cfg.get("episode_len_sec", 15.0),
            init_height=env_cfg.get("init_height", 0.5),
            speed_limit=env_cfg.get("speed_limit", 0.5),
            tilt_threshold=env_cfg.get("tilt_threshold", 1.0),
            z_min=env_cfg.get("z_min", 0.02),
            waypoint_threshold=env_cfg.get("waypoint_threshold", 0.2),
            drone_model=DroneModel[env_cfg.get("drone_model", "CF2X")],
            physics=Physics[env_cfg.get("physics", "PYB")],
            obs=ObservationType[env_cfg.get("obs_type", "KIN")],
            act=ActionType[env_cfg.get("action_type", "VEL")],
        )

        if severity != "clean":
            surprise_config = SurpriseConfig.from_severity(severity)
            env = SurpriseWrapper(env, surprise_config)

        return env

    def _create_agent(self) -> PPOAgent:
        """Create a fresh PPOAgent from config (weights not loaded)."""
        train_cfg = self.config.get("training", {})
        policy_cfg = self.config.get("policy", {})
        env_cfg = self.config.get("env", {})

        return PPOAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            num_drones=env_cfg.get("num_drones", 2),
            lr=train_cfg.get("lr", 3e-4),
            gamma=train_cfg.get("gamma", 0.99),
            gae_lambda=train_cfg.get("gae_lambda", 0.95),
            clip_epsilon=train_cfg.get("clip_epsilon", 0.2),
            entropy_coef=train_cfg.get("entropy_coef", 0.01),
            value_coef=train_cfg.get("value_coef", 0.5),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
            n_epochs=train_cfg.get("n_epochs", 10),
            batch_size=train_cfg.get("batch_size", 64),
            rollout_steps=train_cfg.get("rollout_steps", 2048),
            hidden_dims=policy_cfg.get("hidden_dims", [256, 256]),
            activation=policy_cfg.get("activation", "tanh"),
            mc_dropout_p=policy_cfg.get("mc_dropout_p", 0.1),
            device=self.device,
        )

    # -- frozen episode runner --

    def _run_frozen_episodes(
        self,
        agent: PPOAgent,
        env: Any,
        n_episodes: int,
        label: str = "",
    ) -> Dict[str, Any]:
        """Run episodes with frozen policy (no gradient updates)."""
        agent.policy.eval()

        all_rewards: List[float] = []
        all_lengths: List[int] = []
        all_waypoints: List[int] = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0

            while not done:
                actions, _, _, _ = agent.select_actions(obs)
                obs, reward, terminated, truncated, info = env.step(actions)
                done = bool(terminated or truncated)
                ep_reward += float(reward)
                step_count += 1

            all_rewards.append(ep_reward)
            all_lengths.append(step_count)
            all_waypoints.append(info.get("current_waypoint_idx", 0))

            if (ep + 1) % 10 == 0:
                recent_reward = float(np.mean(all_rewards[-10:]))
                print(
                    f"  [{label}] Episode {ep + 1}/{n_episodes}: "
                    f"avg_reward(last10)={recent_reward:.2f}"
                )

        total_waypoints = info.get("total_waypoints", 5) if info else 5
        success_count = sum(
            1 for w in all_waypoints if w >= total_waypoints - 1
        )

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_episode_length": float(np.mean(all_lengths)),
            "mean_waypoints_reached": float(np.mean(all_waypoints)),
            "success_rate": float(success_count / max(n_episodes, 1)),
            "all_rewards": [float(r) for r in all_rewards],
            "all_waypoints": [int(w) for w in all_waypoints],
            "n_episodes": n_episodes,
        }

    @staticmethod
    def _compute_success_rate_from_results(results: Dict[str, Any]) -> float:
        """Compute success rate from LifelongTrainer results.

        Success = waypoints_reached >= total_waypoints - 1.
        """
        wp = results.get("mean_waypoints_reached", 0)
        total_waypoints = results.get("total_waypoints", 3)
        return float(wp >= total_waypoints - 1)

    # -- comparison --

    def _compare_results(
        self,
        baseline_results: Dict[str, Dict[str, Any]],
        lifelong_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate comparison metrics between baseline and lifelong."""
        comparison: Dict[str, Any] = {}

        for severity in baseline_results:
            bl = baseline_results[severity]
            ll = lifelong_results.get(severity, {})

            if not ll:
                continue

            bl_reward = bl.get("mean_reward", 0.0)
            ll_reward = ll.get("mean_reward", 0.0)
            reward_improvement = ll_reward - bl_reward
            reward_improvement_pct = (
                (reward_improvement / abs(bl_reward) * 100.0)
                if abs(bl_reward) > 1e-8
                else 0.0
            )

            bl_wp = bl.get("mean_waypoints_reached", 0.0)
            ll_wp = ll.get("mean_waypoints_reached", 0.0)

            comparison[severity] = {
                "baseline_reward": float(bl_reward),
                "lifelong_reward": float(ll_reward),
                "reward_improvement": float(reward_improvement),
                "reward_improvement_pct": float(reward_improvement_pct),
                "baseline_waypoints": float(bl_wp),
                "lifelong_waypoints": float(ll_wp),
                "waypoint_improvement": float(ll_wp - bl_wp),
                "baseline_success_rate": float(bl.get("success_rate", 0.0)),
                "lifelong_success_rate": float(ll.get("success_rate", 0.0)),
                "adaptation_rate": float(ll.get("adaptation_rate", 0.0)),
                "total_adaptations": int(ll.get("total_adaptations", 0)),
                "mean_confidence": float(ll.get("mean_confidence", 0.0)),
            }

        return comparison

    # -- I/O and reporting --

    def save_results(
        self, results: Dict[str, Any], filename: str = "evaluation_results.json"
    ) -> None:
        """Save results to JSON (numpy types -> native Python)."""
        path = self.save_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(item) for item in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        with open(path, "w") as f:
            json.dump(_convert(results), f, indent=2, default=str)

        print(f"\nResults saved to {path}")

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted comparison table."""
        comparison = results.get("comparison", {})
        forgetting = results.get("forgetting", {})

        print("\n" + "=" * 80)
        print("  EVALUATION SUMMARY")
        print("=" * 80)

        header = (
            f"{'Severity':<12} │ "
            f"{'Baseline Rew':>12} │ "
            f"{'Lifelong Rew':>12} │ "
            f"{'Δ Reward':>10} │ "
            f"{'Δ %':>7} │ "
            f"{'Adapt Rate':>10}"
        )
        print(header)
        print("─" * 80)

        for severity in self.default_severity_levels:
            if severity not in comparison:
                continue
            c = comparison[severity]
            print(
                f"{severity:<12} │ "
                f"{c['baseline_reward']:>12.2f} │ "
                f"{c['lifelong_reward']:>12.2f} │ "
                f"{c['reward_improvement']:>+10.2f} │ "
                f"{c['reward_improvement_pct']:>+6.1f}% │ "
                f"{c['adaptation_rate']:>10.2f}"
            )

        print("─" * 80)

        # waypoints sub-table
        print(
            f"\n{'Severity':<12} │ "
            f"{'BL Waypts':>10} │ "
            f"{'LL Waypts':>10} │ "
            f"{'BL Success':>10} │ "
            f"{'LL Success':>10} │ "
            f"{'Confidence':>10}"
        )
        print("─" * 80)

        for severity in self.default_severity_levels:
            if severity not in comparison:
                continue
            c = comparison[severity]
            print(
                f"{severity:<12} │ "
                f"{c['baseline_waypoints']:>10.2f} │ "
                f"{c['lifelong_waypoints']:>10.2f} │ "
                f"{c['baseline_success_rate']:>10.2%} │ "
                f"{c['lifelong_success_rate']:>10.2%} │ "
                f"{c['mean_confidence']:>10.3f}"
            )

        if forgetting:
            print("\n" + "─" * 80)
            print("  FORGETTING ANALYSIS")
            print("─" * 80)
            print(
                f"  Baseline on clean:  reward = "
                f"{forgetting.get('baseline_clean', {}).get('mean_reward', 0):.2f}"
            )
            print(
                f"  Adapted on clean:   reward = "
                f"{forgetting.get('adapted_clean', {}).get('mean_reward', 0):.2f}"
            )
            print(
                f"  Reward drop:        {forgetting.get('reward_drop', 0):.2f} "
                f"({forgetting.get('reward_drop_pct', 0):.1f}%)"
            )
            print(
                f"  Forgetting detected: "
                f"{'YES' if forgetting.get('forgetting_detected') else 'NO'}"
            )

        summary = results.get("summary", {})
        if "total_time_sec" in summary:
            elapsed = summary["total_time_sec"]
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"\n  Total evaluation time: {minutes}m {seconds:.1f}s")

        print("=" * 80)

    def plot_results(
        self, results: Dict[str, Any], filename: str = "evaluation_plots.png"
    ) -> None:
        """Generate matplotlib comparison plots (2x2 grid).

        Gracefully skips if matplotlib not installed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "[WARNING] matplotlib not available — skipping plots. "
                "Install with: pip install matplotlib"
            )
            return

        comparison = results.get("comparison", {})
        lifelong = results.get("lifelong", {})

        severities = [s for s in self.default_severity_levels if s in comparison]
        if not severities:
            print("[WARNING] No comparison data for plotting.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Frozen Baseline vs Lifelong Adaptation",
            fontsize=14,
            fontweight="bold",
        )

        x = np.arange(len(severities))
        width = 0.35

        # plot 1: mean reward
        ax1 = axes[0, 0]
        bl_rewards = [comparison[s]["baseline_reward"] for s in severities]
        ll_rewards = [comparison[s]["lifelong_reward"] for s in severities]
        ax1.bar(x - width / 2, bl_rewards, width, label="Frozen Baseline", color="#4C72B0")
        ax1.bar(x + width / 2, ll_rewards, width, label="Lifelong", color="#DD8452")
        ax1.set_xlabel("Surprise Severity")
        ax1.set_ylabel("Mean Episode Reward")
        ax1.set_title("Mean Reward by Severity")
        ax1.set_xticks(x)
        ax1.set_xticklabels(severities)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # plot 2: waypoints
        ax2 = axes[0, 1]
        bl_wp = [comparison[s]["baseline_waypoints"] for s in severities]
        ll_wp = [comparison[s]["lifelong_waypoints"] for s in severities]
        ax2.bar(x - width / 2, bl_wp, width, label="Frozen Baseline", color="#4C72B0")
        ax2.bar(x + width / 2, ll_wp, width, label="Lifelong", color="#DD8452")
        ax2.set_xlabel("Surprise Severity")
        ax2.set_ylabel("Mean Waypoints Reached")
        ax2.set_title("Waypoints Reached by Severity")
        ax2.set_xticks(x)
        ax2.set_xticklabels(severities)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # plot 3: confidence over episodes
        ax3 = axes[1, 0]
        colors = ["#55A868", "#C44E52", "#8172B3", "#CCB974"]
        for idx, severity in enumerate(severities):
            ll_data = lifelong.get(severity, {})
            confidences = ll_data.get("all_confidences", [])
            if confidences:
                color = colors[idx % len(colors)]
                ax3.plot(
                    range(1, len(confidences) + 1),
                    confidences,
                    label=severity,
                    color=color,
                    alpha=0.8,
                )
        ax3.axhline(
            y=self.config.get("adaptation", {}).get("confidence_threshold", 0.5),
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Threshold",
        )
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Mean Confidence")
        ax3.set_title("Confidence Over Episodes (Lifelong)")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # plot 4: adaptation rate
        ax4 = axes[1, 1]
        adapt_rates = [comparison[s]["adaptation_rate"] for s in severities]
        bar_colors = ["#55A868", "#C44E52", "#8172B3", "#CCB974"]
        ax4.bar(
            x,
            adapt_rates,
            width * 1.5,
            color=[bar_colors[i % len(bar_colors)] for i in range(len(severities))],
        )
        ax4.set_xlabel("Surprise Severity")
        ax4.set_ylabel("Adaptation Rate")
        ax4.set_title("Adaptation Rate by Severity (Lifelong)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(severities)
        ax4.set_ylim(0, 1.0)
        ax4.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to {save_path}")
