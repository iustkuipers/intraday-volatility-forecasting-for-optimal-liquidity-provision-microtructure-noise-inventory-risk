"""
simulator/output.py

Saves all outputs from a simulation run to a timestamped experiment folder.

Every call to OutputManager.save_run() creates:

    results/
        YYYYMMDD_HHMMSS/
            results.csv
            metrics.json
            pnl.png
            inventory.png
            quotes.png

This ensures no run is ever overwritten and produces a self-contained
research archive for every experiment.

Usage
-----
from simulator.output import OutputManager

path = OutputManager.save_run(results_df, metrics_dict, label="ConstantSpread")
print(f"Saved to {path}")
"""

import os
import json
import math
from datetime import datetime

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import pandas as pd


class OutputManager:

    RESULTS_DIR = "results"

    @staticmethod
    def save_run(
        results:  pd.DataFrame,
        metrics:  dict,
        label:    str = "",
        base_dir: str = None,
    ) -> str:
        """
        Persist a full simulation run.

        Parameters
        ----------
        results  : pd.DataFrame   — output of Simulator.run()
        metrics  : dict           — output of Metrics.compute(results)
        label    : str            — optional tag added to folder name
        base_dir : str            — override results root (useful in tests)

        Returns
        -------
        str — absolute path to the experiment folder
        """
        root    = base_dir or OutputManager.RESULTS_DIR
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        name    = f"{ts}_{label}" if label else ts
        run_dir = os.path.join(root, name)
        os.makedirs(run_dir, exist_ok=True)

        OutputManager._save_csv(results,  run_dir)
        OutputManager._save_metrics(metrics, run_dir, label)
        OutputManager._plot_pnl(results,  run_dir, label)
        OutputManager._plot_inventory(results, run_dir, label)
        OutputManager._plot_quotes(results, run_dir, label)

        print(f"[OutputManager] Run saved → {run_dir}")
        return run_dir

    # ── csv ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_csv(results: pd.DataFrame, run_dir: str) -> None:
        path = os.path.join(run_dir, "results.csv")
        results.to_csv(path, index=False)

    # ── metrics json ─────────────────────────────────────────────────────────

    @staticmethod
    def _save_metrics(metrics: dict, run_dir: str, label: str) -> None:
        payload = {"label": label, **metrics}
        # Replace NaN/inf with None so json.dump doesn't crash
        clean = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in payload.items()
        }
        path = os.path.join(run_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _plot_pnl(results: pd.DataFrame, run_dir: str, label: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(results["portfolio_value"].values, color="steelblue", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"Portfolio Value — {label}" if label else "Portfolio Value")
        ax.set_xlabel("Event index")
        ax.set_ylabel("USD")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "pnl.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_inventory(results: pd.DataFrame, run_dir: str, label: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(results["inventory"].values, color="darkorange", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"Inventory — {label}" if label else "Inventory")
        ax.set_xlabel("Event index")
        ax.set_ylabel("Shares")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "inventory.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_quotes(results: pd.DataFrame, run_dir: str, label: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))

        # Subsample for readability — plotting 22M points is unusable
        step = max(1, len(results) // 10_000)
        sub  = results.iloc[::step]

        if "mid" in sub.columns:
            ax.plot(sub.index, sub["mid"].values,       color="black",      linewidth=0.6, label="mid")
        if "bid_quote" in sub.columns:
            ax.plot(sub.index, sub["bid_quote"].values, color="steelblue",  linewidth=0.5, label="bid quote", alpha=0.7)
        if "ask_quote" in sub.columns:
            ax.plot(sub.index, sub["ask_quote"].values, color="firebrick",  linewidth=0.5, label="ask quote", alpha=0.7)

        ax.set_title(f"Mid vs Posted Quotes — {label}" if label else "Mid vs Posted Quotes")
        ax.set_xlabel("Event index")
        ax.set_ylabel("Price (USD)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "quotes.png"), dpi=150)
        plt.close(fig)
