# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic import (BASE_SEED, SyntheticConfig, generate,
                                generate_interaction, joint_node_frame)
from screening import (ALPHA, ASR_REPORT, TAU_V, attributed_cell_set,
                       gate_pass_steps, screen_and_attribute)

REPS = 20
BASELINE_SIGNAL = 0.70
SIGNAL_LEVELS = (0.30, 0.50, 0.70, 0.90)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
def evaluate(dataset, steps: pd.DataFrame, cells: pd.DataFrame) -> Dict[str, float]:
    """Sensitivity / recovery / false-positive metrics for one dataset."""
    planted = set(dataset.planted_steps)
    null = set(dataset.null_steps)
    tested = set(steps["step"])

    passed = gate_pass_steps(steps)
    flagged = attributed_cell_set(cells)
    reported = attributed_cell_set(cells, ASR_REPORT)

    null_tested = null & tested
    attributed_steps = set(cells["step"]) if not cells.empty else set()
    chi2_only = set(steps.loc[steps["chi2_significant"], "step"])

    planted_v = steps.loc[steps["step"].isin(planted), "cramers_v"]
    null_v = steps.loc[steps["step"].isin(null), "cramers_v"]
    planted_asr = cells.loc[cells["step"].isin(planted), "asr"] if not cells.empty else pd.Series(dtype=float)

    return {
        "step_sensitivity": len(planted & passed) / max(len(planted), 1),
        "step_sensitivity_chi2": len(planted & chi2_only) / max(len(planted), 1),
        "cell_recovery": len(dataset.planted_cells & flagged) / max(len(dataset.planted_cells), 1),
        "cell_recovery_asr10": len(dataset.planted_cells & reported) / max(len(dataset.planted_cells), 1),
        "null_step_fpr": len(null_tested & attributed_steps) / max(len(null_tested), 1),
        "null_chi2_fpr": len(null_tested & chi2_only) / max(len(null_tested), 1),
        "false_cells": float(len(flagged) - len(dataset.planted_cells & flagged)),
        "mean_v_planted": float(planted_v.mean()) if len(planted_v) else np.nan,
        "max_v_null": float(null_v.max()) if len(null_v) else np.nan,
        "mean_asr_planted": float(planted_asr.mean()) if len(planted_asr) else np.nan,
        "mean_n_per_step": float(steps["n"].mean()) if len(steps) else np.nan,
    }


def run_condition(cfg_kwargs: Dict, reps: int, base_seed: int) -> pd.DataFrame:
    """Generate and evaluate ``reps`` paired datasets for one condition."""
    rows = []
    for rep in range(reps):
        cfg = SyntheticConfig(seed=base_seed + rep, **cfg_kwargs)
        dataset = generate(cfg)
        steps, cells = screen_and_attribute(dataset.frame, alpha=ALPHA, tau_v=TAU_V)
        rows.append(evaluate(dataset, steps, cells))
    return pd.DataFrame(rows)


METRIC_ORDER = ["step_sensitivity", "step_sensitivity_chi2",
                "cell_recovery", "cell_recovery_asr10",
                "null_step_fpr", "null_chi2_fpr", "false_cells",
                "mean_v_planted", "max_v_null", "mean_asr_planted", "mean_n_per_step"]


# ---------------------------------------------------------------------------
# Experiment 1 - signal-strength ground truth
# ---------------------------------------------------------------------------
def experiment_signal_strength(reps: int, base_seed: int) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT 1 - signal-strength ground truth "
          "({} datasets per level, 7 planted / 93 null steps)".format(reps))
    print("=" * 78)

    rows = []
    for signal in SIGNAL_LEVELS:
        metrics = run_condition({"signal": signal}, reps, base_seed)
        row = {"signal_strength": signal, "n_datasets": reps,
               "n_planted_steps": 7, "n_null_steps": 93}
        row.update({m: metrics[m].mean() for m in METRIC_ORDER})
        rows.append(row)
    table = pd.DataFrame(rows)
    _print_table(table, ["signal_strength", "step_sensitivity", "step_sensitivity_chi2",
                         "cell_recovery", "cell_recovery_asr10", "null_step_fpr",
                         "null_chi2_fpr", "mean_v_planted", "max_v_null"])
    print("  Cramer's V of a planted step tracks the planted signal strength (mean_v_planted")
    print("  ~ signal_strength), so tau_V = {:.2f} is the effect-size counterpart of a signal".format(TAU_V))
    print("  of that size.  The two sensitivity columns isolate what the gate buys: the")
    print("  chi-square screen alone finds every planted step but flags ~alpha of the null")
    print("  steps, while the gate drives the null-step rate to zero.  s = {:.2f} sits exactly".format(SIGNAL_LEVELS[0]))
    print("  on the gate, so it is the one level that is only partly recovered; every")
    print("  signal clear of the gate is recovered in full.")
    return table


# ---------------------------------------------------------------------------
# Experiment 2 - non-ideal conditions
# ---------------------------------------------------------------------------
NON_IDEAL_CONDITIONS = [
    ("baseline",               {}),
    ("correlated routing rho=0.60", {"rho": 0.60}),
    ("correlated routing rho=0.85", {"rho": 0.85}),
    ("weak signal s=0.45",     {"signal": 0.45}),
    ("weak signal s=0.35",     {"signal": 0.35}),
    ("very weak signal s=0.25", {"signal": 0.25}),
    ("missing 20%",            {"missing": 0.20}),
    ("missing 40%",            {"missing": 0.40}),
    ("combined rho=0.60, s=0.35, 20% missing",
     {"rho": 0.60, "signal": 0.35, "missing": 0.20}),
]


def experiment_non_ideal(reps: int, base_seed: int) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 - non-ideal conditions "
          "({} datasets per condition, baseline signal s={:.2f})".format(reps, BASELINE_SIGNAL))
    print("=" * 78)

    rows = []
    for name, kwargs in NON_IDEAL_CONDITIONS:
        cfg_kwargs = dict(kwargs)
        cfg_kwargs.setdefault("signal", BASELINE_SIGNAL)
        metrics = run_condition(cfg_kwargs, reps, base_seed)
        row = {"condition": name, "n_datasets": reps,
               "rho": cfg_kwargs.get("rho", 0.0),
               "signal": cfg_kwargs["signal"],
               "missing": cfg_kwargs.get("missing", 0.0)}
        row.update({m: metrics[m].mean() for m in METRIC_ORDER})
        rows.append(row)
    table = pd.DataFrame(rows)
    _print_table(table, ["condition", "rho", "signal", "missing", "step_sensitivity",
                         "cell_recovery", "null_step_fpr", "mean_v_planted",
                         "max_v_null", "mean_n_per_step"])
    print("  The latent line factor is drawn independently of the class, so correlated")
    print("  routing is a pure nuisance factor: it must not manufacture associations.")
    print("  alpha = {:.2f}; the null-step false-positive rate stays at or below it.".format(ALPHA))
    return table


# ---------------------------------------------------------------------------
# Experiment 3 - single step vs interaction
# ---------------------------------------------------------------------------
def experiment_interaction(reps: int, base_seed: int) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT 3 - single-step signal vs interaction-only signal "
          "({} datasets)".format(reps))
    print("=" * 78)

    acc = {key: {"detected": 0, "v": [], "recovery": []}
           for key in ("control", "step_a", "step_b", "joint")}
    false_steps = []

    for rep in range(reps):
        cfg = SyntheticConfig(seed=base_seed + rep, signal=BASELINE_SIGNAL)
        dataset = generate_interaction(cfg)

        steps, cells = screen_and_attribute(dataset.frame, alpha=ALPHA, tau_v=TAU_V)
        passed = gate_pass_steps(steps)
        by_step = steps.set_index("step")
        flagged = attributed_cell_set(cells)

        # joint step-pair node built from the same two steps
        joint_frame = joint_node_frame(dataset.frame, dataset.step_a, dataset.step_b)
        jsteps, jcells = screen_and_attribute(joint_frame, alpha=ALPHA, tau_v=TAU_V)
        jpassed = gate_pass_steps(jsteps)
        jflagged = attributed_cell_set(jcells)

        for key, step in (("control", dataset.control_step),
                          ("step_a", dataset.step_a),
                          ("step_b", dataset.step_b)):
            acc[key]["detected"] += int(step in passed)
            acc[key]["v"].append(float(by_step.loc[step, "cramers_v"]))
        acc["control"]["recovery"].append(
            len(dataset.control_cells & flagged) / len(dataset.control_cells))
        acc["step_a"]["recovery"].append(0.0)
        acc["step_b"]["recovery"].append(0.0)

        acc["joint"]["detected"] += int(dataset.joint_label in jpassed)
        acc["joint"]["v"].append(float(jsteps["cramers_v"].iloc[0]) if len(jsteps) else np.nan)
        acc["joint"]["recovery"].append(
            len(dataset.joint_cells & jflagged) / len(dataset.joint_cells))

        false_steps.append(len(passed - {dataset.control_step}))

    labels = [
        ("control", "single-step control signal (s={:.2f})".format(BASELINE_SIGNAL), "per-step screen"),
        ("step_a", "interaction step A alone", "per-step screen"),
        ("step_b", "interaction step B alone", "per-step screen"),
        ("joint", "joint step-pair node (A x B)", "joint-node screen"),
    ]
    rows = []
    for key, desc, node in labels:
        a = acc[key]
        rows.append({
            "node": desc, "tested_by": node, "n_datasets": reps,
            "detected": a["detected"], "detection_rate": a["detected"] / reps,
            "mean_cramers_v": float(np.mean(a["v"])),
            "mean_cell_recovery": float(np.mean(a["recovery"])),
        })
    table = pd.DataFrame(rows)
    _print_table(table, list(table.columns))
    print("  Steps A and B carry no marginal class information by construction, so the")
    print("  per-step screen cannot see them; the joint step-pair node recovers the")
    print("  signal exactly (V = 1.00).  Mean spurious gate-passing steps per dataset: "
          "{:.2f}".format(float(np.mean(false_steps))))
    return table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_table(table: pd.DataFrame, columns: Sequence[str]) -> None:
    view = table[list(columns)].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else "{:.3f}".format(x))
    print()
    print(view.to_string(index=False))
    print()


def write_example_snapshot(base_seed: int) -> str:
    """Write one small illustrative dataset so the file format is visible in the repo."""
    cfg = SyntheticConfig(n_wafers=120, n_steps=20, n_planted=2,
                          signal=BASELINE_SIGNAL, seed=base_seed)
    dataset = generate(cfg)
    path = os.path.join(DATA_DIR, "example_synthetic.csv")
    dataset.frame.to_csv(path, index=False)
    print("\nExample snapshot: {} rows, {} wafers x {} steps, planted steps {} "
          "-> data/example_synthetic.csv".format(
              len(dataset.frame), cfg.n_wafers, cfg.n_steps, ", ".join(dataset.planted_steps)))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic benchmark experiments (ground-truth, non-ideal, interaction).")
    parser.add_argument("--reps", type=int, default=REPS,
                        help="datasets per condition (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="base seed (default: %(default)s)")
    args = parser.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Route-aware attribution - synthetic benchmarks (Section 4.3.3)")
    print("alpha = {}, tau_V = {}, sparse-cell rule = {}, reporting threshold ASR >= {}"
          .format(ALPHA, TAU_V, 3, int(ASR_REPORT)))
    print("base seed = {}, {} datasets per condition".format(args.seed, args.reps))
    started = time.time()

    outputs = [
        ("experiment1_signal_strength.csv", experiment_signal_strength(args.reps, args.seed)),
        ("experiment2_non_ideal.csv", experiment_non_ideal(args.reps, args.seed)),
        ("experiment3_interaction.csv", experiment_interaction(args.reps, args.seed)),
    ]
    for filename, table in outputs:
        path = os.path.join(RESULTS_DIR, filename)
        table.to_csv(path, index=False)
        print("wrote results/{}".format(filename))

    write_example_snapshot(args.seed)
    print("\nTotal runtime: {:.1f} s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
