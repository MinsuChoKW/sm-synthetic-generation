# -*- coding: utf-8 -*-
"""
screening.py
============
Screening-attribution core for route-aware attribution of multi-class overlay
patterns.

The pipeline is applied independently to every process step:

  1. Build the (equipment x class) contingency table for the step.
  2. Omnibus chi-square test of independence (no continuity correction).
     Steps with fewer than two equipment nodes or two classes are not testable
     and are skipped.
  3. Effect-size gate.  Cramer's V = sqrt(chi2 / (n * min(r-1, k-1))) must
     reach tau_V; the gate is what separates a statistically significant but
     practically irrelevant step from a genuine routing signature.
  4. For steps that pass both the chi-square screen and the effect-size gate,
     compute an adjusted standardized residual (ASR) per cell.  Cells that are
     sparse in *both* the expected and the observed count (< MIN_COUNT) are
     excluded from the residual analysis.
  5. Cell-wise Bonferroni FWER control: the per-step family is the r * k cells
     of that step's table, so the two-sided critical value is
     z_crit = Phi^-1(1 - alpha / (2 * r * k)).  Only positive associations
     (ASR > z_crit) are attributed.
  6. ASR_REPORT is the reporting threshold used for the compressed tables in
     the paper; it is a presentation cut applied on top of the FWER gate, not a
     second inference step.

This module has no dependency on any particular data source: it consumes a long
frame with one row per (wafer, step) visit and the columns
STEP_ID / EQP_ID / CLASS.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# --- method constants (paper defaults) --------------------------------------
ALPHA = 0.05          # family-wise error rate, and omnibus screen level
TAU_V = 0.30          # Cramer's V effect-size gate
MIN_COUNT = 3         # sparse-cell rule: drop cell if expected < 3 AND observed < 3
ASR_REPORT = 10.0     # reporting threshold for the compressed tables

STEP_COL = "STEP_ID"
EQP_COL = "EQP_ID"
CLASS_COL = "CLASS"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """Omnibus screen outcome for a single step."""
    step: str
    n: int
    n_eqp: int
    n_class: int
    chi2: float
    dof: int
    p_value: float
    cramers_v: float
    chi2_significant: bool     # p < alpha
    gate_pass: bool            # p < alpha AND V >= tau_V
    z_crit: float              # Bonferroni critical value used for the cells


@dataclass
class CellResult:
    """One attributed (step, equipment, class) cell."""
    step: str
    eqp: str
    cls: str
    observed: int
    expected: float
    asr: float
    p_two_sided: float
    z_crit: float
    fwer_significant: bool
    reported: bool             # fwer_significant AND asr >= ASR_REPORT


# ---------------------------------------------------------------------------
# Contingency tables
# ---------------------------------------------------------------------------
def contingency_tables(frame: pd.DataFrame,
                       step_col: str = STEP_COL,
                       eqp_col: str = EQP_COL,
                       class_col: str = CLASS_COL) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield (step, equipment x class table) for every step in the frame."""
    counts = frame.groupby([step_col, eqp_col, class_col]).size()
    for step, sub in counts.groupby(level=0):
        table = sub.droplevel(0).unstack(fill_value=0)
        table = table.sort_index(axis=0).sort_index(axis=1)
        yield str(step), table


def cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    """Cramer's V for an r x k table with n observations."""
    denom = min(r - 1, k - 1)
    if denom <= 0 or n <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * denom)))


# ---------------------------------------------------------------------------
# Per-step analysis
# ---------------------------------------------------------------------------
def analyse_table(step: str,
                  table: pd.DataFrame,
                  alpha: float = ALPHA,
                  tau_v: float = TAU_V,
                  min_count: int = MIN_COUNT,
                  asr_report: float = ASR_REPORT) -> Tuple[Optional[StepResult], List[CellResult]]:
    """Run screen + attribution on one (equipment x class) table.

    Returns (StepResult, attributed cells).  The StepResult is None when the
    table is not testable (fewer than two rows or two columns).  Cells are
    returned only for steps that pass the gate, and only for positive
    associations that survive the Bonferroni FWER threshold.
    """
    observed = table.to_numpy()
    r, k = observed.shape
    if r < 2 or k < 2:
        return None, []

    chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)
    n = int(observed.sum())
    v = cramers_v(chi2, n, r, k)

    adj_alpha = alpha / (r * k)                       # Bonferroni over the step's cells
    z_crit = float(stats.norm.ppf(1.0 - adj_alpha / 2.0))

    chi2_significant = bool(p_value < alpha)
    gate_pass = bool(chi2_significant and v >= tau_v)

    step_result = StepResult(
        step=step, n=n, n_eqp=r, n_class=k,
        chi2=float(chi2), dof=int(dof), p_value=float(p_value), cramers_v=v,
        chi2_significant=chi2_significant, gate_pass=gate_pass, z_crit=z_crit,
    )
    if not gate_pass:
        return step_result, []

    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()

    cells: List[CellResult] = []
    for i in range(r):
        for j in range(k):
            o_ij = observed[i, j]
            e_ij = expected[i, j]
            # sparse-cell rule: excluded only when BOTH counts are small
            if e_ij < min_count and o_ij < min_count:
                continue
            if e_ij <= 0 or total <= 0:
                continue
            # variance correction of the standardized residual
            var = e_ij * (1.0 - row_sums[i] / total) * (1.0 - col_sums[j] / total)
            if var <= 0:
                continue
            asr = float((o_ij - e_ij) / np.sqrt(var))
            if asr <= z_crit:                          # positive associations only
                continue
            p_two = float(2.0 * stats.norm.sf(abs(asr)))
            cells.append(CellResult(
                step=step, eqp=str(table.index[i]), cls=str(table.columns[j]),
                observed=int(o_ij), expected=float(e_ij), asr=asr,
                p_two_sided=p_two, z_crit=z_crit,
                fwer_significant=True, reported=bool(asr >= asr_report),
            ))
    return step_result, cells


# ---------------------------------------------------------------------------
# Whole-frame driver
# ---------------------------------------------------------------------------
def screen_and_attribute(frame: pd.DataFrame,
                         alpha: float = ALPHA,
                         tau_v: float = TAU_V,
                         min_count: int = MIN_COUNT,
                         asr_report: float = ASR_REPORT,
                         step_col: str = STEP_COL,
                         eqp_col: str = EQP_COL,
                         class_col: str = CLASS_COL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the screening-attribution pipeline to every step of a long frame.

    Returns
    -------
    steps  : one row per testable step (chi2, V, gate decision).
    cells  : one row per attributed cell (positive, FWER-significant).
    """
    step_rows: List[Dict] = []
    cell_rows: List[Dict] = []
    for step, table in contingency_tables(frame, step_col, eqp_col, class_col):
        step_result, cells = analyse_table(step, table, alpha, tau_v, min_count, asr_report)
        if step_result is None:
            continue
        step_rows.append(vars(step_result))
        cell_rows.extend(vars(c) for c in cells)

    steps = pd.DataFrame(step_rows, columns=[f.name for f in StepResult.__dataclass_fields__.values()])
    cells = pd.DataFrame(cell_rows, columns=[f.name for f in CellResult.__dataclass_fields__.values()])
    return steps, cells


def attributed_cell_set(cells: pd.DataFrame, asr_threshold: float = 0.0) -> set:
    """Set of (step, eqp, class) tuples attributed at or above an ASR threshold."""
    if cells.empty:
        return set()
    sel = cells.loc[cells["asr"] >= asr_threshold]
    return set(zip(sel["step"], sel["eqp"], sel["cls"]))


def gate_pass_steps(steps: pd.DataFrame) -> set:
    """Set of steps that cleared the chi-square screen and the Cramer's V gate."""
    if steps.empty:
        return set()
    return set(steps.loc[steps["gate_pass"], "step"])
