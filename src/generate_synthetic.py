# -*- coding: utf-8 -*-
"""
generate_synthetic.py
=====================
Procedural generator for the synthetic routing benchmarks.

Everything produced here is drawn from a seeded NumPy generator; no fab,
equipment or metrology data is read, and the identifiers (STEP_xxx, EQP_xx,
CLASS_xx, Wxxxx) are arbitrary labels with no relation to any real tool, step
or product.

Data model
----------
A dataset is a long frame with one row per (wafer, step) visit:

    WAFER_ID   wafer identity            (600 wafers)
    CLASS      overlay pattern class     (10 balanced classes)
    STEP_ID    process step              (100 steps)
    EQP_ID     equipment node used       (10 nodes per step)

Signal model
------------
At a *planted* step, a class -> equipment map is drawn (a permutation of the
equipment nodes, so each class has its own designated node).  A wafer follows
its class's designated node with probability ``signal``; otherwise it is routed
by the background process.  The remaining steps are null: routing is the
background process alone and carries no class information.

Under this parameterisation the expected Cramer's V of a planted step is equal
to the signal strength itself (plus a small O(dof/n) noise floor), which is why
the effect-size gate tau_V = 0.30 is the natural counterpart of a signal of
strength 0.30.  A perfectly deterministic routing signature (signal = 1) gives
V = 1.0, the value observed at the photolithography steps of the case study.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# --- label formats ----------------------------------------------------------
BASE_SEED = 20260914
STEP_FMT = "STEP_{:03d}"
EQP_FMT = "EQP_{:02d}"
CLASS_FMT = "CLASS_{:02d}"
WAFER_FMT = "W{:04d}"
JOINT_SEP = "+"

WAFER_COL = "WAFER_ID"
CLASS_COL = "CLASS"
STEP_COL = "STEP_ID"
EQP_COL = "EQP_ID"
COLUMNS = [WAFER_COL, CLASS_COL, STEP_COL, EQP_COL]


@dataclass
class SyntheticConfig:
    """Generator settings.  Defaults match the case-study scale."""
    n_wafers: int = 600
    n_steps: int = 100
    n_classes: int = 10
    n_eqp: int = 10           # equipment nodes available at every step
    n_planted: int = 7        # steps carrying a class -> equipment signature
    signal: float = 0.70      # probability a wafer follows its designated node
    rho: float = 0.0          # strength of the latent line factor (correlated routing)
    n_lines: int = 3          # number of latent lines
    dirichlet_alpha: float = 1.0   # concentration of the per-line node preference
    missing: float = 0.0      # fraction of (wafer, step) records deleted
    seed: int = BASE_SEED


@dataclass
class SyntheticDataset:
    """A generated dataset together with its ground truth."""
    frame: pd.DataFrame
    config: SyntheticConfig
    planted_steps: List[str] = field(default_factory=list)
    planted_cells: Set[Tuple[str, str, str]] = field(default_factory=set)
    all_steps: List[str] = field(default_factory=list)

    @property
    def null_steps(self) -> List[str]:
        planted = set(self.planted_steps)
        return [s for s in self.all_steps if s not in planted]


@dataclass
class InteractionDataset:
    """A dataset with one single-step control signal and one interaction-only pair."""
    frame: pd.DataFrame
    config: SyntheticConfig
    control_step: str
    step_a: str
    step_b: str
    control_cells: Set[Tuple[str, str, str]] = field(default_factory=set)
    joint_cells: Set[Tuple[str, str, str]] = field(default_factory=set)
    joint_label: str = "JOINT"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _labels(cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.array([STEP_FMT.format(i) for i in range(cfg.n_steps)])
    eqps = np.array([EQP_FMT.format(i) for i in range(cfg.n_eqp)])
    classes = np.array([CLASS_FMT.format(i) for i in range(cfg.n_classes)])
    return steps, eqps, classes


def _balanced_classes(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    """Balanced class assignment (n_wafers / n_classes wafers per class), shuffled."""
    per_class, remainder = divmod(cfg.n_wafers, cfg.n_classes)
    codes = np.repeat(np.arange(cfg.n_classes), per_class)
    if remainder:
        codes = np.concatenate([codes, np.arange(remainder)])
    return rng.permutation(codes)


def _background_draws(u_line: np.ndarray, u_pick: np.ndarray,
                      wafer_cum: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """Background routing: line-biased with probability rho, else uniform.

    Both branches consume the same uniform draw so that the random stream is
    identical for every rho, which keeps datasets paired across conditions.
    """
    uniform_choice = np.minimum((u_pick * cfg.n_eqp).astype(np.int64), cfg.n_eqp - 1)
    line_choice = np.minimum((wafer_cum < u_pick[:, None]).sum(axis=1), cfg.n_eqp - 1)
    return np.where(u_line < cfg.rho, line_choice, uniform_choice)


def _assemble(cfg: SyntheticConfig, rng: np.random.Generator,
              eqp_matrix: np.ndarray, class_codes: np.ndarray) -> pd.DataFrame:
    """Turn an (n_steps x n_wafers) routing matrix into a long frame."""
    steps, eqps, classes = _labels(cfg)
    frame = pd.DataFrame({
        WAFER_COL: np.tile(np.array([WAFER_FMT.format(w) for w in range(cfg.n_wafers)]),
                           cfg.n_steps),
        CLASS_COL: np.tile(classes[class_codes], cfg.n_steps),
        STEP_COL: np.repeat(steps, cfg.n_wafers),
        EQP_COL: eqps[eqp_matrix.ravel()],
    }, columns=COLUMNS)

    if cfg.missing > 0.0:
        keep = rng.random(len(frame)) >= cfg.missing
        frame = frame.loc[keep]

    return frame.sort_values([WAFER_COL, STEP_COL], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate(cfg: SyntheticConfig) -> SyntheticDataset:
    """Generate one synthetic routing dataset with planted step signatures."""
    rng = np.random.default_rng(cfg.seed)
    steps, eqps, classes = _labels(cfg)

    class_codes = _balanced_classes(rng, cfg)
    planted_idx = np.sort(rng.choice(cfg.n_steps, size=cfg.n_planted, replace=False))
    # class -> equipment map for each planted step
    mappings = {int(s): rng.permutation(cfg.n_eqp)[:cfg.n_classes] for s in planted_idx}

    lines = rng.integers(0, cfg.n_lines, size=cfg.n_wafers)
    prefs = rng.dirichlet(np.full(cfg.n_eqp, cfg.dirichlet_alpha), size=cfg.n_lines)
    wafer_cum = np.cumsum(prefs, axis=1)[lines]

    planted_set = set(int(s) for s in planted_idx)
    eqp_matrix = np.empty((cfg.n_steps, cfg.n_wafers), dtype=np.int64)
    for step in range(cfg.n_steps):
        # draw counts are independent of signal/rho/missing so that datasets
        # generated from the same seed stay paired across conditions
        u_follow = rng.random(cfg.n_wafers)
        u_line = rng.random(cfg.n_wafers)
        u_pick = rng.random(cfg.n_wafers)

        routed = _background_draws(u_line, u_pick, wafer_cum, cfg)
        if step in planted_set:
            designated = mappings[step][class_codes]
            routed = np.where(u_follow < cfg.signal, designated, routed)
        eqp_matrix[step] = routed

    frame = _assemble(cfg, rng, eqp_matrix, class_codes)

    planted_steps = [STEP_FMT.format(int(s)) for s in planted_idx]
    planted_cells = {
        (STEP_FMT.format(int(s)), eqps[mappings[int(s)][c]], classes[c])
        for s in planted_idx for c in range(cfg.n_classes)
    }
    return SyntheticDataset(frame=frame, config=cfg, planted_steps=planted_steps,
                            planted_cells=planted_cells, all_steps=list(steps))


def generate_interaction(cfg: SyntheticConfig) -> InteractionDataset:
    """Generate a dataset with a single-step control signal and an interaction-only pair.

    The interaction pair (step_a, step_b) is built so that the equipment used at
    each step on its own is uniform for every class -- there is no single-step
    association to find -- while the *pair* of nodes identifies the class:
    node_b = (node_a + class) mod n_eqp, so (node_b - node_a) mod n_eqp recovers
    the class exactly.
    """
    rng = np.random.default_rng(cfg.seed)
    steps, eqps, classes = _labels(cfg)

    class_codes = _balanced_classes(rng, cfg)
    control_idx, a_idx, b_idx = (int(x) for x in rng.choice(cfg.n_steps, size=3, replace=False))
    control_map = rng.permutation(cfg.n_eqp)[:cfg.n_classes]

    lines = rng.integers(0, cfg.n_lines, size=cfg.n_wafers)
    prefs = rng.dirichlet(np.full(cfg.n_eqp, cfg.dirichlet_alpha), size=cfg.n_lines)
    wafer_cum = np.cumsum(prefs, axis=1)[lines]

    eqp_matrix = np.empty((cfg.n_steps, cfg.n_wafers), dtype=np.int64)
    node_a = None
    for step in range(cfg.n_steps):
        u_follow = rng.random(cfg.n_wafers)
        u_line = rng.random(cfg.n_wafers)
        u_pick = rng.random(cfg.n_wafers)

        routed = _background_draws(u_line, u_pick, wafer_cum, cfg)
        if step == control_idx:
            routed = np.where(u_follow < cfg.signal, control_map[class_codes], routed)
        elif step == a_idx:
            node_a = routed                      # uniform / background, class-independent
        eqp_matrix[step] = routed

    # step_b is fixed after the loop so that it can depend on step_a's outcome
    eqp_matrix[b_idx] = (node_a + class_codes) % cfg.n_eqp

    frame = _assemble(cfg, rng, eqp_matrix, class_codes)

    control_cells = {(STEP_FMT.format(control_idx), eqps[control_map[c]], classes[c])
                     for c in range(cfg.n_classes)}
    joint_cells = {(joint_label(STEP_FMT.format(a_idx), STEP_FMT.format(b_idx)),
                    eqps[a] + JOINT_SEP + eqps[(a + c) % cfg.n_eqp], classes[c])
                   for c in range(cfg.n_classes) for a in range(cfg.n_eqp)}

    return InteractionDataset(
        frame=frame, config=cfg,
        control_step=STEP_FMT.format(control_idx),
        step_a=STEP_FMT.format(a_idx), step_b=STEP_FMT.format(b_idx),
        control_cells=control_cells, joint_cells=joint_cells,
        joint_label=joint_label(STEP_FMT.format(a_idx), STEP_FMT.format(b_idx)),
    )


# ---------------------------------------------------------------------------
# Joint step-pair node
# ---------------------------------------------------------------------------
def joint_label(step_a: str, step_b: str) -> str:
    return "JOINT_{}_{}".format(step_a, step_b)


def joint_node_frame(frame: pd.DataFrame, step_a: str, step_b: str) -> pd.DataFrame:
    """Collapse two steps into a single joint node ("EQP_a+EQP_b") per wafer.

    The result is a long frame in the same shape as the input, so the identical
    screening-attribution core can be applied to it without modification.
    """
    a = frame.loc[frame[STEP_COL] == step_a, [WAFER_COL, CLASS_COL, EQP_COL]]
    b = frame.loc[frame[STEP_COL] == step_b, [WAFER_COL, EQP_COL]]
    merged = a.merge(b, on=WAFER_COL, suffixes=("_A", "_B"))
    return pd.DataFrame({
        WAFER_COL: merged[WAFER_COL].to_numpy(),
        CLASS_COL: merged[CLASS_COL].to_numpy(),
        STEP_COL: joint_label(step_a, step_b),
        EQP_COL: merged[EQP_COL + "_A"].to_numpy() + JOINT_SEP + merged[EQP_COL + "_B"].to_numpy(),
    }, columns=COLUMNS)


if __name__ == "__main__":
    ds = generate(SyntheticConfig())
    print(ds.frame.head())
    print("rows={} wafers={} steps={} planted={}".format(
        len(ds.frame), ds.frame[WAFER_COL].nunique(),
        ds.frame[STEP_COL].nunique(), ds.planted_steps))
