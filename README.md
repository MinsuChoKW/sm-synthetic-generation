# Route-Aware Attribution — Synthetic Benchmark

Reproducible synthetic benchmarks for the paper
*"Route-Aware Attribution of Multi-Class Overlay Patterns in Semiconductor
Manufacturing"* (Journal of Intelligent Manufacturing).

**This repository contains only procedurally generated synthetic data and code.**
No production, fab, equipment, or metrology data is included, and none is required
to run anything here. The real case-study data are proprietary and cannot be
shared.

The synthetic benchmarks reproduce the ground-truth, robustness, and
interaction-effect analyses reported in Section 4.3.3 of the paper — i.e. that the
ASR + FWER screening–attribution pipeline recovers planted associations, controls
false positives under non-ideal conditions, and (by construction) misses purely
interaction-driven signals that a joint step-pair node recovers.

## What is and isn't here

Included (all synthetic):
- generation of synthetic routing data with planted step–equipment–class signals;
- the screening–attribution core (chi-square screen, Cramér's V gate, adjusted
  standardized residuals, Bonferroni FWER);
- the three synthetic experiments: signal-strength ground truth, non-ideal
  conditions (correlated routing, weak signal, missing data), and the
  single-step-vs-interaction test.

Not included (depends on proprietary data, cannot be released):
- the real case-study analysis, the τ_V sweep, the temporal splits, and the
  STEP_053 contemporaneous-cohort analysis, which all run on the real event log.

## Repository layout

```
.
├── README.md
├── LICENSE                    # (choose per institutional policy)
├── requirements.txt
├── .gitignore
├── src/
│   ├── generate_synthetic.py  # synthetic routing data (fixed seeds)
│   ├── screening.py           # chi-square + Cramér's V + ASR + Bonferroni FWER
│   └── run_experiments.py     # runs the three experiments, writes results/
├── data/
│   └── example_synthetic.csv  # one small generated snapshot (illustrative)
└── results/                   # tables written by run_experiments.py
```

## Requirements

```
python >= 3.9
numpy
scipy
pandas
```
Install: `pip install -r requirements.txt`

## Reproduce

```bash
python src/run_experiments.py
```
This regenerates the synthetic datasets (fixed seeds), prints a summary of each
experiment, and writes result tables to `results/`. Runtime is well under a
minute (the ground-truth and non-ideal conditions are averaged over 20 datasets
each). `--reps` and `--seed` override the defaults.

## Method

The screening–attribution core in `src/screening.py` is applied independently to
every process step, on that step's (equipment × class) contingency table:

| Stage | Rule |
|---|---|
| Omnibus screen | chi-square test of independence, no continuity correction, α = 0.05 |
| Effect-size gate | Cramér's V = √(χ² / (n · min(r−1, k−1))) ≥ τ_V = 0.30 |
| Sparse cells | cell dropped from the residual analysis when expected < 3 **and** observed < 3 |
| Attribution | adjusted standardized residual (ASR) with the variance correction, positive associations only |
| Multiplicity | cell-wise Bonferroni FWER over the step's r · k cells: z_crit = Φ⁻¹(1 − α / (2 r k)) |
| Reporting | ASR ≥ 10 (a presentation cut applied on top of the FWER gate, not a second inference step) |

## Signal model

Datasets are 600 wafers × 100 steps × 10 balanced classes, with 10 equipment
nodes per step — the scale of the case study. At a planted step, each class has a
designated equipment node, and a wafer follows its class's node with probability
*s* (the signal strength); otherwise it is routed by the background process. The
other 93 steps are null.

Under this parameterisation the expected Cramér's V of a planted step equals the
planted signal strength (plus a small O(dof/n) noise floor), so **τ_V = 0.30 is
the effect-size counterpart of a signal of strength 0.30**, and a deterministic
routing signature (*s* = 1) gives V = 1.0 — the value observed at the
photolithography steps of the case study.

Non-ideal conditions extend the same generator: *correlated routing* gives each
wafer a latent "line" factor with a fixed per-line equipment preference shared
across all steps, followed with probability ρ, which makes routing correlated
across steps and equipment marginals uneven; the line factor is drawn
independently of the class, so it is a pure nuisance factor. *Weak signal* lowers
*s*, and *missing data* deletes a fraction of the (wafer, step) records.

## Result → paper mapping

| Script output | Paper element |
|---|---|
| `results/experiment1_signal_strength.csv` | Table (synthetic validation), upper block |
| `results/experiment2_non_ideal.csv` | Table (synthetic validation), lower block |
| `results/experiment3_interaction.csv` | Interaction paragraph, Section 4.3.3 |

Headline numbers, 20 datasets per condition, base seed 20260914:

- **Ground truth.** Every planted step is recovered and every planted cell
  attributed (sensitivity = recovery = 1.00) at *s* = 0.50, 0.70, 0.90, with a
  null-step false-positive rate of 0.000. At *s* = 0.30 — exactly on the gate —
  recovery is partial (0.86). The `step_sensitivity_chi2` column shows what the
  gate buys: the chi-square screen alone finds every planted step at every *s*,
  but flags ≈ α (0.056) of the null steps; the effect-size gate drives that to
  zero.
- **Non-ideal conditions.** Sensitivity and cell recovery stay at 1.00 under
  ρ = 0.60 and ρ = 0.85, under weak signals *s* = 0.45 and 0.35, under 20 % and
  40 % missingness, and at 0.99 for the combined condition
  (ρ = 0.60, *s* = 0.35, 20 % missing). The null-step false-positive rate is
  0.000 throughout, i.e. at or below α. The one condition that is not recovered
  is the very weak signal *s* = 0.25, which falls below the gate (0.10) — the
  intended behaviour of an effect-size threshold.
- **Interaction.** The single-step control signal is detected in 20/20 datasets.
  The two interaction steps, whose marginal class distributions are uniform by
  construction, are detected in 0/20 by the per-step screen (mean V ≈ 0.12).
  Collapsing the two steps into a joint step-pair node recovers the signal in
  20/20 datasets at V = 1.00.

Seeds are fixed so the reported numbers regenerate deterministically. Small
differences (< a rounding unit) may arise across NumPy/SciPy versions; the pinned
versions in `requirements.txt` reproduce the values above.

## Notes

- Equipment, step, and class identifiers in the synthetic data are arbitrary
  labels and do not correspond to any real tool, step, or product.
- `data/example_synthetic.csv` is a deliberately reduced snapshot (120 wafers ×
  20 steps) written on each run, so that the data format is visible without
  committing a full-scale dataset.
- If you use this code, please cite the paper.
