# Route-Aware Attribution — Synthetic Benchmark

Reproducible synthetic benchmarks for the paper
*"Route-Aware Attribution of Multi-Class Overlay Patterns in Semiconductor
Manufacturing"* (Journal of Intelligent Manufacturing).

**This repository contains only procedurally generated synthetic data and code.**
No production, fab, equipment, or metrology data is included, and none is required
to run anything here. The real case-study data are proprietary and cannot be
shared.

## Repository layout

```
.
├── README.md
├── LICENSE                   
├── requirements.txt
├── .gitignore
├── src/
│   ├── generate_synthetic.py  # synthetic routing data (fixed seeds)
│   ├── screening.py           # chi-square + Cramér's V + ASR + Bonferroni FWER
│   └── run_experiments.py     # runs the three experiments, writes results/
├── data/
│   └── example_synthetic.csv  # one small generated snapshot (illustrative)
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


## Notes

- Equipment, step, and class identifiers in the synthetic data are arbitrary
  labels and do not correspond to any real tool, step, or product.
- `data/example_synthetic.csv` is a deliberately reduced snapshot (120 wafers ×
  20 steps) written on each run, so that the data format is visible without
  committing a full-scale dataset.
- If you use this code, please cite the paper.
