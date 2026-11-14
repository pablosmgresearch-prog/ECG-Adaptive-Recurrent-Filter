# Progress log

A short record of decisions, problems and next steps. Every entry is
taken from the git history (`git log --date=short`) and the README; dates
are the real commit dates.

## 2026-11-11 — Repository created

- `31d5bd3` Initial commit (README).
- `94f03fa` Uploaded the Thakor & Zhu (1991) paper PDF for reference.
  - **Problem found later:** the brief forbids committing paper PDFs. See
    2026-11-14.

## 2026-11-14 — Baseline, verification, modification and experiments

**Baseline**
- `17e23af` Implemented the LMS reference-noise canceller (`src/ARF.py`),
  the synthetic ECG generator from the course guide, and a first FFT test.
- **Decision:** keep the paper's update convention, `w += 2·mu·e·x`, and
  document that the guide's `mu` equals 2·`mu_paper`.

**Verification**
- `e1ea19c` Added pass/fail checks (`tests/test_arf.py`). They cover:
  - the guide's hand-computed example;
  - limiting cases;
  - an independent transcription of the update;
  - recovery of the true pathway;
  - SNR gain on development seeds 0–19.

  Also added the first three JSON test cases.
- `5ce4750` Reworked the visual test: cases that differ only in a fixed `mu`
  are plotted together. Added test cases 4–14 (`mu` sweeps).

**Modification**
- `6afc7e7` Implemented the change, DCT-LMS (`src/DCT-LMS_ARF.py`), and
  `tests/compare.py` (both methods on the JSON cases at the same `mu`).
  - **Decision:** transform-domain LMS with per-coefficient power
    normalisation (Narayan et al. 1983; Beaufays 1995).
  - **Ablation:** with `normalize=False` it is exactly LMS.
- `1753b06` Switched runtime measurement to `timeit`, which turns garbage
  collection off while timing.

**Experiments**
- `8b79b6b` Added the paired statistical comparison (`tests/paired_stats.py`)
  and general clean-up.
  - **Protocol:**
    - validation seeds 100–109 for tuning, test seeds 1000–1019;
    - three reference scales and two stress cases (changing pathway,
      leakage);
    - median paired difference with a bootstrap CI, win count, and
      Wilcoxon signed-rank test.
  - **Problem:** the first tuning grid picked settings on its edge. The
    grid was widened (README).
- `867488d` Merged the `DCT-LMS` branch (pull request #1).

**Repository**
- `9a4cb96` Pinned the tested environment in `requirements.txt` (`pip freeze`).
- `49a0ab3` Removed the paper PDF from the working tree.
  - **Still open:** it remains in the history (`94f03fa`).
- `7540a13` All scripts write their results to CSV files in `results/`.

## 2026-11-14 — Submission preparation (branch `submission-prep`)

- `e0ff4ac` `.gitignore` blocks `*.pdf` except `report/*.pdf`.
- `69a7529` `paired_stats.py` now saves machine-readable outputs:
  - `paired_results.csv`, `tuning.csv`, `summary.csv`;
  - `learning_curves.csv`, `r_peaks.csv`;
  - `run_info.json`.

  `make_figures.py` builds the four report figures from those files.
  `plt.show()` now only runs with `--show`.
- `69b3a68` Added a `--quick` smoke run (3 + 3 seeds, reduced grid, about
  14 s) that writes to `results/smoke/`.
- `3af148b` Refined the tuning grid: 16 log-spaced `mu` values and M = 1
  added, the same 96 settings for both methods.
  - **Reason:** LMS's best `mu` scales with 1/scale², so a coarse grid
    could penalise it.
  - **Outcome:**
    - The scale-0.5 result became a small significant gain for DCT-LMS.
    - The changing-pathway deficit shrank from −0.66 to −0.26 dB.
    - The leakage result flipped sign again, now −0.08 dB. It is reported
      as no practical difference.
- `f04d832` Measured per-sample runtime (`tests/runtime.py`).
  - **Finding:** runtime hardly depends on M at these sizes; Python loop
    overhead dominates. DCT-LMS is about 3× slower.
- This entry: progress log, README update, report scaffold.

## Next steps

- Decide how to handle the paper PDF that remains in the git history
  (make the repository private, or rewrite history before tagging).
- Merge `submission-prep` and tag `v1.0-submission`, then record the
  commit hash.
