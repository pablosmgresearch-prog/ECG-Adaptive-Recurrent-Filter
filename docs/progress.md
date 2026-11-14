# Progress log

A short record of decisions, problems and next steps. Every entry is
taken from the git history (`git log --date=short`) and the README; dates
are the real commit dates. Hashes are the ones after the history rewrite
of 2026-11-14 (see the last entry).

## 2026-11-11 — Repository created

- `d5ba8fe` Initial commit (README).
- The Thakor & Zhu (1991) paper PDF was also uploaded for reference.
  - **Problem found later:** the brief forbids committing paper PDFs. That
    commit has since been removed from the history (see 2026-11-14).

## 2026-11-14 — Baseline, verification, modification and experiments

**Baseline**
- `be39144` Implemented the LMS reference-noise canceller (`src/ARF.py`),
  the synthetic ECG generator from the course guide, and a first FFT test.
- **Decision:** keep the paper's update convention, `w += 2·mu·e·x`, and
  document that the guide's `mu` equals 2·`mu_paper`.

**Verification**
- `71ac456` Added pass/fail checks (`tests/test_arf.py`). They cover:
  - the guide's hand-computed example;
  - limiting cases;
  - an independent transcription of the update;
  - recovery of the true pathway;
  - SNR gain on development seeds 0–19.

  Also added the first three JSON test cases.
- `b4e1b24` Reworked the visual test: cases that differ only in a fixed `mu`
  are plotted together. Added test cases 4–14 (`mu` sweeps).

**Modification**
- `c001572` Implemented the change, DCT-LMS (`src/DCT-LMS_ARF.py`), and
  `tests/compare.py` (both methods on the JSON cases at the same `mu`).
  - **Decision:** transform-domain LMS with per-coefficient power
    normalisation (Narayan et al. 1983; Beaufays 1995).
  - **Ablation:** with `normalize=False` it is exactly LMS.
- `41b17c9` Switched runtime measurement to `timeit`, which turns garbage
  collection off while timing.

**Experiments**
- `7349525` Added the paired statistical comparison (`tests/paired_stats.py`)
  and general clean-up.
  - **Protocol:**
    - validation seeds 100–109 for tuning, test seeds 1000–1019;
    - three reference scales and two stress cases (changing pathway,
      leakage);
    - median paired difference with a bootstrap CI, win count, and
      Wilcoxon signed-rank test.
  - **Problem:** the first tuning grid picked settings on its edge. The
    grid was widened (README).
- `10ee817` Merged the `DCT-LMS` branch (pull request #1).

**Repository**
- `bac4839` Pinned the tested environment in `requirements.txt` (`pip freeze`).
- The paper PDF was deleted from the working tree. (That commit is gone
  too after the rewrite, because it became empty.)
- `aaff69c` All scripts write their results to CSV files in `results/`.

## 2026-11-14 — Submission preparation (branch `submission-prep`)

- `1b1e078` `.gitignore` blocks `*.pdf` except `report/*.pdf`.
- `eea1af3` `paired_stats.py` now saves machine-readable outputs:
  - `paired_results.csv`, `tuning.csv`, `summary.csv`;
  - `learning_curves.csv`, `r_peaks.csv`;
  - `run_info.json`.

  `make_figures.py` builds the four report figures from those files.
  `plt.show()` now only runs with `--show`.
- `c75bd87` Added a `--quick` smoke run (3 + 3 seeds, reduced grid, about
  14 s) that writes to `results/smoke/`.
- `f2b1bab` Refined the tuning grid: 16 log-spaced `mu` values and M = 1
  added, the same 96 settings for both methods.
  - **Reason:** LMS's best `mu` scales with 1/scale², so a coarse grid
    could penalise it.
  - **Outcome:**
    - The scale-0.5 result became a small significant gain for DCT-LMS.
    - The changing-pathway deficit shrank from −0.66 to −0.26 dB.
    - The leakage result flipped sign again, now −0.08 dB. It is reported
      as no practical difference.
- `41de3d6` Measured per-sample runtime (`tests/runtime.py`).
  - **Finding:** runtime hardly depends on M at these sizes; Python loop
    overhead dominates. DCT-LMS is about 3× slower.
- `04e8654` Added this progress log. `3ef07bb` Documented the full run
  order, outputs and figures in the README.
- `8310f98` An AI-written report draft was added, then removed in
  `e43a3b3` (clean-up), together with `make_figures.py`.
  - **Problem:** without `make_figures.py` the committed figures could
    not be regenerated.
  - **Decision:** `c974cef` restored it. Re-running it gives the same
    R-peak numbers as the README and byte-identical PNGs.

## 2026-11-14 — History rewrite, merge and tag

- **Decision:** remove the paper PDF from the whole history instead of
  making the repository private.
  - Rewrote `main`, `DCT-LMS` and `submission-prep` with
    `git filter-branch`, dropping every `*.pdf` from every commit. The two
    commits that only added and deleted the PDF became empty and were
    dropped. All commit hashes changed.
  - In the same pass, the commit messages written with the AI assistant
    were shortened.
  - Checked afterwards: no commit on any branch contains a PDF, and the
    final tree is unchanged apart from the restored `make_figures.py`.
- Merged `submission-prep` into `main` (fast-forward) and tagged
  `v1.0-submission`. The hash is given by `git rev-parse v1.0-submission`.

## Next steps

- Force-push `main`, `DCT-LMS` and `submission-prep`, and push the tag.
- The old commits stay visible on GitHub through pull request #1 and
  cached views even after a force-push. To purge the PDF there completely,
  ask GitHub Support to remove the cached views and the PR refs.
- Write the technical report in `report/`, with the `v1.0-submission`
  hash in its reproduction section.
