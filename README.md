# ECG Adaptive Noise Cancellation: LMS vs DCT-LMS

Course project for EEP55C36 Computational Methods and Algorithms, **Track B
(Biomedical: adaptive ECG noise cancellation)**.

The project reproduces the reference-noise canceller from

> N. V. Thakor and Y.-S. Zhu, "Applications of Adaptive Filtering to ECG
> Analysis: Noise Cancellation and Arrhythmia Detection," *IEEE Transactions
> on Biomedical Engineering*, 38(8), 785–794, 1991.
> https://doi.org/10.1109/10.83591

and compares it with one modification: **transform-domain (DCT) LMS**.
Arrhythmia detection from the paper is out of scope.

All signals are **synthetic ECG-like waveforms**. No clinical claims are made.

## The problem

A measured ECG `d = s + v` is the clean ECG `s` plus interference `v` (here
50 Hz mains hum through an unknown path). A second **reference** input `r`
picks up the interference but not the ECG. An adaptive FIR filter learns to
predict `v` from `r`. The cleaned ECG is the residual `e = d − y`, where `y`
is that prediction.

## Baseline: LMS ([src/ARF.py](src/ARF.py))

Transversal filter with M taps, updated with LMS (paper Eq. 4):

```
x_k     = [r[k], r[k-1], ..., r[k-M+1]]^T    (samples before the start are zero)
y[k]    = w_k^T x_k                          (interference estimate)
e[k]    = d[k] - y[k]                        (cleaned ECG)
w_{k+1} = w_k + 2·mu·e[k]·x_k
```

The factor 2 follows the paper. The course guide writes `w += mu·e·x`, which
is the same update with `mu_guide = 2·mu_paper`. Weights start at zero for
every recording. The cost is O(M) per sample.

`ARF.py` also contains `make_ecg` (the guide's synthetic data generator) and
`snr_db`.

## Modification: DCT-LMS ([src/DCT-LMS_ARF.py](src/DCT-LMS_ARF.py))

The tap vector is rotated into the DCT domain, and every DCT coefficient gets
its own step size, normalised by a running estimate of its power:

```
u_k     = C x_k                              (C: M x M orthonormal DCT-II, built with scipy)
y[k]    = v_k^T u_k
e[k]    = d[k] - y[k]
P_k     = beta·P_{k-1} + (1 - beta)·u_k²     (per-coefficient power, bias-corrected)
v_{k+1} = v_k + 2·mu·e[k]·u_k / (P_k + eps)
```

- **Why it may help.** The reference is a strong 50 Hz sine plus weak white
  noise. Its autocorrelation matrix therefore has a large eigenvalue spread,
  and plain LMS learns the weak (broadband) directions very slowly. The DCT
  roughly decorrelates the taps. Per-coefficient normalisation then evens out
  the convergence speed.
- **Where it may hurt.** Weak coefficients get large steps. The ECG acts as
  noise to the filter, so this can make the adapted weights jumpier,
  especially at large `mu`. The strong 50 Hz direction also adapts more
  slowly than in LMS.
- **Cost.** O(M²) per sample because of the DCT matrix product, against
  O(M) for LMS.
- **Ablation.** `normalize=False` switches off only the per-coefficient
  step. Because C is orthonormal, the method then equals `ARF()` exactly (up
  to round-off).
- **The meaning of `mu` differs.** DCT-LMS divides out the reference power,
  so the same `mu` value is a different effective step in each method.
- **Attribution.** S. S. Narayan, A. M. Peterson and M. J. Narasimha,
  "Transform domain LMS algorithm," *IEEE Trans. ASSP*, 31(3), 609–615, 1983.
  F. Beaufays, "Transform-domain adaptive filters: an analytical approach,"
  *IEEE Trans. Signal Processing*, 43(2), 422–431, 1995.

The DCT-LMS file reuses `ARF`, `make_ecg` and `snr_db` from `ARF.py`.

## Repository layout

```
requirements.txt        pinned package versions (pip freeze)
src/
  ARF.py                baseline LMS, synthetic ECG generator, SNR
  DCT-LMS_ARF.py        modification (DCT-LMS) + LMS vs DCT-LMS demo
tests/
  test_arf.py           pass/fail checks for LMS and DCT-LMS
  test.py               visual test of the baseline on the JSON cases
  compare.py            LMS vs DCT-LMS on the JSON cases (same mu)
  paired_stats.py       tuned, paired statistical comparison incl. stress cases
  test_cases/           test_case_1.json ... test_case_14.json
```

## Setup

Tested on Windows 11 with Python 3.13.7. The exact package versions are
pinned in [requirements.txt](requirements.txt), generated with `pip freeze`
from the tested environment. The direct dependencies are numpy 2.5.4,
matplotlib 3.11.2 and scipy 1.18.1; the rest are their dependencies.

```bash
git clone <repository URL>
cd ECG-Adaptive-Recurrent-Filter
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

Run every command from the repository root. Times were measured on the
tested machine.

| Command | What it does | Time |
|---|---|---|
| `python tests/test_arf.py` | Pass/fail checks, prints `21/21 passed` | ~4 s |
| `python src/ARF.py` | Baseline on one recording (seed 0): weights, SNR, time and spectrum plot | ~1 s |
| `python src/DCT-LMS_ARF.py` | LMS vs DCT-LMS on one recording (seed 0) | ~3 s |
| `python tests/test.py` | Baseline on all JSON cases, one figure per signal | ~7 s |
| `python tests/compare.py` | LMS vs DCT-LMS on all JSON cases, tables + figures | ~12 s |
| `python tests/paired_stats.py` | Main result: tuned paired comparison on 20 test recordings per condition | ~90 s |

The figures open in windows (`plt.show()`); nothing is written to disk.

### Checks ([tests/test_arf.py](tests/test_arf.py))

Runs without extra packages (pytest also works if installed). It separates
two questions.

**Is the code correct?**
- The guide's hand-computed example: d=[1,1], r=[1,1], M=1 gives e=[1, 0.5]
  and w=0.75.
- Limiting cases: a zero reference, and `mu=0`, both pass the input through
  unchanged.
- `e + y = d` holds exactly.
- Results agree with an independent transcription of the update.
- Output shapes are right and all values finite.
- Inputs are not modified.
- Bad arguments raise `ValueError`.

**Is the model valid?**
- With a white-noise reference the weights recover the true pathway
  `[0.35, -0.15, 0.08]`.
- On development seeds 0–19 the SNR gain is above 10 dB.
- Far above the stability bound, the filter diverges.

**DCT-LMS checks** (same structure):
- A hand-computed example where the normalisation matters: d=[1,1],
  r=[2,2], M=1, mu=0.25 gives e=[1, 0.5] and w=0.375. Plain LMS would give
  w=0.
- Ablation: with `normalize=False` it equals `ARF()` to 1e-12, for M = 1,
  8 and 16.
- The DCT matrix is orthonormal and matches `scipy.fft.dct`.
- A zero reference, and `mu=0`, pass the input through unchanged.
- Shapes, finite outputs and `e + y = d` hold; inputs are not modified;
  bad `beta` or `eps` values raise `ValueError`.
- It recovers the true pathway with a white-noise reference, and gains
  more than 10 dB SNR on seeds 0–19.

### Test cases ([tests/test_cases/](tests/test_cases/))

Each JSON file sets `seed`, `seconds`, `fs`, `scale` (reference amplitude),
`n_taps`, `mu`, `warmup_s` and `smooth_window_s`. The 14 cases cover three
signals, each with a sweep of fixed `mu` values:

| Signal | seed | scale | M | fs | `mu` values | Files |
|---|---|---|---|---|---|---|
| Baseline | 0 | 1.0 | 8 | 250 Hz | 0.0005, 0.001, 0.005, 0.02, 0.05 | 1, 4–7 |
| Strong interference | 1 | 3.0 | 16 | 250 Hz | 0.0001, 0.0005, 0.001, 0.003, 0.006 | 2, 8–11 |
| Weak interference | 2 | 0.5 | 4 | 500 Hz | 0.0005, 0.002, 0.01, 0.05 | 3, 12–14 |

Cases that differ only in `mu` are plotted together.

### Visual test ([tests/test.py](tests/test.py))

One figure per signal, one curve per `mu`:
- **Time domain:** the output over the first 4 s.
- **Frequency domain:** the spectrum after warm-up, with the 50 Hz
  attenuation in the legend.
- **Learning curve:** smoothed output power in dB.

It uses only what the filter sees (`d`, `r`, `e`), never the clean ECG.

### Comparison ([tests/compare.py](tests/compare.py))

Both methods get the same `d`, `r`, M and `mu`. The script prints:
- one table per signal, with SNR, 50 Hz attenuation, output power and
  runtime (median of 3 runs after a warm-up run);
- a summary of each method's best and worst case, ranked by **SNR gain**
  (output SNR minus input SNR).

Runtime is measured with `timeit`.

Figures:
- **Per signal:** attenuation against `mu`, and learning curves for both
  methods.
- **Best and worst cases:** a 2×2 grid showing both methods' outputs.

The clean ECG is used only to compute SNR. It is never given to the filters
and never plotted.

SNR follows the course guide: `SNR = 10·log10(Σ s² / Σ (ŝ − s)²)`, computed
after a 1 s warm-up. Higher is better. A small `eps` (1e-12) is added to
both sums so the ratio stays finite for a perfect estimate.

### Paired statistical comparison ([tests/paired_stats.py](tests/paired_stats.py))

This is the main experiment. It follows the course guide's protocol.

1. **Seeds.**
   - Validation seeds 100–109 select the settings.
   - Independent test seeds 1000–1019 are used once, for the final
     comparison.
   - Development seeds 0–19 are not used here.
2. **Tuning.**
   - Each method searches the same grid: M ∈ {2, 3, 4, 8, 16} and
     `mu` ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, so both get an
     equal budget of 35 settings.
   - The setting with the highest mean validation SNR is kept.
   - The grid was widened once, after the first run picked settings on its
     edge. The script warns if a tuned setting lands on the edge again.
3. **Conditions.**
   - Reference scales 0.5, 1.0 and 3.0, each tuned on its own validation
     recordings.
   - Two stress cases, using the settings tuned at scale 1.0:
     - **changing pathway:** the interference path switches halfway
       through the recording;
     - **leakage:** ECG leaks into the reference, `r ← r + 0.2·s`.
4. **Metric**, fixed in advance: SNR after warm-up (higher is better).
5. **Statistics** on the paired difference SNR(DCT-LMS) − SNR(LMS), with no
   normality assumption:
   - the median difference, with a 95% percentile-bootstrap CI (10 000
     resamples of whole recordings, fixed seed);
   - the win count;
   - a two-sided Wilcoxon signed-rank test, with a Bonferroni-corrected
     threshold of 0.05 / 5 = 0.01.

   Each recording counts as one independent sample.

## Results

From `python tests/paired_stats.py`, with 20 paired test recordings per
condition:

| Condition | Tuned M, `mu` (LMS / DCT-LMS) | Median LMS | Median DCT-LMS | Median difference [95% CI] | Wins | Wilcoxon p |
|---|---|---|---|---|---|---|
| scale 0.5 | 2, 0.03 / 3, 0.01 | 18.56 | 18.49 | +0.03 [−0.14, +0.15] | 11/20 | 0.93 |
| scale 1.0 | 3, 0.03 / 3, 0.01 | 17.16 | 18.19 | **+1.11** [+1.06, +1.14] | 20/20 | 1.9e-6 |
| scale 3.0 | 3, 0.003 / 3, 0.01 | 13.61 | 15.96 | **+2.27** [+2.06, +2.50] | 20/20 | 1.9e-6 |
| changing pathway | as scale 1.0 | 12.55 | 11.93 | **−0.66** [−0.71, −0.59] | 0/20 | 1.9e-6 |
| leakage | as scale 1.0 | 13.02 | 13.33 | +0.27 [+0.25, +0.30] | 20/20 | 1.9e-6 |

SNR values are in dB. 1.9e-6 is the smallest p-value the Wilcoxon test can
give with 20 recordings, reached when one method wins every recording.

- **DCT-LMS helps more the stronger the interference.** There is no
  difference at scale 0.5, +1.1 dB at scale 1.0 and +2.3 dB at scale 3.0.
- **It recovers worse when the pathway changes mid-recording.**
  Normalisation slows the dominant 50 Hz direction to the same pace as the
  others, so re-learning it takes longer.
- **The leakage result depends on tuning.** It favoured LMS before the grid
  was widened, so it is the least convincing result.
- **It is about 3 times slower than LMS** (O(M²) vs O(M) per sample), from
  `python tests/compare.py`.

**Why the normalisation helps.** The reference's autocorrelation matrix has
a large eigenvalue spread: 23 at M=3 and 55 at M=8. LMS's slowest direction
therefore converges very slowly. With DCT plus normalisation the spread
halves at M=3 (to 12.7), but at M=8 it gets worse (69). A 50 Hz tone does not
land on a DCT bin, so the DCT only roughly diagonalises the matrix. The gain
therefore holds for short filters, which is what validation selected, not
in general.

`python tests/compare.py` gives a complementary, untuned view: the same
`mu` for both methods on the 14 JSON cases. There, DCT-LMS degrades sharply
at large `mu`.

## Limitations

- **One recording length and sample rate** (12 s at 250 Hz) in the
  statistical comparison. The 20 test recordings per condition differ in
  heart rate, interference phase and noise, but not in duration.
- **The stress cases reuse settings tuned at scale 1.0.** They test
  robustness to an unexpected change, not best-case performance.
- **ECG peak distortion is not measured separately.** SNR covers it only
  indirectly.
- **The synthetic ECG is a sum of Gaussian pulses,** not a physiological
  model. The interference is a single 50 Hz tone.
- **The method assumes the reference is uncorrelated with the ECG.** If the
  reference contains ECG, the filter will remove part of the ECG too.

## Assistance declaration

Parts of the code and this README were written with the help of an AI
assistant (Claude, Anthropic).
