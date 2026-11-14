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
src/
  ARF.py                baseline LMS, synthetic ECG generator, SNR
  DCT-LMS_ARF.py        modification (DCT-LMS) + LMS vs DCT-LMS demo
tests/
  test_arf.py           pass/fail checks for the baseline
  test.py               visual test of the baseline on the JSON cases
  compare.py            LMS vs DCT-LMS on the JSON cases
  test_cases/           test_case_1.json ... test_case_14.json
```

## Setup

Tested on Windows 11 with Python 3.13.7, numpy 2.5.4, matplotlib 3.11.2 and
scipy 1.18.1.

```bash
git clone <repository URL>
cd ECG-Adaptive-Recurrent-Filter
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
python -m pip install numpy==2.5.4 matplotlib==3.11.2 scipy==1.18.1
```

## Running

Run every command from the repository root. Times were measured on the
tested machine.

| Command | What it does | Time |
|---|---|---|
| `python tests/test_arf.py` | Pass/fail checks, prints `13/13 passed` | ~3 s |
| `python src/ARF.py` | Baseline on one recording (seed 0): weights, SNR, time and spectrum plot | ~1 s |
| `python src/DCT-LMS_ARF.py` | LMS vs DCT-LMS on one recording (seed 0) | ~3 s |
| `python tests/test.py` | Baseline on all JSON cases, one figure per signal | ~7 s |
| `python tests/compare.py` | LMS vs DCT-LMS on all JSON cases, tables + figures | ~12 s |

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

Figures:
- **Per signal:** attenuation against `mu`, and learning curves for both
  methods.
- **Best and worst cases:** a 2×2 grid showing both methods' outputs.

The clean ECG is used only to compute SNR. It is never given to the filters
and never plotted.

SNR follows the course guide: `SNR = 10·log10(Σ s² / Σ (ŝ − s)²)`, computed
after a 1 s warm-up. Higher is better.

## Results so far

From `python tests/compare.py`, on the 14 JSON cases (one recording each):

| | Case | `mu` | LMS gain | DCT-LMS gain |
|---|---|---|---|---|
| LMS best | test_case_10 (strong) | 0.003 | +16.4 dB | +19.0 dB |
| LMS worst | test_case_12 (weak) | 0.0005 | +5.7 dB | +11.8 dB |
| DCT-LMS best | test_case_11 (strong) | 0.006 | +9.6 dB | +19.5 dB |
| DCT-LMS worst | test_case_7 (baseline) | 0.05 | +9.2 dB | +1.5 dB |

- **DCT-LMS gains more SNR in most cases.** It also recovers the pathway
  more closely, even though it removes less of the 50 Hz component itself.
- **It degrades sharply at large `mu`,** as predicted.
- **It is about 2–3 times slower** than LMS.

## Limitations

- **This is not yet a tuned comparison.** Both methods use the same `mu`
  values, and `mu` means something different in each. A fair comparison
  needs:
  - `M` and `mu` selected per method on validation seeds (e.g. 100–109);
  - evaluation on independent test seeds (e.g. 1000–1019);
  - 95% confidence intervals of the paired differences.
- **Each JSON case is a single recording,** so no variability is reported
  yet.
- **The stress cases from the guide are not implemented yet:** a changing
  pathway, and reference leakage `r ← r + 0.2·s`.
- **The synthetic ECG is a sum of Gaussian pulses,** not a physiological
  model. The interference is a single 50 Hz tone.
- **The method assumes the reference is uncorrelated with the ECG.** If the
  reference contains ECG, the filter will remove part of the ECG too.

## Assistance declaration

Parts of the code and this README were written with the help of an AI
assistant (Claude, Anthropic).
