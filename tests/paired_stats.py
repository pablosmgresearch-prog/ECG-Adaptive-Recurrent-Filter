"""Paired comparison: DCT-LMS vs baseline LMS on independent test recordings.

Protocol (course guide, Track B, p.3):

1. Seeds. Validation 100-109 select the settings; independent test seeds
   1000-1019 are used once for the final comparison. Development seeds 0-19
   (used while writing the code) are not used here.
2. Tuning. For each method separately, the same grid of filter lengths M
   and step sizes mu is searched (equal budget: 5 x 7 = 35 settings each).
   The setting with the highest mean validation SNR is kept. Diverged runs
   score -inf. The grid was widened once after the first run picked
   settings on its edge; the check prints a warning if that happens again.
3. Conditions:
     * reference scale 0.5, 1.0 and 3.0 (each tuned on its own validation
       recordings);
     * stress: pathway changes halfway through the recording;
     * stress: ECG leaks into the reference, r <- r + 0.2 s.
   The stress conditions reuse the settings tuned at scale 1.0: they model
   something going wrong after the filter was configured.
4. Metric (chosen in advance): SNR after a 1 s warm-up, in dB, larger is
   better. The clean ECG is used only to compute it; the filters never see
   it.
5. Statistics. For each condition, the paired difference
   SNR(DCT-LMS) - SNR(LMS) on the same 20 test recordings. No normality
   assumption is made:
     * effect size: median difference, with a 95% percentile-bootstrap
       confidence interval (10 000 resamples of whole recordings, fixed
       seed so the interval is reproducible);
     * win count: on how many recordings DCT-LMS has the higher SNR;
     * significance: two-sided Wilcoxon signed-rank test
       (scipy.stats.wilcoxon). Five conditions are tested, so the
       Bonferroni-corrected threshold 0.05 / 5 = 0.01 is used.
   Each recording is one independent sample; samples within a recording
   are not treated as independent.

Run from the repository root:  python tests/paired_stats.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from ARF import ARF, make_ecg, snr_db  # noqa: E402
from compare import DCT_LMS_ARF  # noqa: E402  (loads src/DCT-LMS_ARF.py)

METHODS = {"LMS": ARF, "DCT-LMS": DCT_LMS_ARF}

VALIDATION_SEEDS = range(100, 110)
TEST_SEEDS = range(1000, 1020)

FS = 250
SECONDS = 12
WARMUP = 1 * FS                       # samples

M_GRID = (2, 3, 4, 8, 16)
MU_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

PATHWAY = np.array([0.35, -0.15, 0.08])     # hardcoded in make_ecg
PATHWAY_AFTER = np.array([-0.25, 0.20, 0.05])  # changing-pathway stress
LEAKAGE = 0.2                                # r <- r + LEAKAGE * s

ALPHA = 0.05
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0

# condition name -> (generator kind, reference scale, tuned on)
CONDITIONS = {
    "scale 0.5": ("normal", 0.5, "scale 0.5"),
    "scale 1.0": ("normal", 1.0, "scale 1.0"),
    "scale 3.0": ("normal", 3.0, "scale 3.0"),
    "changing pathway": ("changing", 1.0, "scale 1.0"),
    "leakage": ("leakage", 1.0, "scale 1.0"),
}


def make_case(kind, seed, scale):
    """(clean, measured, reference) for one recording of a condition.

    All variants start from make_ecg, so the clean ECG and the reference
    are identical to the normal case with the same seed and scale.
    """
    clean, measured, reference = make_ecg(seed=seed, seconds=SECONDS, fs=FS,
                                          scale=scale)
    if kind == "normal":
        return clean, measured, reference

    if kind == "changing":
        # make_ecg's interference is exactly convolve(r, PATHWAY); swap it
        # for PATHWAY_AFTER in the second half of the recording.
        n = len(measured)
        before = np.convolve(reference, PATHWAY, mode="full")[:n]
        after = np.convolve(reference, PATHWAY_AFTER, mode="full")[:n]
        switched = np.where(np.arange(n) < n // 2, before, after)
        return clean, measured - before + switched, reference

    if kind == "leakage":
        # The interference in d still comes from the original reference;
        # only the filter's reference input is contaminated with ECG.
        return clean, measured, reference + LEAKAGE * clean

    raise ValueError(f"unknown condition kind {kind!r}")


def run_snr(fn, clean, d, r, n_taps, mu):
    """Output SNR after warm-up, or -inf if the filter diverged."""
    with np.errstate(over="ignore", invalid="ignore"):
        e, _, _ = fn(d, r, n_taps=n_taps, mu=mu)
    if not np.all(np.isfinite(e)):
        return -np.inf
    return snr_db(clean[WARMUP:], e[WARMUP:])


def tune(fn, scale):
    """Best (M, mu, mean validation SNR) on the validation seeds."""
    cases = [make_case("normal", s, scale) for s in VALIDATION_SEEDS]
    best = (None, None, -np.inf)
    for M in M_GRID:
        for mu in MU_GRID:
            score = np.mean([run_snr(fn, c, d, r, M, mu) for c, d, r in cases])
            if score > best[2]:
                best = (M, mu, score)
    return best


def bootstrap_ci(values, statistic=np.median, n_boot=N_BOOTSTRAP,
                 seed=BOOTSTRAP_SEED):
    """95% percentile-bootstrap CI of `statistic` over the recordings.

    Whole recordings are resampled with replacement; the seed is fixed so
    the interval is reproducible.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = statistic(values[idx], axis=1)
    return tuple(np.percentile(boot, [100 * ALPHA / 2,
                                      100 * (1 - ALPHA / 2)]))


def paired_test(a, b):
    """Stats for the paired difference a - b (arrays of equal length)."""
    diff = np.asarray(a) - np.asarray(b)
    _, p = stats.wilcoxon(a, b, alternative="two-sided")
    return {"diff": diff, "median": float(np.median(diff)),
            "ci": bootstrap_ci(diff), "wins": int(np.sum(diff > 0)),
            "p": float(p), "n": len(diff)}


def main():
    start = time.perf_counter()

    # ---- 1. tuning on validation seeds ----
    print("Tuning on validation seeds "
          f"{VALIDATION_SEEDS.start}-{VALIDATION_SEEDS.stop - 1}, "
          f"grid M in {M_GRID}, mu in {MU_GRID}")
    tuned = {}
    for name, (kind, scale, _) in CONDITIONS.items():
        if kind != "normal":
            continue
        for method, fn in METHODS.items():
            tuned[(name, method)] = tune(fn, scale)
    print(f"\n{'tuned on':<12} {'method':<8} {'M':>3} {'mu':>8} "
          f"{'val SNR (dB)':>13}")
    for (name, method), (M, mu, score) in tuned.items():
        edge = []
        if M in (M_GRID[0], M_GRID[-1]):
            edge.append("M")
        if mu in (MU_GRID[0], MU_GRID[-1]):
            edge.append("mu")
        note = f"  WARNING: {' and '.join(edge)} on grid edge" if edge else ""
        print(f"{name:<12} {method:<8} {M:>3} {mu:>8g} {score:13.2f}{note}")

    # ---- 2. final comparison on test seeds ----
    results = {}
    for name, (kind, scale, tuned_on) in CONDITIONS.items():
        cases = [make_case(kind, s, scale) for s in TEST_SEEDS]
        snr_in = np.array([snr_db(c[WARMUP:], d[WARMUP:])
                           for c, d, _ in cases])
        snrs = {}
        for method, fn in METHODS.items():
            M, mu, _ = tuned[(tuned_on, method)]
            snrs[method] = np.array([run_snr(fn, c, d, r, M, mu)
                                     for c, d, r in cases])
        results[name] = {"snr_in": snr_in, "snrs": snrs}

    # ---- 3. paired statistics ----
    n_tests = len(CONDITIONS)
    alpha_bonf = ALPHA / n_tests
    print(f"\nPaired comparison on test seeds {TEST_SEEDS.start}-"
          f"{TEST_SEEDS.stop - 1} (n = {len(TEST_SEEDS)} recordings per "
          f"condition)")
    print("difference = SNR(DCT-LMS) - SNR(LMS); positive favours DCT-LMS")
    print(f"\n{'condition':<17} {'input':>6} {'LMS':>6} {'DCT':>6} | "
          f"{'median diff':>11} {'95% bootstrap CI':^17} {'wins':>6} "
          f"{'Wilcoxon p':>10} | result")
    for name, res in results.items():
        a, b = res["snrs"]["DCT-LMS"], res["snrs"]["LMS"]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            print(f"{name:<17} a method diverged on some test recordings; "
                  f"statistics skipped")
            res["test"] = None
            continue
        st = paired_test(a, b)
        res["test"] = st
        if st["p"] < alpha_bonf:
            verdict = ("DCT-LMS better" if st["median"] > 0 else "LMS better")
            verdict += f" (p < {alpha_bonf:g})"
        elif st["p"] < ALPHA:
            verdict = ("DCT-LMS better" if st["median"] > 0 else "LMS better")
            verdict += f" (p < {ALPHA:g}, not after correction)"
        else:
            verdict = "no significant difference"
        print(f"{name:<17} {np.median(res['snr_in']):6.2f} "
              f"{np.median(b):6.2f} {np.median(a):6.2f} | "
              f"{st['median']:+11.2f} "
              f"[{st['ci'][0]:+6.2f}, {st['ci'][1]:+6.2f}] "
              f"{st['wins']:>3}/{st['n']:<2} {st['p']:10.2e} | {verdict}")
    print("\n(input / LMS / DCT columns are median SNR in dB; wins = "
          "recordings where DCT-LMS has the higher SNR)")
    print(f"Total time: {time.perf_counter() - start:.1f} s")

    # ---- 4. figure ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(f"DCT-LMS vs LMS on {len(TEST_SEEDS)} paired test "
                 f"recordings per condition")
    names = list(results)
    x = np.arange(len(names))
    rng = np.random.default_rng(0)                 # jitter only

    for i, name in enumerate(names):
        st = results[name]["test"]
        if st is None:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=st["n"])
        ax1.scatter(i + jitter, st["diff"], s=14, color="0.5", alpha=0.7,
                    label="One recording" if i == 0 else None)
        ax1.errorbar(i, st["median"],
                     yerr=[[st["median"] - st["ci"][0]],
                           [st["ci"][1] - st["median"]]],
                     fmt="o", color="C3", capsize=6, lw=2,
                     label="Median, 95% bootstrap CI" if i == 0 else None)
        ax1.annotate(f"{st['wins']}/{st['n']} wins\np = {st['p']:.1e}",
                     (i, max(st["ci"][1], st["diff"].max())),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize="small")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x, names)
    ax1.set_ylabel("SNR(DCT-LMS) − SNR(LMS) (dB)")
    ax1.set_title("Paired SNR difference (above 0 favours DCT-LMS; "
                  "p: Wilcoxon signed-rank)")
    ax1.legend(loc="best", fontsize="small")
    ax1.grid(True, axis="y")

    width = 0.25
    for j, (label, key) in enumerate((("Input", None), ("LMS", "LMS"),
                                      ("DCT-LMS", "DCT-LMS"))):
        medians, lower, upper = [], [], []
        for name in names:
            vals = (results[name]["snr_in"] if key is None
                    else results[name]["snrs"][key])
            vals = vals[np.isfinite(vals)]
            med = float(np.median(vals))
            lo, hi = bootstrap_ci(vals)
            medians.append(med)
            lower.append(med - lo)
            upper.append(hi - med)
        ax2.bar(x + (j - 1) * width, medians, width, yerr=[lower, upper],
                capsize=3, label=label, color=("0.7", "C0", "C1")[j])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x, names)
    ax2.set_ylabel("SNR after warm-up (dB)")
    ax2.set_title("Median SNR per method (error bars: 95% bootstrap CI)")
    ax2.legend(loc="best", fontsize="small")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
