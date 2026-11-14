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

Outputs (folder created automatically, default results/):
  * paired_results.csv   one row per (condition, seed, method): M, mu,
                         input/output SNR, SNR gain, runtime of the call
  * tuning.csv           every (condition, method, M, mu) tried on the
                         validation recordings, with its mean SNR
  * summary.csv          the paired statistics, one row per condition
  * learning_curves.csv  error power (e - s)^2 over time, averaged over the
                         test recordings, per condition and method
  * r_peaks.csv          output vs clean amplitude at every R peak after
                         warm-up (true R-peak times come from make_ecg's beat
                         schedule; the filters never see them)
  * run_info.json        software versions, machine, git commit, settings
Figures are made from these files by make_figures.py at the repository
root. --show additionally opens the summary figure on screen.

--quick is a smoke run (3 validation and 3 test seeds, reduced grid) that
writes the same files to results/smoke/ instead.

Run from the repository root:  python tests/paired_stats.py [--quick] [--show]
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
import timeit
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from ARF import ARF, make_ecg, snr_db  # noqa: E402
from compare import DCT_LMS_ARF  # noqa: E402  (loads src/DCT-LMS_ARF.py)
from test import RESULTS_DIR, write_csv  # noqa: E402

METHODS = {"LMS": ARF, "DCT-LMS": DCT_LMS_ARF}

VALIDATION_SEEDS = range(100, 110)
TEST_SEEDS = range(1000, 1020)

FS = 250
SECONDS = 12
WARMUP = 1 * FS                       # samples

M_GRID = (2, 3, 4, 8, 16)
MU_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

# --quick: smoke run to check the installation, not a scientific result.
QUICK_VALIDATION_SEEDS = range(100, 103)
QUICK_TEST_SEEDS = range(1000, 1003)
QUICK_M_GRID = (2, 3, 4)
QUICK_MU_GRID = (1e-3, 3e-3, 1e-2, 3e-2)
QUICK_DIR_NAME = "smoke"

PATHWAY = np.array([0.35, -0.15, 0.08])     # hardcoded in make_ecg
PATHWAY_AFTER = np.array([-0.25, 0.20, 0.05])  # changing-pathway stress
LEAKAGE = 0.2                                # r <- r + LEAKAGE * s

ALPHA = 0.05
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0

LC_SMOOTH_S = 0.25    # s, moving average applied to the learning curves
LC_STEP = 10          # samples, decimation of the saved learning curves

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


def r_peak_times(seed, clean):
    """True R-peak times (s) of make_ecg(seed, SECONDS, FS).

    make_ecg does not return its beat schedule, so it is reproduced here:
    the first draw of its generator is the beat interval, and beats start
    at 0.6 s; the R wave is the pulse with offset 0. The clean ECG is
    checked to really peak at those samples, so a change to make_ecg is
    caught instead of silently giving wrong peaks. Only used for
    evaluation; the filters never see these times.
    """
    interval = np.random.default_rng(seed).uniform(0.75, 1.05)
    beats = np.arange(0.6, SECONDS - 0.4, interval)
    idx = np.rint(beats * FS).astype(int)
    if not np.all(clean[idx] > 0.9):
        raise RuntimeError("R-peak times no longer match make_ecg")
    return beats


def run_filter(fn, d, r, n_taps, mu, timed=False):
    """Filter output e, and the runtime of one call in ms (if timed).

    The first call produces the output and doubles as the warm-up; the
    runtime is then measured with timeit (number=1, garbage collection
    off). Data generation is not included.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        e, _, _ = fn(d, r, n_taps=n_taps, mu=mu)
        ms = (1000 * timeit.timeit(lambda: fn(d, r, n_taps=n_taps, mu=mu),
                                   number=1) if timed else np.nan)
    return e, ms


def output_snr(clean, e):
    """SNR after warm-up, or -inf if the filter diverged."""
    if not np.all(np.isfinite(e)):
        return -np.inf
    return snr_db(clean[WARMUP:], e[WARMUP:])


def tune(fn, scale, val_seeds, m_grid, mu_grid):
    """Best (M, mu, mean validation SNR); also every grid point tried."""
    cases = [make_case("normal", s, scale) for s in val_seeds]
    best = (None, None, -np.inf)
    grid = []
    for M in m_grid:
        for mu in mu_grid:
            score = np.mean([output_snr(c, run_filter(fn, d, r, M, mu)[0])
                             for c, d, r in cases])
            grid.append((M, mu, score))
            if score > best[2]:
                best = (M, mu, score)
    return best, grid


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


def smoothed_db(x):
    """Moving average over LC_SMOOTH_S seconds, in dB.

    Near the ends the window is shortened (divided by the number of real
    samples it covers) so the curve does not dip at the edges.
    """
    w = max(1, int(LC_SMOOTH_S * FS))
    kernel = np.ones(w)
    avg = (np.convolve(x, kernel, mode="same")
           / np.convolve(np.ones_like(x), kernel, mode="same"))
    return 10 * np.log10(avg + 1e-12)


def git_commit():
    """(commit hash, working tree dirty?) or (None, None) outside git."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        return head.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_info(args, val_seeds, test_seeds, m_grid, mu_grid, elapsed_s):
    commit, dirty = git_commit()
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).as_posix()] + sys.argv[1:]),
        "mode": "quick" if getattr(args, "quick", False) else "full",
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "validation_seeds": list(val_seeds),
        "test_seeds": list(test_seeds),
        "m_grid": list(m_grid),
        "mu_grid": [float(mu) for mu in mu_grid],
        "fs_hz": FS,
        "recording_s": SECONDS,
        "warmup_s": WARMUP / FS,
        "learning_curve_smoothing_s": LC_SMOOTH_S,
        "learning_curve_step_samples": LC_STEP,
        "conditions": {k: {"kind": v[0], "scale": v[1], "tuned_on": v[2]}
                       for k, v in CONDITIONS.items()},
        "runtime_ms_measures": (
            "one call of the filter function on one recording, timed with "
            "timeit (number=1, gc off) after a warm-up call; excludes data "
            "generation; includes the per-call setup inside the function "
            "(input checks, allocation, and for DCT-LMS building the DCT "
            "matrix)"),
        "runtime_includes_setup": False,
        "elapsed_s": round(elapsed_s, 1),
    }


def show_figure(results, n_test):
    """Summary figure on screen (--show). Saved figures: make_figures.py."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(f"DCT-LMS vs LMS on {n_test} paired test "
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show", action="store_true",
                        help="also open the summary figure on screen")
    parser.add_argument("--quick", action="store_true",
                        help=f"smoke run: {len(QUICK_VALIDATION_SEEDS)} "
                             f"validation and {len(QUICK_TEST_SEEDS)} test "
                             f"seeds on a reduced grid, written to "
                             f"results/{QUICK_DIR_NAME}/")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    start = time.perf_counter()
    if args.quick:
        out_dir = RESULTS_DIR / QUICK_DIR_NAME
        val_seeds, test_seeds = QUICK_VALIDATION_SEEDS, QUICK_TEST_SEEDS
        m_grid, mu_grid = QUICK_M_GRID, QUICK_MU_GRID
        print("QUICK SMOKE RUN: checks the installation; with only "
              f"{len(test_seeds)} test recordings the statistics are not "
              "meaningful.\n")
    else:
        out_dir = RESULTS_DIR
        val_seeds, test_seeds = VALIDATION_SEEDS, TEST_SEEDS
        m_grid, mu_grid = M_GRID, MU_GRID

    # ---- 1. tuning on validation seeds ----
    print(f"Tuning on validation seeds {list(val_seeds)}, "
          f"grid M in {list(m_grid)}, mu in "
          f"{[float(f'{mu:.3g}') for mu in mu_grid]}")
    tuned = {}
    tuning_rows = []
    for name, (kind, scale, _) in CONDITIONS.items():
        if kind != "normal":
            continue
        for method, fn in METHODS.items():
            best, grid = tune(fn, scale, val_seeds, m_grid, mu_grid)
            tuned[(name, method)] = best
            for M, mu, score in grid:
                tuning_rows.append({
                    "condition": name, "scale": scale, "method": method,
                    "M": M, "mu": mu, "mean_val_snr": score,
                    "selected": (M, mu) == best[:2],
                    "n_val_recordings": len(val_seeds),
                })
    print(f"\n{'tuned on':<12} {'method':<8} {'M':>3} {'mu':>10} "
          f"{'val SNR (dB)':>13}")
    for (name, method), (M, mu, score) in tuned.items():
        edge = []
        if M in (m_grid[0], m_grid[-1]):
            edge.append("M")
        if mu in (mu_grid[0], mu_grid[-1]):
            edge.append("mu")
        note = f"  WARNING: {' and '.join(edge)} on grid edge" if edge else ""
        print(f"{name:<12} {method:<8} {M:>3} {mu:>10.4g} {score:13.2f}{note}")

    # ---- 2. final comparison on test seeds ----
    results = {}
    result_rows, curve_rows, peak_rows = [], [], []
    for name, (kind, scale, tuned_on) in CONDITIONS.items():
        cases = [make_case(kind, s, scale) for s in test_seeds]
        snr_in = np.array([snr_db(c[WARMUP:], d[WARMUP:])
                           for c, d, _ in cases])
        n = len(cases[0][0])
        sq_err = {"input": np.mean([(d - c) ** 2 for c, d, _ in cases],
                                   axis=0)}
        snrs = {}
        for method, fn in METHODS.items():
            M, mu, _ = tuned[(tuned_on, method)]
            out, acc, n_ok = [], np.zeros(n), 0
            for seed, (c, d, r), s_in in zip(test_seeds, cases, snr_in):
                e, ms = run_filter(fn, d, r, M, mu, timed=True)
                s_out = output_snr(c, e)
                out.append(s_out)
                result_rows.append({
                    "condition": name, "seed": seed, "method": method,
                    "M": M, "mu": mu, "snr_in_db": s_in,
                    "snr_out_db": s_out, "snr_gain_db": s_out - s_in,
                    "runtime_ms": ms,
                })
                if np.isfinite(s_out):
                    acc += (e - c) ** 2
                    n_ok += 1
                    for t_r in r_peak_times(seed, c):
                        i = int(round(t_r * FS))
                        if i < WARMUP:
                            continue
                        peak_rows.append({
                            "condition": name, "seed": seed,
                            "method": method, "r_peak_time_s": t_r,
                            "clean_amp": c[i], "output_amp": e[i],
                            "amp_error": e[i] - c[i],
                        })
            snrs[method] = np.array(out)
            if n_ok:
                sq_err[method] = acc / n_ok
        for method, curve in sq_err.items():
            db = smoothed_db(curve)
            for i in range(0, n, LC_STEP):
                curve_rows.append({"condition": name, "method": method,
                                   "t_s": i / FS, "err_power_db": db[i]})
        results[name] = {"snr_in": snr_in, "snrs": snrs}

    # ---- 3. paired statistics ----
    n_tests = len(CONDITIONS)
    alpha_bonf = ALPHA / n_tests
    print(f"\nPaired comparison on test seeds {list(test_seeds)} "
          f"(n = {len(test_seeds)} recordings per condition)")
    print("difference = SNR(DCT-LMS) - SNR(LMS); positive favours DCT-LMS")
    print(f"\n{'condition':<17} {'input':>6} {'LMS':>6} {'DCT':>6} | "
          f"{'median diff':>11} {'95% bootstrap CI':^17} {'wins':>6} "
          f"{'Wilcoxon p':>10} | result")
    summary_rows = []
    for name, res in results.items():
        a, b = res["snrs"]["DCT-LMS"], res["snrs"]["LMS"]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            print(f"{name:<17} a method diverged on some test recordings; "
                  f"statistics skipped")
            res["test"] = None
            summary_rows.append({"condition": name,
                                 "result": "a method diverged; skipped"})
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
        tuned_on = CONDITIONS[name][2]
        summary_rows.append({
            "condition": name, "tuned_on": tuned_on,
            "lms_M": tuned[(tuned_on, "LMS")][0],
            "lms_mu": tuned[(tuned_on, "LMS")][1],
            "dct_lms_M": tuned[(tuned_on, "DCT-LMS")][0],
            "dct_lms_mu": tuned[(tuned_on, "DCT-LMS")][1],
            "n_recordings": st["n"],
            "median_snr_in_db": float(np.median(res["snr_in"])),
            "median_lms_snr_db": float(np.median(b)),
            "median_dct_lms_snr_db": float(np.median(a)),
            "median_diff_db": st["median"],
            "ci95_low_db": st["ci"][0], "ci95_high_db": st["ci"][1],
            "dct_lms_wins": st["wins"], "wilcoxon_p": st["p"],
            "alpha_bonferroni": alpha_bonf, "result": verdict,
        })
    print("\n(input / LMS / DCT columns are median SNR in dB; wins = "
          "recordings where DCT-LMS has the higher SNR)")

    # ---- 4. save ----
    write_csv("paired_results.csv", result_rows, out_dir)
    write_csv("tuning.csv", tuning_rows, out_dir)
    write_csv("summary.csv", summary_rows, out_dir)
    write_csv("learning_curves.csv", curve_rows, out_dir)
    write_csv("r_peaks.csv", peak_rows, out_dir)
    elapsed = time.perf_counter() - start
    info = run_info(args, val_seeds, test_seeds, m_grid, mu_grid, elapsed)
    info_path = out_dir / "run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    print(f"Saved {info_path}")
    print(f"Total time: {elapsed:.1f} s")

    if args.show:
        show_figure(results, len(test_seeds))


if __name__ == "__main__":
    main()
