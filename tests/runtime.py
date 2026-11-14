"""Per-sample runtime of LMS and DCT-LMS.

Times one call of each filter on one test recording (seed 1000, scale 1.0,
12 s at 250 Hz = 3000 samples) for the tuned filter length (read from
results/summary.csv, condition "scale 1.0") and for M = 8 and 16.

Protocol:
  * the recording is generated once, before timing: data generation is
    excluded;
  * one warm-up call, then timeit.repeat(number=1, repeat=REPEATS) with
    garbage collection off; the median and the minimum are reported;
  * the per-call setup inside the filter functions (input checks,
    allocation, and for DCT-LMS building the M x M DCT matrix) is included
    in the call time. The DCT matrix build is also timed on its own
    (dct_setup_ms) so it can be separated from the per-sample work;
  * mu does not affect the amount of work, so the same small, stable
    mu = 1e-3 is used for every M (a tuned mu can diverge at M = 16, and
    overflowing values would distort the timing).

Output: results/runtime.csv. Run from the repository root (after
tests/paired_stats.py):  python tests/runtime.py
"""

import csv
import platform
import sys
import timeit
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from ARF import ARF, make_ecg  # noqa: E402
from compare import DCT_LMS_ARF, _dct  # noqa: E402
from test import RESULTS_DIR, write_csv  # noqa: E402

SEED, SECONDS, FS, SCALE = 1000, 12, 250, 1.0
FIXED_M = (8, 16)
MU = 1e-3
REPEATS = 7
METHODS = {"LMS": ARF, "DCT-LMS": DCT_LMS_ARF}
SUMMARY_COLUMN = {"LMS": "lms_M", "DCT-LMS": "dct_lms_M"}


def tuned_m(method):
    """Tuned filter length at scale 1.0 from results/summary.csv."""
    with open(RESULTS_DIR / "summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] == "scale 1.0":
                return int(float(row[SUMMARY_COLUMN[method]]))
    raise RuntimeError("scale 1.0 not found in results/summary.csv; "
                       "run tests/paired_stats.py first")


def time_call(fn):
    """(median ms, min ms) of fn() over REPEATS runs after one warm-up."""
    fn()
    times = timeit.repeat(fn, number=1, repeat=REPEATS)
    return 1000 * float(np.median(times)), 1000 * float(np.min(times))


def main():
    _, d, r = make_ecg(seed=SEED, seconds=SECONDS, fs=FS, scale=SCALE)
    n = len(d)
    rows = []
    print(f"Runtime per call on {n} samples (seed {SEED}, scale {SCALE}), "
          f"median of {REPEATS} timeit runs after a warm-up; data "
          f"generation excluded")
    print(f"CPU: {platform.processor()} ({platform.machine()}), "
          f"Python {platform.python_version()}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}\n")
    print(f"{'method':<8} {'M':>3} {'which':<6} {'median ms':>10} "
          f"{'min ms':>8} {'us/sample':>10} {'DCT setup ms':>13}")
    for method, fn in METHODS.items():
        m_tuned = tuned_m(method)
        for M, label in [(m_tuned, "tuned")] + [(m, "fixed")
                                                for m in FIXED_M]:
            med, low = time_call(lambda: fn(d, r, n_taps=M, mu=MU))
            setup = (time_call(lambda: _dct.dct_matrix(M))[0]
                     if method == "DCT-LMS" else 0.0)
            rows.append({
                "method": method, "M": M, "M_choice": label, "mu": MU,
                "n_samples": n, "repeats": REPEATS,
                "median_ms": med, "min_ms": low,
                "us_per_sample": 1000 * med / n,
                "dct_setup_ms": setup,
                "includes_data_generation": False,
                "cpu": platform.processor(), "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__,
            })
            print(f"{method:<8} {M:>3} {label:<6} {med:10.2f} {low:8.2f} "
                  f"{1000 * med / n:10.2f} {setup:13.4f}")
    write_csv("runtime.csv", rows, RESULTS_DIR)


if __name__ == "__main__":
    main()
