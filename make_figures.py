"""Make the report figures from the saved results of tests/paired_stats.py.

Reads only the CSV/JSON files in the results folder (default results/) and
writes PNGs to <results>/figures/:

  a_representative_example.png  one test recording, first 4 s: input, clean
                                ECG, LMS and DCT-LMS outputs
  b_paired_difference.png       paired SNR difference per condition, median
                                and 95% bootstrap CI (from summary.csv)
  c_stress_changing_pathway.png learning curves of both methods when the
                                interference pathway changes mid-recording
  d_peak_distortion_leakage.png zoom on R peaks in the leakage condition,
                                clean vs LMS vs DCT-LMS

It also prints the mean R-peak amplitude error per condition and method
(from r_peaks.csv).

Figures b and c, and every printed number, come only from the saved files.
Figures a and d need a full time series, which is not stored: they re-run
both filters on ONE test recording, using the seed and the tuned (M, mu)
read from paired_results.csv, so they show exactly the settings the
statistics were computed with.

Run from the repository root (after tests/paired_stats.py):
    python make_figures.py [--results results] [--show]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

# Only used to re-run one recording for figures a and d (see docstring).
from paired_stats import CONDITIONS, METHODS, make_case  # noqa: E402

STYLE = {"input": dict(color="0.65", lw=0.8),
         "clean": dict(color="k", lw=1.2, ls="--"),
         "LMS": dict(color="C0", lw=1.0),
         "DCT-LMS": dict(color="C1", lw=1.0)}


def read_csv(path):
    """List of dicts; numeric-looking values converted to float."""
    def convert(v):
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: convert(v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def paired_differences(results, condition):
    """{seed: SNR(DCT-LMS) - SNR(LMS)} for one condition."""
    snr = {}
    for row in results:
        if row["condition"] == condition:
            snr[(int(row["seed"]), row["method"])] = row["snr_out_db"]
    seeds = sorted({s for s, _ in snr})
    return {s: snr[(s, "DCT-LMS")] - snr[(s, "LMS")] for s in seeds}


def representative_seed(results, condition):
    """Test seed whose paired difference is closest to the median."""
    diffs = paired_differences(results, condition)
    med = np.median(list(diffs.values()))
    return min(diffs, key=lambda s: abs(diffs[s] - med))


def rerun(results, condition, seed):
    """Re-run both filters on one recording with the settings in the CSV.

    Returns (t, clean, measured, {method: output}, {method: (M, mu)}).
    """
    kind, scale, _ = CONDITIONS[condition]
    clean, d, r = make_case(kind, seed, scale)
    outputs, settings = {}, {}
    for row in results:
        if row["condition"] == condition and int(row["seed"]) == seed:
            M, mu = int(row["M"]), row["mu"]
            outputs[row["method"]] = METHODS[row["method"]](
                d, r, n_taps=M, mu=mu)[0]
            settings[row["method"]] = (M, mu)
    t = np.arange(len(d)) / info["fs_hz"]
    return t, clean, d, outputs, settings


def settings_label(settings):
    return ", ".join(f"{m}: M={M}, mu={mu:.3g}"
                     for m, (M, mu) in settings.items())


def figure_a(results, out):
    condition = "scale 1.0"
    seed = representative_seed(results, condition)
    t, clean, d, outputs, settings = rerun(results, condition, seed)
    zoom = t < 4.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.suptitle(f"Representative test recording ({condition}, seed {seed}; "
                 f"synthetic ECG)\n{settings_label(settings)}",
                 fontsize="medium")
    ax1.plot(t[zoom], d[zoom], label="Input d (ECG + interference)",
             **STYLE["input"])
    ax1.plot(t[zoom], clean[zoom], label="Clean ECG s", **STYLE["clean"])
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.legend(loc="upper right", fontsize="small")
    ax2.plot(t[zoom], clean[zoom], label="Clean ECG s", **STYLE["clean"])
    for m in METHODS:
        ax2.plot(t[zoom], outputs[m][zoom], label=f"{m} output e",
                 **STYLE[m])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (a.u.)")
    ax2.legend(loc="upper right", fontsize="small")
    for ax in (ax1, ax2):
        ax.axvline(info["warmup_s"], color="k", ls=":", lw=0.8)
        ax.grid(True)
    ax2.text(info["warmup_s"], ax2.get_ylim()[0], " end of warm-up",
             va="bottom", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def figure_b(results, summary, out):
    names = [row["condition"] for row in summary]
    x = np.arange(len(names))
    rng = np.random.default_rng(0)                 # jitter only
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(summary):
        if "median_diff_db" not in row or row["median_diff_db"] == "":
            continue
        diffs = np.array(list(paired_differences(results,
                                                 row["condition"]).values()))
        ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(diffs)), diffs,
                   s=14, color="0.5", alpha=0.7,
                   label="One test recording" if i == 0 else None)
        med, lo, hi = (row["median_diff_db"], row["ci95_low_db"],
                       row["ci95_high_db"])
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o",
                    color="C3", capsize=6, lw=2,
                    label="Median, 95% bootstrap CI" if i == 0 else None)
        ax.annotate(f"{int(row['dct_lms_wins'])}/{int(row['n_recordings'])}"
                    f" wins\np = {row['wilcoxon_p']:.1e}",
                    (i, max(hi, diffs.max())), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize="small")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("SNR(DCT-LMS) − SNR(LMS) (dB)")
    n = int(summary[0]["n_recordings"])
    ax.set_title(f"Paired SNR difference on {n} test recordings per "
                 f"condition (above 0 favours DCT-LMS; p: Wilcoxon "
                 f"signed-rank)", fontsize="medium")
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True, axis="y")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def figure_c(curves, summary, out):
    condition = "changing pathway"
    row = next(r for r in summary if r["condition"] == condition)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for method in ("input", "LMS", "DCT-LMS"):
        pts = [(c["t_s"], c["err_power_db"]) for c in curves
               if c["condition"] == condition and c["method"] == method]
        t, db = np.array(pts).T
        label = ("Input d (no filtering)" if method == "input"
                 else f"{method}")
        ax.plot(t, db, label=label, **STYLE[method])
    switch = info["recording_s"] / 2
    ax.axvline(info["warmup_s"], color="k", ls=":", lw=0.8)
    ax.axvline(switch, color="C3", ls="--", lw=1)
    ax.text(switch, ax.get_ylim()[1], " pathway changes", color="C3",
            va="top", fontsize="small")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error power (e − s)² (dB)")
    ax.set_title(f"Stress case: changing pathway, error power averaged over "
                 f"{int(row['n_recordings'])} test recordings "
                 f"({LC_LABEL} moving average)", fontsize="medium")
    ax.legend(loc="lower left", fontsize="small")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def peak_error_table(peaks):
    """Print and return {(condition, method): (mean error, mean |error|)}."""
    table = {}
    for p in peaks:
        table.setdefault((p["condition"], p["method"]), []).append(
            p["amp_error"])
    print("\nR-peak amplitude error, output minus clean at the true R peak "
          "(after warm-up; clean R amplitude is about 1):")
    print(f"{'condition':<17} {'method':<8} {'peaks':>6} {'mean':>8} "
          f"{'mean |err|':>10}")
    stats = {}
    for (cond, method), errs in table.items():
        errs = np.array(errs)
        stats[(cond, method)] = (errs.mean(), np.abs(errs).mean())
        print(f"{cond:<17} {method:<8} {len(errs):>6} {errs.mean():+8.4f} "
              f"{np.abs(errs).mean():10.4f}")
    return stats


def figure_d(results, peaks, peak_stats, out):
    condition = "leakage"
    seed = representative_seed(results, condition)
    t, clean, d, outputs, settings = rerun(results, condition, seed)
    times = sorted({p["r_peak_time_s"] for p in peaks
                    if p["condition"] == condition and int(p["seed"]) == seed
                    and p["r_peak_time_s"] >= 2.0})[:3]
    window = (t >= times[0] - 0.35) & (t <= times[-1] + 0.45)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(t[window], clean[window], label="Clean ECG s", **STYLE["clean"])
    for m in METHODS:
        mean_err, mean_abs = peak_stats[(condition, m)]
        ax.plot(t[window], outputs[m][window],
                label=f"{m} (mean R-peak error {mean_err:+.4f}, "
                      f"|err| {mean_abs:.4f})", **STYLE[m])
    for i, tr in enumerate(times):
        ax.axvline(tr, color="0.5", ls=":", lw=0.8,
                   label="True R-peak time" if i == 0 else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"ECG peak distortion with reference leakage r ← r + 0.2·s "
                 f"(seed {seed}; errors averaged over all test recordings)\n"
                 f"{settings_label(settings)}", fontsize="medium")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize="small")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def main(argv=None):
    global info, LC_LABEL
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default=str(ROOT / "results"),
                        help="folder with the paired_stats.py outputs")
    parser.add_argument("--show", action="store_true",
                        help="also open the figures on screen")
    args = parser.parse_args(argv)

    res_dir = Path(args.results)
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "run_info.json", encoding="utf-8") as f:
        info = json.load(f)
    LC_LABEL = f"{info['learning_curve_smoothing_s']:g} s"
    results = read_csv(res_dir / "paired_results.csv")
    summary = read_csv(res_dir / "summary.csv")
    curves = read_csv(res_dir / "learning_curves.csv")
    peaks = read_csv(res_dir / "r_peaks.csv")

    peak_stats = peak_error_table(peaks)
    paths = {
        "a": fig_dir / "a_representative_example.png",
        "b": fig_dir / "b_paired_difference.png",
        "c": fig_dir / "c_stress_changing_pathway.png",
        "d": fig_dir / "d_peak_distortion_leakage.png",
    }
    figure_a(results, paths["a"])
    figure_b(results, summary, paths["b"])
    figure_c(curves, summary, paths["c"])
    figure_d(results, peaks, peak_stats, paths["d"])
    print()
    for path in paths.values():
        print(f"Saved {path}")

    if args.show:
        plt.show()


info = {}
LC_LABEL = ""

if __name__ == "__main__":
    main()
