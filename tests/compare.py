"""Compare baseline LMS (src/ARF.py) with DCT-LMS (src/DCT-LMS_ARF.py).

Both methods run on every JSON case in tests/test_cases/ with exactly the
same input d, reference r, filter length M and fixed step size mu. Cases
that share a signal (and differ only in mu) form one group and one figure,
as in tests/test.py.

Metrics per case and method (all after warm-up):
  * SNR against the clean ECG (dB, larger is better). The clean ECG is
    used only to compute this number; it is never given to the filters
    and never plotted.
  * 50 Hz attenuation (dB, larger is better).
  * Output power (dB). e = s + residual interference, and s is the same
    for both methods, so lower means less interference left.
  * Runtime: median of 3 timeit runs after 1 warm-up run (ms), console only.

Figures:
  * One per signal group: 50 Hz attenuation against mu, learning curves.
  * One for the best and worst case of each method, ranked by SNR gain
    (output SNR minus input SNR), showing both methods' outputs.

Note: mu is used as-is for both methods. DCT-LMS divides out the
reference power, so the same number means a different effective step.
This shows each method's sensitivity to mu, not a tuned comparison.

Results are saved to results/compare.csv (one row per case and method)
and results/compare_best_worst.csv.

Run from the repository root:  python tests/compare.py
"""

import importlib.util
import sys
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from ARF import ARF, make_ecg, snr_db  # noqa: E402
from test import (  # noqa: E402  (reuse helpers from tests/test.py)
    LINE_FREQ, ZOOM_S, band_attenuation_db, group_by_signal, load_cases,
    smoothed_power_db, write_csv,
)

# The file name contains a hyphen, so it cannot be imported normally.
_spec = importlib.util.spec_from_file_location(
    "dct_lms_arf", SRC / "DCT-LMS_ARF.py")
_dct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dct)
DCT_LMS_ARF = _dct.DCT_LMS_ARF

METHODS = {"LMS": ARF, "DCT-LMS": DCT_LMS_ARF}
STYLES = {"LMS": "-", "DCT-LMS": "--"}
MARKERS = {"LMS": "o", "DCT-LMS": "s"}
COLORS = {"LMS": "C0", "DCT-LMS": "C1"}
TIMING_REPEATS = 3


def timed_run(fn, d, r, n_taps, mu):
    """Run fn once (output + warm-up), then time it with timeit.

    timeit.repeat runs one full call per repeat (number=1) and disables
    garbage collection while timing. Returns (output e, median runtime in ms).
    """
    with np.errstate(over="ignore", invalid="ignore"):
        out, _, _ = fn(d, r, n_taps=n_taps, mu=mu)
        times = timeit.repeat(lambda: fn(d, r, n_taps=n_taps, mu=mu),
                              number=1, repeat=TIMING_REPEATS)
    return out, 1000 * float(np.median(times))


def power_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def group_title(s):
    return (f"seed {s['seed']}, scale {s['scale']}, M = {s['n_taps']}, "
            f"fs = {s['fs']} Hz")


def run_group(members):
    """Run both methods on every case of one group.

    Returns (signals, records): the shared signals of the group and one
    record per case with each method's output and metrics.
    """
    s = members[0]
    fs = s["fs"]
    clean, measured, reference = make_ecg(
        seed=s["seed"], seconds=s["seconds"], fs=fs, scale=s["scale"]
    )
    warmup = int(s["warmup_s"] * fs)
    snr_in = snr_db(clean[warmup:], measured[warmup:])
    signals = {"clean": clean, "measured": measured, "fs": fs,
               "warmup": warmup, "cfg": s, "snr_in": snr_in}

    records = []
    for cfg in members:
        rec = {"cfg": cfg, "group": group_title(s), "signals": signals}
        for name, fn in METHODS.items():
            out, ms = timed_run(fn, measured, reference, cfg["n_taps"],
                                cfg["mu"])
            ok = bool(np.all(np.isfinite(out)))
            snr = snr_db(clean[warmup:], out[warmup:]) if ok else np.nan
            rec[name] = {
                "out": out,
                "ms": ms,
                "snr": snr,
                "gain": snr - snr_in,       # SNR improvement over input
                "atten": band_attenuation_db(measured[warmup:], out[warmup:],
                                             fs) if ok else np.nan,
                "power": power_db(out[warmup:]) if ok else np.nan,
            }
        records.append(rec)
    return signals, records


def print_group(signals, records):
    s, w = signals["cfg"], signals["warmup"]
    print(f"\n=== {group_title(s)} ===")
    print(f"input after warm-up: SNR {signals['snr_in']:.2f} dB, "
          f"power {power_db(signals['measured'][w:]):.1f} dB")
    print(f"{'mu':>9} | {'SNR (dB)':^15} | {'50 Hz atten (dB)':^15} | "
          f"{'out power (dB)':^15} | {'runtime (ms)':^15} | higher SNR")
    print(f"{'':>9} | {'LMS':>7} {'DCT':>7} | {'LMS':>7} {'DCT':>7} | "
          f"{'LMS':>7} {'DCT':>7} | {'LMS':>7} {'DCT':>7} |")
    for rec in records:
        a, b = rec["LMS"], rec["DCT-LMS"]
        if np.isnan(a["snr"]) and np.isnan(b["snr"]):
            better = "both diverged"
        elif np.isnan(b["snr"]) or a["snr"] > b["snr"]:
            better = "LMS"
        else:
            better = "DCT-LMS"
        print(f"{rec['cfg']['mu']:>9g} | {a['snr']:7.2f} {b['snr']:7.2f} | "
              f"{a['atten']:7.1f} {b['atten']:7.1f} | "
              f"{a['power']:7.2f} {b['power']:7.2f} | "
              f"{a['ms']:7.1f} {b['ms']:7.1f} | {better}")


def plot_group(signals, records):
    s = signals["cfg"]
    fs, measured = signals["fs"], signals["measured"]
    t = np.arange(len(measured)) / fs
    window = max(1, int(s["smooth_window_s"] * fs))
    mus = [rec["cfg"]["mu"] for rec in records]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(records)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"LMS vs DCT-LMS: {group_title(s)}\n"
                 f"({', '.join(rec['cfg']['file'] for rec in records)})")

    # 1. 50 Hz attenuation against mu
    for name in METHODS:
        ax1.semilogx(mus, [rec[name]["atten"] for rec in records],
                     STYLES[name], marker=MARKERS[name], label=name)
    ax1.set_title(f"{LINE_FREQ:.0f} Hz attenuation after warm-up "
                  f"(higher is better)")
    ax1.set_xlabel("mu")
    ax1.set_ylabel("Attenuation (dB)")

    # 2. Learning curves: colour = mu, line style = method
    ax2.plot(t, smoothed_power_db(measured, window), color="0.7", lw=0.8,
             label="Input d²")
    for rec, color in zip(records, colors):
        for name in METHODS:
            out = rec[name]["out"]
            if np.all(np.isfinite(out)):
                ax2.plot(t, smoothed_power_db(out, window), STYLES[name],
                         color=color, lw=0.9,
                         label=f"{name}, mu={rec['cfg']['mu']:g}")
    ax2.axvline(s["warmup_s"], color="k", ls=":", lw=0.8)
    ax2.set_title(f"Learning curve: output power ({s['smooth_window_s']} s "
                  f"moving average; solid = LMS, dashed = DCT-LMS)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Power (dB)")

    for ax in (ax1, ax2):
        ax.legend(loc="best", fontsize="x-small", ncol=2)
        ax.grid(True)

    fig.tight_layout()
    return fig


def best_and_worst(records):
    """{(method, 'best'|'worst'): record} by that method's SNR gain.

    Ranked by improvement over the input SNR, not absolute SNR, because
    the input SNR differs a lot between signals.
    """
    picks = {}
    for name in METHODS:
        valid = [r for r in records if np.isfinite(r[name]["gain"])]
        picks[(name, "best")] = max(valid, key=lambda r: r[name]["gain"])
        picks[(name, "worst")] = min(valid, key=lambda r: r[name]["gain"])
    return picks


def print_best_and_worst(picks):
    print("\n=== Best and worst SNR-gain case per method ===")
    print(f"{'':<16} {'case':<18} {'mu':>8} | {'input':>6} | "
          f"{'LMS SNR':>8} {'gain':>6} | {'DCT SNR':>8} {'gain':>6} | signal")
    for (name, which), rec in picks.items():
        a, b = rec["LMS"], rec["DCT-LMS"]
        print(f"{name + ' ' + which:<16} {rec['cfg']['file']:<18} "
              f"{rec['cfg']['mu']:>8g} | {rec['signals']['snr_in']:6.2f} | "
              f"{a['snr']:8.2f} {a['gain']:+6.2f} | "
              f"{b['snr']:8.2f} {b['gain']:+6.2f} | {rec['group']}")


def plot_best_and_worst(picks):
    """2 x 2 grid: rows best/worst, columns method whose SNR picked it.

    Each panel shows the input and both methods' outputs on that case.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    fig.suptitle("Best and worst SNR-gain cases (first seconds; clean ECG "
                 "not shown)")
    for col, name in enumerate(METHODS):
        for row, which in enumerate(("best", "worst")):
            ax = axes[row][col]
            rec = picks[(name, which)]
            sig = rec["signals"]
            t = np.arange(len(sig["measured"])) / sig["fs"]
            zoom = t < ZOOM_S
            ax.plot(t[zoom], sig["measured"][zoom], color="0.7", lw=0.8,
                    label="Input d")
            for m in METHODS:
                out = rec[m]["out"]
                if np.all(np.isfinite(out)):
                    ax.plot(t[zoom], out[zoom], STYLES[m], color=COLORS[m],
                            lw=0.8, label=f"{m} (SNR {rec[m]['snr']:.2f} dB,"
                                          f" gain {rec[m]['gain']:+.2f} dB)")
            ax.axvline(rec["cfg"]["warmup_s"], color="k", ls=":", lw=0.8)
            ax.set_title(f"{name} {which}: {rec['cfg']['file']}, "
                         f"mu={rec['cfg']['mu']:g}, input SNR "
                         f"{sig['snr_in']:.2f} dB\n{rec['group']}",
                         fontsize="small")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (a.u.)")
            ax.legend(loc="upper right", fontsize="x-small")
            ax.grid(True)
    fig.tight_layout()
    return fig


def result_rows(records):
    """One row per (case, method) for results/compare.csv."""
    rows = []
    for rec in records:
        cfg = rec["cfg"]
        for name in METHODS:
            m = rec[name]
            rows.append({
                "case": cfg["file"], "group": rec["group"],
                "seed": cfg["seed"], "seconds": cfg["seconds"],
                "fs": cfg["fs"], "scale": cfg["scale"],
                "n_taps": cfg["n_taps"], "mu": cfg["mu"],
                "warmup_s": cfg["warmup_s"], "method": name,
                "diverged": not np.isfinite(m["snr"]),
                "snr_in_db": rec["signals"]["snr_in"],
                "snr_db": m["snr"], "snr_gain_db": m["gain"],
                "atten_50hz_db": m["atten"], "out_power_db": m["power"],
                "runtime_ms": m["ms"],
            })
    return rows


def best_worst_rows(picks):
    """One row per (method, best/worst) for results/compare_best_worst.csv."""
    return [{
        "ranked_by": name, "which": which, "case": rec["cfg"]["file"],
        "group": rec["group"], "mu": rec["cfg"]["mu"],
        "snr_in_db": rec["signals"]["snr_in"],
        "lms_snr_db": rec["LMS"]["snr"], "lms_gain_db": rec["LMS"]["gain"],
        "dct_lms_snr_db": rec["DCT-LMS"]["snr"],
        "dct_lms_gain_db": rec["DCT-LMS"]["gain"],
    } for (name, which), rec in picks.items()]


def main():
    all_records = []
    for members in group_by_signal(load_cases()).values():
        signals, records = run_group(members)
        print_group(signals, records)
        plot_group(signals, records)
        all_records.extend(records)

    picks = best_and_worst(all_records)
    print_best_and_worst(picks)
    plot_best_and_worst(picks)
    write_csv("compare.csv", result_rows(all_records))
    write_csv("compare_best_worst.csv", best_worst_rows(picks))
    plt.show()


if __name__ == "__main__":
    main()
