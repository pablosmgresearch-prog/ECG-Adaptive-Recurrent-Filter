"""Visual test cases for the LMS noise canceller in src/ARF.py.

Each test case is a JSON file in tests/test_cases/. Cases that share the
same signal settings (seed, duration, fs, scale, n_taps, ...) and differ
only in their fixed step size mu form a group. Each group gets one figure
in which every curve is one mu, so the effect of mu is visible directly.

The plots use only signals the filter itself has access to: the measured
input d, the reference r, the output e = d - y and the weights w. The clean
ECG is never used here; quantitative checks against ground truth live in
tests/test_arf.py.

Panels (one curve per mu):
  1. Time domain: output e over the first seconds (input d in grey).
  2. Frequency domain: power spectrum of e after warm-up (input d in grey),
     with the 50 Hz attenuation of each mu in the legend.
  3. Learning curve: smoothed output power e^2 over time, in dB.

Results (one row per case) are saved to results/test_cases.csv.

Run from the repository root:  python tests/test.py
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ARF import ARF, make_ecg  # noqa: E402

CASES_DIR = HERE / "test_cases"
RESULTS_DIR = HERE.parent / "results"

LINE_FREQ = 50.0      # Hz, interference frequency in make_ecg
BAND = 1.0            # Hz, half-width of the band used for attenuation
ZOOM_S = 4.0          # s, time span shown in the time-domain panel

# Settings that define the signal and filter length; cases equal in all of
# these (and differing only in mu) are plotted together.
GROUP_KEYS = ("seed", "seconds", "fs", "scale", "n_taps",
              "warmup_s", "smooth_window_s")


def write_csv(name, rows, out_dir=None):
    """Write a list of dicts to <out_dir>/<name> (folder created if needed).

    out_dir defaults to results/. Returns the path. Columns are every key
    that appears in any row, in first-seen order; missing values are left
    empty.
    """
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    return path


def load_cases():
    """All JSON cases, in numeric order of their file names."""
    def number(path):
        m = re.search(r"(\d+)", path.stem)
        return int(m.group(1)) if m else 0
    paths = sorted(CASES_DIR.glob("*.json"), key=number)
    cases = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["file"] = path.name
        cases.append(cfg)
    return cases


def group_by_signal(cases):
    """Dict {settings tuple: [cases sorted by mu]}, in first-seen order."""
    groups = {}
    for cfg in cases:
        groups.setdefault(tuple(cfg[k] for k in GROUP_KEYS), []).append(cfg)
    for members in groups.values():
        members.sort(key=lambda c: c["mu"])
    return groups


def power_spectrum(x, fs):
    """One-sided power spectrum. Returns (freqs [Hz], power [linear])."""
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    power = (np.abs(np.fft.rfft(x)) / len(x)) ** 2
    return freqs, power


def band_attenuation_db(d, e, fs, f0=LINE_FREQ, half_width=BAND, eps=1e-12):
    """Power reduction (dB) of e relative to d in the band f0 +/- half_width."""
    freqs, p_d = power_spectrum(d, fs)
    _, p_e = power_spectrum(e, fs)
    band = np.abs(freqs - f0) <= half_width
    # eps keeps the ratio finite when either band power is exactly zero.
    return 10 * np.log10((p_d[band].sum() + eps) / (p_e[band].sum() + eps))


def smoothed_power_db(x, window):
    """Moving-average of x^2 over `window` samples, in dB."""
    p = np.convolve(x ** 2, np.ones(window) / window, mode="same")
    return 10 * np.log10(p + 1e-12)


def plot_group(members, rows=None):
    """One figure for cases sharing a signal, one curve per fixed mu.

    If `rows` is a list, one result dict per case is appended to it.
    """
    s = members[0]
    fs = s["fs"]
    _, measured, reference = make_ecg(          # clean ECG deliberately unused
        seed=s["seed"], seconds=s["seconds"], fs=fs, scale=s["scale"]
    )
    t = np.arange(len(measured)) / fs
    warmup = int(s["warmup_s"] * fs)
    window = max(1, int(s["smooth_window_s"] * fs))
    zoom = t < ZOOM_S
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(members)))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(f"seed {s['seed']}, scale {s['scale']}, M = {s['n_taps']}, "
                 f"fs = {fs} Hz  ({', '.join(c['file'] for c in members)})")

    # Input once, in grey, as the common reference for every mu.
    freqs, p_d = power_spectrum(measured[warmup:], fs)
    ax1.plot(t[zoom], measured[zoom], color="0.7", lw=0.8, label="Input d")
    ax2.plot(freqs, 10 * np.log10(p_d + 1e-12), color="0.7", lw=0.8,
             label="Input d")
    ax3.plot(t, smoothed_power_db(measured, window), color="0.7", lw=0.8,
             label="Input d²")

    print(f"\n=== seed {s['seed']}, scale {s['scale']}, "
          f"M = {s['n_taps']}, fs = {fs} Hz ===")
    for cfg, color in zip(members, colors):
        mu = cfg["mu"]
        with np.errstate(over="ignore", invalid="ignore"):
            out, _, w = ARF(measured, reference, n_taps=cfg["n_taps"], mu=mu)
        atten = band_attenuation_db(measured[warmup:], out[warmup:], fs)
        print(f"  mu={mu:<8g} 50 Hz attenuation {atten:6.1f} dB   "
              f"final weights {np.round(w, 3)}")
        if rows is not None:
            rows.append({
                "case": cfg["file"], "name": cfg["name"],
                "seed": cfg["seed"], "seconds": cfg["seconds"], "fs": fs,
                "scale": cfg["scale"], "n_taps": cfg["n_taps"], "mu": mu,
                "warmup_s": cfg["warmup_s"], "method": "LMS",
                "atten_50hz_db": atten,
                "final_weights": " ".join(f"{x:.6g}" for x in w),
            })

        _, p_e = power_spectrum(out[warmup:], fs)
        ax1.plot(t[zoom], out[zoom], color=color, lw=0.8, label=f"mu={mu:g}")
        ax2.plot(freqs, 10 * np.log10(p_e + 1e-12), color=color, lw=0.8,
                 label=f"mu={mu:g} ({atten:.1f} dB)")
        ax3.plot(t, smoothed_power_db(out, window), color=color, lw=0.8,
                 label=f"mu={mu:g}")

    for ax in (ax1, ax3):
        ax.axvline(s["warmup_s"], color="k", ls=":", lw=0.8)
    ax2.axvline(LINE_FREQ, color="r", ls="--", lw=0.8)

    ax1.set_title("Time domain: output e (dotted line = warm-up end)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (a.u.)")
    ax2.set_title(f"Frequency domain after warm-up "
                  f"(legend: {LINE_FREQ:.0f} Hz attenuation)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax3.set_title(f"Learning curve: output power "
                  f"({s['smooth_window_s']} s moving average)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Power (dB)")
    for ax in (ax1, ax2, ax3):
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True)

    fig.tight_layout()
    return fig


def main():
    rows = []
    for members in group_by_signal(load_cases()).values():
        plot_group(members, rows)
    write_csv("test_cases.csv", rows)
    plt.show()


if __name__ == "__main__":
    main()
