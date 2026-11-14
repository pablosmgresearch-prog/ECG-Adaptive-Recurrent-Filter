"""Test cases for the ARF / LMS noise canceller in ARF.py.

Each test case is described by a JSON file (test_case_1.json ... _3.json)
in the test_cases/ folder. For every case the script runs the filter and plots:

  1. RMSE between the clean reference ECG and the filtered signal over time
     (sliding window), compared with the unfiltered input.
  2. The frequency domain (power spectrum) of input, filtered and clean ECG.
  3. The learning error across time: (e[k] - s[k])^2, raw and smoothed.

Run from anywhere:  python tests/test.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ARF import ARF, make_ecg  # noqa: E402
CASES_DIR = HERE / "test_cases"
CASE_FILES = ["test_case_1.json", "test_case_2.json", "test_case_3.json"]


def load_case(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sliding_rmse(a, b, window):
    """RMSE of (a - b) over a centred moving window of `window` samples."""
    sq = (a - b) ** 2
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(sq, kernel, mode="same"))


def power_db(x):
    return 10 * np.log10((np.abs(np.fft.rfft(x)) / len(x)) ** 2 + 1e-12)


def run_case(cfg):
    fs = cfg["fs"]
    clean, measured, reference = make_ecg(
        seed=cfg["seed"], seconds=cfg["seconds"], fs=fs, scale=cfg["scale"]
    )
    t = np.arange(len(measured)) / fs
    warmup = int(cfg["warmup_s"] * fs)

    out, _, w = ARF(measured, reference, n_taps=cfg["n_taps"], mu=cfg["mu"])

    rmse_in = rmse(clean[warmup:], measured[warmup:])
    rmse_out = rmse(clean[warmup:], out[warmup:])

    print(f"\n=== {cfg['name']} ===")
    print(f"n_taps={cfg['n_taps']}  mu={cfg['mu']}  scale={cfg['scale']}  fs={fs}")
    print(f"Final weights: {np.round(w, 3)}")
    print(f"RMSE input : {rmse_in:.4f}")
    print(f"RMSE output: {rmse_out:.4f}")

    # ---- plots ----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(cfg["name"])

    window = max(1, int(cfg["rmse_window_s"] * fs))

    # Learning error: squared error of the filter output against the clean
    # ECG, (e[k] - s[k])^2, i.e. the residual interference left at step k.
    learn_err = (out - clean) ** 2
    learn_err_smooth = np.convolve(learn_err, np.ones(window) / window, mode="same")
    ax1.plot(t, sliding_rmse(clean, measured, window), label="Input vs clean")
    ax1.plot(t, sliding_rmse(clean, out, window), label="Filtered vs clean")
    ax1.axvline(cfg["warmup_s"], color="k", ls=":", label="Warm-up end")
    ax1.set_title(f"RMSE (input {rmse_in:.4f}, filtered {rmse_out:.4f})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMSE")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)

    freqs = np.fft.rfftfreq(len(measured), 1 / fs)
    ax2.plot(freqs, power_db(measured), label="Input")
    ax2.plot(freqs, power_db(out), label="Filtered")
    ax2.plot(freqs, power_db(clean), "--", label="Clean")
    ax2.set_title("Frequency domain")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(t, learn_err, alpha=0.3, label="Squared error")
    ax3.plot(t, learn_err_smooth, label="Smoothed")
    ax3.axvline(cfg["warmup_s"], color="k", ls=":", label="Warm-up end")
    ax3.set_title(f"Learning error (mu = {cfg['mu']})")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("(e - s)^2")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)

    fig.tight_layout()
    return fig


def main():
    for name in CASE_FILES:
        run_case(load_case(CASES_DIR / name))
    plt.show()


if __name__ == "__main__":
    main()
