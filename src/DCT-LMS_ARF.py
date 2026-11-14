"""Modification: DCT-LMS (transform-domain LMS) reference-noise canceller.

Baseline: src/ARF.py, the LMS canceller of Thakor & Zhu (1991), Eq. (4).
Change: the tap vector is rotated into the DCT domain and every DCT
coefficient gets its own step size, normalised by a running estimate of
that coefficient's power.

    x_k   = [r[k], r[k-1], ..., r[k-M+1]]^T      (same delay line as ARF)
    u_k   = C x_k                                (C: M x M orthonormal DCT-II)
    y[k]  = v_k^T u_k                            (interference estimate)
    e[k]  = d[k] - y[k]                          (cleaned ECG estimate)
    P_k   = beta * P_{k-1} + (1 - beta) * u_k^2  (per-coefficient power)
    v_{k+1} = v_k + 2 * mu * e[k] * u_k / (P_k + eps)

Only the last line differs from ARF beyond the rotation: ARF uses one
step size 2*mu for every tap, here coefficient i uses 2*mu / (P_i + eps).

Why: the reference is a strong 50 Hz sine plus weak white noise, so the
autocorrelation matrix of x_k has a large eigenvalue spread. Plain LMS
learns the weak (broadband) directions very slowly. The DCT approximately
decorrelates x_k, and dividing by each coefficient's power equalises the
convergence speed of all directions.

Ablation: with normalize=False the update is v += 2*mu*e*u. Because C is
orthonormal, v = C w and this is exactly ARF() (up to round-off).

Attribution: S. S. Narayan, A. M. Peterson and M. J. Narasimha,
"Transform domain LMS algorithm," IEEE Trans. ASSP, 31(3), 609-615, 1983;
F. Beaufays, "Transform-domain adaptive filters: an analytical approach,"
IEEE Trans. Signal Processing, 43(2), 422-431, 1995.

Note on mu: with normalisation the step is dimensionless (the reference
power is divided out), so its useful range differs from ARF's mu. Both
must be tuned separately on validation data. Roughly, 0 < mu < 1/M.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ARF import ARF, make_ecg, snr_db  # noqa: E402  (reused from baseline)


def dct_matrix(n_taps):
    """Orthonormal DCT-II matrix C, shape (M, M), so that C @ x == dct(x).

    Built once with scipy by transforming the identity, then applied as a
    matrix product inside the loop (faster than calling dct per sample).
    """
    return dct(np.eye(n_taps), type=2, norm="ortho", axis=0)


def DCT_LMS_ARF(primary, reference, n_taps, mu, beta=0.99, eps=1e-8,
                normalize=True):
    """Adaptive reference-noise cancellation with DCT-domain LMS.

    Parameters
    ----------
    primary : array_like, shape (N,)
        Measured signal d = s + v.
    reference : array_like, shape (N,)
        Reference input r, correlated with the interference v.
    n_taps : int
        Filter length M (number of weights), M >= 1.
    mu : float
        Step size (paper convention, update uses 2*mu). mu >= 0.
    beta : float
        Forgetting factor of the per-coefficient power estimate, 0 <= beta < 1.
        Memory is about 1 / (1 - beta) samples.
    eps : float
        Small positive regulariser added to the power before dividing.
    normalize : bool
        True: DCT-LMS. False: ablation, identical to ARF().

    Returns
    -------
    cleaned : ndarray, shape (N,)
        Residual e = d - y, the estimate of the clean ECG.
    noise_est : ndarray, shape (N,)
        Filter output y, the estimate of the interference.
    weights : ndarray, shape (M,)
        Final weights mapped back to the time domain, w = C^T v, so they
        are directly comparable with ARF()'s weights.

    Notes
    -----
    Weights and power estimates start at zero for every call. The power
    estimate is bias-corrected, P / (1 - beta^(k+1)), so the first samples
    are not divided by a near-zero power. Inputs are not modified.
    Cost is O(M^2) operations per sample (the DCT matrix product), against
    O(M) for ARF().
    """
    d = np.asarray(primary, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)

    # Same input checks as ARF(), plus the two new parameters.
    if d.ndim != 1 or r.ndim != 1:
        raise ValueError("primary and reference must be 1-D arrays")
    if d.shape != r.shape:
        raise ValueError("primary and reference must have the same length")
    if int(n_taps) != n_taps or n_taps < 1:
        raise ValueError("n_taps must be a positive integer")
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    if eps <= 0:
        raise ValueError("eps must be positive")

    n_taps = int(n_taps)
    C = dct_matrix(n_taps)
    v = np.zeros(n_taps)                 # DCT-domain weights, v_0 = 0
    x = np.zeros(n_taps)                 # tapped delay line, as in ARF
    P = np.zeros(n_taps)                 # per-coefficient power estimate
    cleaned = np.empty_like(d)
    noise_est = np.empty_like(d)

    beta_k = 1.0                         # beta^(k+1), for bias correction
    for k in range(len(d)):
        x[1:] = x[:-1]                   # shift: x = [r[k], r[k-1], ...]
        x[0] = r[k]
        u = C @ x                        # rotate into the DCT domain
        noise_est[k] = v @ u             # y = v^T u
        cleaned[k] = d[k] - noise_est[k] # e = d - y
        if normalize:
            P = beta * P + (1 - beta) * u * u
            beta_k *= beta
            step = 2 * mu / (P / (1 - beta_k) + eps)
        else:
            step = 2 * mu                # ablation: plain LMS
        v += step * cleaned[k] * u

    return cleaned, noise_est, C.T @ v


def main():
    """Baseline vs DCT-LMS on one synthetic recording (seed 0)."""
    fs = 250
    clean, sig, reference = make_ecg(seed=0, seconds=12, fs=fs, scale=1.0)
    t = np.arange(len(sig)) / fs
    warmup = fs  # evaluate after 1 s

    out_lms, _, w_lms = ARF(sig, reference, n_taps=8, mu=0.005)
    out_dct, _, w_dct = DCT_LMS_ARF(sig, reference, n_taps=8, mu=0.005)

    print("True pathway        : [0.35, -0.15, 0.08]")
    print(f"LMS weights         : {np.round(w_lms, 3)}")
    print(f"DCT-LMS weights     : {np.round(w_dct, 3)}")
    print(f"SNR input  : {snr_db(clean[warmup:], sig[warmup:]):6.2f} dB")
    print(f"SNR LMS    : {snr_db(clean[warmup:], out_lms[warmup:]):6.2f} dB")
    print(f"SNR DCT-LMS: {snr_db(clean[warmup:], out_dct[warmup:]):6.2f} dB")

    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    eps = 1e-12

    def pow_db(x):
        return 10 * np.log10((np.abs(np.fft.rfft(x)) / len(x)) ** 2 + eps)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, sig, label="Input")
    ax1.plot(t, out_lms, label="LMS output")
    ax1.plot(t, out_dct, label="DCT-LMS output")
    ax1.plot(t, clean, "--", label="Clean ECG")
    ax1.set_xlim(0, 4)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(freqs, pow_db(sig), label="Input")
    ax2.plot(freqs, pow_db(out_lms), label="LMS output")
    ax2.plot(freqs, pow_db(out_dct), label="DCT-LMS output")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
