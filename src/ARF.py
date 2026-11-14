"""Baseline: LMS reference-noise canceller (Thakor & Zhu, 1991).

Implements the Type 1(a) adaptive filter of Fig. 1(a) / Section III-B in
N. V. Thakor and Y.-S. Zhu, "Applications of Adaptive Filtering to ECG
Analysis: Noise Cancellation and Arrhythmia Detection," IEEE Trans. Biomed.
Eng., 38(8), 785-794, 1991. doi:10.1109/10.83591

Primary input  : d[k] = s[k] + v[k]   (measured ECG = clean ECG + interference)
Reference input: r[k]                 (measurement correlated with v, not with s)

Transversal filter with the LMS update of Eq. (4):

    x_k     = [r[k], r[k-1], ..., r[k-M+1]]^T     (reference samples before
                                                   the recording are zero)
    y[k]    = w_k^T x_k                           (interference estimate)
    e[k]    = d[k] - y[k]                         (cleaned ECG estimate)
    w_{k+1} = w_k + 2 * mu * e[k] * x_k

The factor 2 follows the paper's convention (gradient of the full squared
error). The course guide's convention, w += mu * e * x, is the same update
with mu_guide = 2 * mu_paper.

Relationship to the adaptive recurrent filter (ARF, Section II-C, Eq. 5):
the ARF is this same algorithm with a one-hot impulse reference vector, so
only one weight is updated per sample and the desired signal appears at y
(Type 1(b)). Here the reference is a real signal and the output is e.
"""

import numpy as np
import matplotlib.pyplot as plt


def ARF(primary, reference, n_taps, mu):
    """Adaptive reference-noise cancellation with the LMS algorithm.

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

    Returns
    -------
    cleaned : ndarray, shape (N,)
        Residual e = d - y, the estimate of the clean ECG.
    noise_est : ndarray, shape (N,)
        Filter output y, the estimate of the interference.
    weights : ndarray, shape (M,)
        Final filter weights.

    Notes
    -----
    Weights start at zero for every call (one recording per call). The
    prediction and residual are computed before the weights are updated.
    Inputs are not modified. Cost is O(M) operations per sample.
    """
    d = np.asarray(primary, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)

    if d.ndim != 1 or r.ndim != 1:
        raise ValueError("primary and reference must be 1-D arrays")
    if d.shape != r.shape:
        raise ValueError("primary and reference must have the same length")
    if int(n_taps) != n_taps or n_taps < 1:
        raise ValueError("n_taps must be a positive integer")
    if mu < 0:
        raise ValueError("mu must be non-negative")

    n_taps = int(n_taps)
    w = np.zeros(n_taps)                 # w_0 = 0
    x = np.zeros(n_taps)                 # tapped delay line, zero before start
    cleaned = np.empty_like(d)
    noise_est = np.empty_like(d)

    for k in range(len(d)):
        x[1:] = x[:-1]                   # shift: x = [r[k], r[k-1], ...]
        x[0] = r[k]
        noise_est[k] = w @ x             # y = w^T x
        cleaned[k] = d[k] - noise_est[k] # e = d - y
        w += 2 * mu * cleaned[k] * x

    return cleaned, noise_est, w

def make_ecg(seed, seconds=12, fs=250, scale=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds*fs)) / fs
    clean = np.zeros_like(t)
    interval = rng.uniform(0.75, 1.05)

    pulses = [(0.12, -0.18, 0.035),
        (-0.15, -0.035, 0.012),
        (1.0, 0.0, 0.016),
        (-0.25, 0.035, 0.014),
        (0.30, 0.24, 0.065)]
    
    for beat in np.arange(0.6, seconds-0.4, interval):
        for amplitude, offset, width in pulses:
            clean += amplitude * np.exp(-0.5*((t-beat-offset)/width)**2)
    
    phase = rng.uniform(0, 2*np.pi)
    reference = scale * (np.sin(2*np.pi*50*t+phase) + 0.2*rng.normal(size=len(t)))
    pathway = np.array([0.35, -0.15, 0.08])
    interference = np.convolve(reference, pathway, mode="full")[:len(t)]
    measured = clean + interference
    measured += 0.02*rng.normal(size=len(t))
    
    return clean, measured, reference

def snr_db(clean, estimate, eps=1e-12):
    """SNR in dB of an estimate against the clean signal (guide, page 3).

    eps keeps the ratio and the logarithm finite when the error (or the
    clean signal) is exactly zero.
    """
    signal = np.sum(clean**2)
    error = np.sum((estimate - clean)**2)
    return 10*np.log10((signal + eps) / (error + eps))


def main():
    fs = 250
    seed = 0

    # clean = s (ground truth, never given to the filter)
    # sig   = d = s + v (measured ECG), reference = r (50 Hz + noise)
    clean, sig, reference = make_ecg(seed=seed, seconds=12, fs=fs, scale=1.0)
    t = np.arange(len(sig)) / fs

    out, noise_est, w = ARF(sig, reference, n_taps=8, mu=0.005)

    warmup = fs  # evaluate after 1 s, as the guide suggests
    print(f"Final weights: {np.round(w, 3)}  (true pathway: [0.35, -0.15, 0.08])")
    print(f"SNR input : {snr_db(clean[warmup:], sig[warmup:]):6.2f} dB")
    print(f"SNR output: {snr_db(clean[warmup:], out[warmup:]):6.2f} dB")

    freqs = np.fft.rfftfreq(len(sig), 1/fs)

    eps = 1e-12
    sig_pow = 10*np.log10((np.abs(np.fft.rfft(sig)) / len(sig))**2 + eps)
    out_pow = 10*np.log10((np.abs(np.fft.rfft(out)) / len(out))**2 + eps)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, sig, label="Input")
    ax1.plot(t, out, label="LMS output (e)")
    ax1.plot(t, clean, "--", label="Clean ECG")
    ax1.set_xlim(0, 4)  # show convergence and a few beats
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(freqs, sig_pow, label="Input")
    ax2.plot(freqs, out_pow, label="LMS output (e)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()