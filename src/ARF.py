import numpy as np
import matplotlib.pyplot as plt

def ARF(input_signal, reference, update):
    w_n = np.zeros(len(input_signal), dtype=np.float64)
    filt_signal = np.zeros(len(input_signal), dtype=np.float64)

    i = 0
    for k in range(len(input_signal)):
        if reference[k] == 1:
            i = 0 # new cycle starts
        filt_signal[k] = w_n[i] # y = w_i
        eta = input_signal[k] - w_n[i] # e = d - y
        w_n[i] = w_n[i] + 2*update*eta # w_i <- w_i + 2*mu*e
        i += 1

    return filt_signal

def main():
    f = 60
    fs = 250
    t = np.arange(0, 10, 1/fs)
    sig = np.sin(2*np.pi * f * t) + np.sin(2*np.pi * 5 * t)

    reference = np.zeros(len(sig))
    reference[::fs // f] = 1

    out = ARF(sig, reference, update=0.005)

    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    
    eps = 1e-12
    sig_pow = 10*np.log10((np.abs(np.fft.rfft(sig)) / len(sig))**2 + eps)
    out_pow = 10*np.log10((np.abs(np.fft.rfft(out)) / len(out))**2 + eps)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, sig, label="Input")
    ax1.plot(t, out, label="ARF output")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(freqs, sig_pow, label="Input")
    ax2.plot(freqs, out_pow, label="ARF output")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()