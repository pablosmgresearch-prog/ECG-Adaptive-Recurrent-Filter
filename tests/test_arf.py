"""Pass/fail checks for the baseline (src/ARF.py) and DCT-LMS
(src/DCT-LMS_ARF.py) noise cancellers.

Two kinds of checks, kept separate on purpose (project brief, "Verify the
baseline"):

* Code correctness: hand-computed example, limiting cases, shapes, finite
  outputs, no input mutation, input validation, agreement with a direct
  transcription of the update equation.
* Model validity: on data that satisfies the model assumptions, the filter
  recovers the true interference pathway and improves SNR.

Run with either:
    python tests/test_arf.py      (no extra dependencies)
    python -m pytest tests/       (if pytest is installed)
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from ARF import ARF, make_ecg, snr_db  # noqa: E402

# The file name contains a hyphen, so it cannot be imported normally.
_spec = importlib.util.spec_from_file_location(
    "dct_lms_arf", SRC / "DCT-LMS_ARF.py")
_dct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dct)
DCT_LMS_ARF, dct_matrix = _dct.DCT_LMS_ARF, _dct.dct_matrix

PATHWAY = np.array([0.35, -0.15, 0.08])  # hardcoded in make_ecg
DEV_SEEDS = range(20)                    # development seeds 0-19 (guide p.3)


def reference_lms(d, r, n_taps, mu):
    """Direct transcription of the update, written independently of ARF().

    Uses the paper's convention w += 2*mu*e*x, as ARF() does.
    """
    w = np.zeros(n_taps)
    e = np.empty(len(d))
    for n in range(len(d)):
        x = np.array([r[n - j] if n >= j else 0.0 for j in range(n_taps)])
        e[n] = d[n] - w @ x
        w = w + 2 * mu * e[n] * x
    return e, w


# ---------------------------------------------------------------- correctness

def test_hand_example():
    # Course guide p.2: d=[1,1], r=[1,1], M=1, mu_guide=0.5 -> e=[1, 0.5], w=0.75.
    # ARF() uses the paper's 2*mu convention, so mu_paper = 0.25.
    #   k=0: y=0,    e=1,   w = 0 + 2*0.25*1*1     = 0.5
    #   k=1: y=0.5,  e=0.5, w = 0.5 + 2*0.25*0.5*1 = 0.75
    e, y, w = ARF([1.0, 1.0], [1.0, 1.0], n_taps=1, mu=0.25)
    np.testing.assert_allclose(e, [1.0, 0.5])
    np.testing.assert_allclose(y, [0.0, 0.5])
    np.testing.assert_allclose(w, [0.75])


def test_zero_reference_passes_input_through():
    rng = np.random.default_rng(1)
    d = rng.normal(size=200)
    e, y, w = ARF(d, np.zeros(200), n_taps=5, mu=0.01)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(y, 0.0)
    np.testing.assert_array_equal(w, 0.0)


def test_zero_step_size_passes_input_through():
    rng = np.random.default_rng(2)
    d, r = rng.normal(size=(2, 200))
    e, y, w = ARF(d, r, n_taps=5, mu=0.0)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(w, 0.0)


def test_output_decomposes_primary():
    # e = d - y must hold exactly at every sample.
    c, d, r = make_ecg(seed=0)
    e, y, _ = ARF(d, r, n_taps=8, mu=0.005)
    np.testing.assert_allclose(e + y, d, rtol=0, atol=1e-12)


def test_matches_reference_implementation():
    rng = np.random.default_rng(3)
    d, r = rng.normal(size=(2, 300))
    for n_taps in (1, 4, 9):
        e, _, w = ARF(d, r, n_taps=n_taps, mu=0.01)
        e_ref, w_ref = reference_lms(d, r, n_taps, 0.01)
        np.testing.assert_allclose(e, e_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w, w_ref, rtol=1e-12, atol=1e-12)


def test_shapes_and_finite_outputs():
    c, d, r = make_ecg(seed=0)
    for n_taps in (1, 8, 32):
        e, y, w = ARF(d, r, n_taps=n_taps, mu=0.005)
        assert e.shape == d.shape and y.shape == d.shape
        assert w.shape == (n_taps,)
        assert np.all(np.isfinite(e)) and np.all(np.isfinite(y))
        assert np.all(np.isfinite(w))


def test_inputs_not_mutated():
    c, d, r = make_ecg(seed=0)
    d0, r0 = d.copy(), r.copy()
    ARF(d, r, n_taps=8, mu=0.005)
    np.testing.assert_array_equal(d, d0)
    np.testing.assert_array_equal(r, r0)


def test_accepts_lists():
    e, y, w = ARF([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], n_taps=2, mu=0.1)
    assert isinstance(e, np.ndarray) and e.shape == (3,)


def test_weights_reset_between_calls():
    c, d, r = make_ecg(seed=0)
    first = ARF(d, r, n_taps=8, mu=0.005)
    second = ARF(d, r, n_taps=8, mu=0.005)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_invalid_inputs_raise():
    good = np.zeros(10)
    bad_calls = [
        (np.zeros((2, 5)), np.zeros((2, 5)), 3, 0.1),  # not 1-D
        (good, np.zeros(9), 3, 0.1),                   # length mismatch
        (good, good, 0, 0.1),                          # n_taps < 1
        (good, good, 2.5, 0.1),                        # non-integer n_taps
        (good, good, 3, -0.1),                         # negative mu
    ]
    for args in bad_calls:
        try:
            ARF(*args)
        except ValueError:
            continue
        raise AssertionError(f"ARF{tuple(type(a).__name__ for a in args)} "
                             f"did not raise ValueError for {args[2:]}")


# ------------------------------------------------------------- model validity

def test_recovers_pathway_white_reference():
    # Ideal case: white reference (all directions excited), no ECG, no noise.
    # The Wiener solution is exactly the pathway, extra taps go to zero.
    rng = np.random.default_rng(4)
    r = rng.normal(size=5000)
    d = np.convolve(r, PATHWAY)[: len(r)]
    _, _, w = ARF(d, r, n_taps=6, mu=0.005)
    np.testing.assert_allclose(w[:3], PATHWAY, atol=1e-6)
    np.testing.assert_allclose(w[3:], 0.0, atol=1e-6)


def test_snr_improves_on_synthetic_ecg():
    # Development seeds 0-19, default settings, 1 s warm-up (guide p.3).
    # Measured gain is >= 12 dB on every seed; 10 dB leaves margin.
    fs, warmup = 250, 250
    for seed in DEV_SEEDS:
        c, d, r = make_ecg(seed=seed, fs=fs)
        e, _, _ = ARF(d, r, n_taps=8, mu=0.005)
        gain = snr_db(c[warmup:], e[warmup:]) - snr_db(c[warmup:], d[warmup:])
        assert gain > 10.0, f"seed {seed}: SNR gain {gain:.2f} dB <= 10 dB"


def test_diverges_above_stability_bound():
    # Paper convention: LMS mean-square stable roughly for
    # 0 < mu < 1 / (M * P_r). Far above it the output must blow up;
    # this confirms the step size really controls the update.
    c, d, r = make_ecg(seed=0)
    M = 8
    mu_max = 1.0 / (M * np.mean(r ** 2))
    with np.errstate(over="ignore", invalid="ignore"):
        e, _, _ = ARF(d, r, n_taps=M, mu=10 * mu_max)
    assert not np.all(np.isfinite(e)) or np.max(np.abs(e)) > 1e6


# ---------------------------------------------------- DCT-LMS: correctness

def test_dct_matrix_orthonormal_and_matches_scipy():
    rng = np.random.default_rng(5)
    for M in (1, 4, 8, 16):
        C = dct_matrix(M)
        np.testing.assert_allclose(C @ C.T, np.eye(M), atol=1e-12)
        x = rng.normal(size=M)
        np.testing.assert_allclose(C @ x, dct(x, type=2, norm="ortho"),
                                   atol=1e-12)


def test_dct_hand_example():
    # M=1, so the DCT is the identity and u = x = r[k]. d=[1,1], r=[2,2],
    # mu=0.25, beta=0.99. Bias-corrected power is exactly r^2 = 4 at every
    # step, so the step is 2*mu/4 = 0.125 (eps=1e-8 is negligible):
    #   k=0: y=0,           e=1,   v = 0     + 0.125*1*2   = 0.25
    #   k=1: y=0.25*2=0.5,  e=0.5, v = 0.25  + 0.125*0.5*2 = 0.375
    # Plain LMS (step 2*mu=0.5) gives w=1 then e=-1, w=0: the normalisation
    # matters.
    e, y, w = DCT_LMS_ARF([1.0, 1.0], [2.0, 2.0], n_taps=1, mu=0.25)
    np.testing.assert_allclose(e, [1.0, 0.5], atol=1e-7)
    np.testing.assert_allclose(y, [0.0, 0.5], atol=1e-7)
    np.testing.assert_allclose(w, [0.375], atol=1e-7)


def test_dct_ablation_equals_lms():
    # normalize=False switches off only the change; C orthonormal makes it
    # exactly LMS.
    c, d, r = make_ecg(seed=0)
    for M in (1, 8, 16):
        e0, y0, w0 = ARF(d, r, n_taps=M, mu=0.005)
        e1, y1, w1 = DCT_LMS_ARF(d, r, n_taps=M, mu=0.005, normalize=False)
        np.testing.assert_allclose(e1, e0, atol=1e-12)
        np.testing.assert_allclose(y1, y0, atol=1e-12)
        np.testing.assert_allclose(w1, w0, atol=1e-12)


def test_dct_zero_reference_and_zero_step_pass_input_through():
    rng = np.random.default_rng(6)
    d, r = rng.normal(size=(2, 200))
    e, y, w = DCT_LMS_ARF(d, np.zeros(200), n_taps=5, mu=0.01)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(w, 0.0)
    e, y, w = DCT_LMS_ARF(d, r, n_taps=5, mu=0.0)
    np.testing.assert_array_equal(e, d)
    np.testing.assert_array_equal(w, 0.0)


def test_dct_shapes_finite_and_no_mutation():
    c, d, r = make_ecg(seed=0)
    d0, r0 = d.copy(), r.copy()
    for M in (1, 8, 32):
        e, y, w = DCT_LMS_ARF(d, r, n_taps=M, mu=0.005)
        assert e.shape == d.shape and y.shape == d.shape
        assert w.shape == (M,)
        assert np.all(np.isfinite(e)) and np.all(np.isfinite(w))
        np.testing.assert_allclose(e + y, d, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(d, d0)
    np.testing.assert_array_equal(r, r0)


def test_dct_invalid_inputs_raise():
    good = np.zeros(10)
    bad_calls = [
        dict(primary=np.zeros((2, 5)), reference=np.zeros((2, 5))),
        dict(reference=np.zeros(9)),
        dict(n_taps=0),
        dict(n_taps=2.5),
        dict(mu=-0.1),
        dict(beta=1.0),
        dict(beta=-0.1),
        dict(eps=0.0),
    ]
    for override in bad_calls:
        kwargs = dict(primary=good, reference=good, n_taps=3, mu=0.1)
        kwargs.update(override)
        try:
            DCT_LMS_ARF(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"DCT_LMS_ARF did not raise for {override}")


# ------------------------------------------------- DCT-LMS: model validity

def test_dct_recovers_pathway_white_reference():
    # Returned weights are mapped back to the time domain (w = C^T v).
    rng = np.random.default_rng(4)
    r = rng.normal(size=5000)
    d = np.convolve(r, PATHWAY)[: len(r)]
    _, _, w = DCT_LMS_ARF(d, r, n_taps=6, mu=0.005)
    np.testing.assert_allclose(w[:3], PATHWAY, atol=1e-6)
    np.testing.assert_allclose(w[3:], 0.0, atol=1e-6)


def test_dct_snr_improves_on_synthetic_ecg():
    # Development seeds 0-19, mu=0.005, 1 s warm-up. Measured gain is
    # >= 13.9 dB on every seed; 10 dB leaves margin.
    fs, warmup = 250, 250
    for seed in DEV_SEEDS:
        c, d, r = make_ecg(seed=seed, fs=fs)
        e, _, _ = DCT_LMS_ARF(d, r, n_taps=8, mu=0.005)
        gain = snr_db(c[warmup:], e[warmup:]) - snr_db(c[warmup:], d[warmup:])
        assert gain > 10.0, f"seed {seed}: SNR gain {gain:.2f} dB <= 10 dB"


# --------------------------------------------------------------------- runner

def main():
    tests = [(n, f) for n, f in globals().items()
             if n.startswith("test_") and callable(f)]
    failed = 0
    rows = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            rows.append({"check": name, "passed": True, "message": ""})
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc}")
            rows.append({"check": name, "passed": False,
                         "message": str(exc).replace("\n", " ")})
    print(f"\n{len(tests) - failed}/{len(tests)} passed")

    # Same layout as the other result files; plain csv so this script
    # keeps working without matplotlib.
    results_dir = SRC.parent / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "checks.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
