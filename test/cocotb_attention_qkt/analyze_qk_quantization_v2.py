#!/usr/bin/env python3
"""Compare INT8 requantization candidates on verified OpenEye Q/K projection data.

Place at test/cocotb_attention_qkt/analyze_qk_quantization.py and run from the
OpenEye repository root:
    python test/cocotb_attention_qkt/analyze_qk_quantization.py \
      test/cocotb_attention_qkt/data/qk_raw_projection.npz

The saved NPZ must contain two 2-D integer arrays named q_raw and k_raw.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

# Import the actual Phase-0 quantization reference from the project rather
# than silently creating another potentially divergent specification.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from open_eye.attention_scheduler import requantize_pow2  # noqa: E402


def quantize_power_of_two(raw, per_row):
    raw = np.asarray(raw, dtype=np.int64)
    if per_row:
        rows = [requantize_pow2(row) for row in raw]
        q = np.stack([pair[0] for pair in rows])
        shifts = np.array([pair[1] for pair in rows], dtype=np.int64)
    else:
        q, shift = requantize_pow2(raw)
        shifts = np.full(raw.shape[0], shift, dtype=np.int64)
    # inverse scale = 2**shift: approximate raw = q * inverse scale
    inverse_scales = np.power(2.0, shifts.astype(np.float64))
    return q, inverse_scales, {"shift": shifts.tolist()}


def quantize_existing_openeye(raw):
    """Calibrated per-row quantization using the existing RTL formula.

    quantized_value = (mant * (psum + offset)) >>> exponent

    offset = 0, exponent = 20 (when scaling needed), and mantissa is a
    positive 25-bit signed integer.  The scale is bounded to avoid output
    overflow since the RTL does not saturate.

    This models Verilog signed arithmetic right-shift with Python/NumPy >>.
    """
    raw = np.asarray(raw, dtype=np.int64)
    result = np.empty_like(raw)
    inverse_scales = np.empty(raw.shape[0], dtype=np.float64)
    mantissas = []
    exponents = []
    for idx, row in enumerate(raw):
        max_abs = int(np.max(np.abs(row))) if row.size else 0
        if max_abs <= 127:
            mant, exponent = 1, 0  # Already representable exactly as INT8
        else:
            exponent = 20
            mant = (127 << exponent) // max_abs
        assert 0 < mant < (1 << 24), "Mantissa must fit signed 25-bit"
        q_row = (row * mant) >> exponent
        assert np.all((-128 <= q_row) & (q_row <= 127)), (
            "Unsafe quantizer configuration: signed INT8 overflow"
        )
        result[idx] = q_row
        inverse_scales[idx] = (2**exponent) / mant
        mantissas.append(mant)
        exponents.append(exponent)
    return result, inverse_scales, {
        "mantissa": mantissas, "exponent": exponents, "offset": [0] * len(mantissas)
    }


def quantize_openeye_per_tensor_pow2(raw):
    """Bit-exact model of the existing quantization_unit.v in pow2 mode.

    quant_mant=1, quant_offset=0, exponent=dynamic shift chosen to avoid
    signed INT8 overflow. Verilog signed >>> corresponds to NumPy >>,
    which rounds toward negative infinity (not nearest).
    """
    raw = np.asarray(raw, dtype=np.int64)
    max_abs = int(np.max(np.abs(raw))) if raw.size else 0
    shift = 0
    while max_abs > (127 << shift):
        shift += 1
    quant = raw >> shift
    assert np.all((-128 <= quant) & (quant <= 127)), "INT8 overflow"
    return quant, np.full(raw.shape[0], float(2**shift)), {
        "mantissa": [1] * raw.shape[0],
        "exponent": [shift] * raw.shape[0],
        "offset": [0] * raw.shape[0],
    }


def quantize_openeye_per_tensor_mantissa(raw):
    """Bit-exact arithmetic of existing RTL with one calibrated scale/tensor.

    Finds a representable 25-bit signed mantissa and shift; this is an
    illustrative max-absolute-value calibration, not an optimized solution.
    """
    raw = np.asarray(raw, dtype=np.int64)
    max_abs = int(np.max(np.abs(raw))) if raw.size else 0
    if max_abs <= 127:
        mantissa, exponent = 1, 0
    else:
        exponent = 20
        mantissa = (127 << exponent) // max_abs
    assert 0 < mantissa < (1 << 24), "Mantissa does not fit signed 25 bits"
    quant = (raw * mantissa) >> exponent
    assert np.all((-128 <= quant) & (quant <= 127)), "INT8 overflow"
    return quant, np.full(raw.shape[0], (2**exponent) / mantissa), {
        "mantissa": [mantissa] * raw.shape[0],
        "exponent": [exponent] * raw.shape[0],
        "offset": [0] * raw.shape[0],
    }


def evaluate(label, q_raw, k_raw, converter):
    q8, q_invscale, q_param = converter(q_raw)
    k8, k_invscale, k_param = converter(k_raw)

    q_recon = q8.astype(np.float64) * q_invscale[:, None]
    k_recon = k8.astype(np.float64) * k_invscale[:, None]
    s_ref = q_raw.astype(np.float64) @ k_raw.astype(np.float64).T
    # The following correctly uses distinct per-row scales for Q and K.
    s_int = q8.astype(np.int64) @ k8.astype(np.int64).T
    s_recon = s_int.astype(np.float64) * q_invscale[:, None] * k_invscale[None, :]
    error = s_recon - s_ref
    q_zero = int(np.count_nonzero((q_raw != 0) & (q8 == 0)))
    k_zero = int(np.count_nonzero((k_raw != 0) & (k8 == 0)))
    denom = float(np.linalg.norm(s_ref))
    relative_error = float(np.linalg.norm(error)) / denom if denom else float(np.linalg.norm(error))
    result = {
        "method": label,
        "q_lost_nonzero": q_zero,
        "k_lost_nonzero": k_zero,
        "q_mae": float(np.mean(np.abs(q_recon - q_raw))),
        "k_mae": float(np.mean(np.abs(k_recon - k_raw))),
        "score_mae": float(np.mean(np.abs(error))),
        "score_max_abs_error": float(np.max(np.abs(error))),
        "score_rel_frobenius_pct": relative_error * 100.0,
    }
    return result, q_param, k_param


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with integer q_raw and k_raw")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input} (run the projection Cocotb test to export it)")
    with np.load(args.input, allow_pickle=False) as data:
        q_raw = np.asarray(data["q_raw"], dtype=np.int64)
        k_raw = np.asarray(data["k_raw"], dtype=np.int64)
    if q_raw.ndim != 2 or k_raw.ndim != 2 or q_raw.shape[1] != k_raw.shape[1]:
        parser.error(f"Expected compatible 2D matrices, got Q={q_raw.shape}, K={k_raw.shape}")
    if np.max(np.abs(q_raw)) > 524288 or np.max(np.abs(k_raw)) > 524288:
        parser.error("Input contains values outside the expected signed 20-bit range")

    methods = [
        ("Per-tensor Pow2", lambda raw: quantize_power_of_two(raw, False)),
        ("Per-row Pow2", lambda raw: quantize_power_of_two(raw, True)),
        ("Per-row OpenEye mantissa/exp", quantize_existing_openeye),
        ("Per-tensor OpenEye m=1 / shift", quantize_openeye_per_tensor_pow2),
        ("Per-tensor OpenEye mantissa/exp", quantize_openeye_per_tensor_mantissa),
    ]
    print(f"Projection data: Q={q_raw.shape}, K={k_raw.shape}")
    print(f"Original nonzero values: Q={np.count_nonzero(q_raw)}, K={np.count_nonzero(k_raw)}")
    print(f"Max absolute: Q={np.max(np.abs(q_raw))}, K={np.max(np.abs(k_raw))}\n")

    rows = []
    for name, converter in methods:
        result, q_param, k_param = evaluate(name, q_raw, k_raw, converter)
        rows.append(result)
        print(name)
        print(f"  New zeros: Q={result['q_lost_nonzero']}, K={result['k_lost_nonzero']}")
        print(f"  Reconstructed Q/K MAE: {result['q_mae']:.5g} / {result['k_mae']:.5g}")
        print(f"  QK^T score MAE: {result['score_mae']:.5g}")
        print(f"  QK^T relative Frobenius error: {result['score_rel_frobenius_pct']:.5g}%")
        print(f"  Q parameters: {q_param}")
        print(f"  K parameters: {k_param}\n")

    outfile = args.input.parent / "qk_quantization_comparison.csv"
    with outfile.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to: {outfile}")
    print("Note: This is a numerical comparison; it does not yet test the RTL controller/feedback.")


if __name__ == "__main__":
    main()
