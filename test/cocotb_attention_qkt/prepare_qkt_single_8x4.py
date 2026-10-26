"""Prepare one 8x4 Row-Stationary QK^T GEMV as an offline DMA image.

Run this file from OpenEye/test/cocotb_attention_qkt/ (or any working
folder after copying it to that directory).  It does NOT simulate hardware.
Existing projection/quantization files are not modified.
"""

import os
from pathlib import Path

import numpy as np
from open_eye import open_eye_parameters as oep
from open_eye.attention_scheduler import requantize_pow2
from open_eye.attention_projection_mapper import (
    AttentionProjectionMapper,
    BASELINE_ENV,
    SECTION_NAMES,
)

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "data" / "qk_raw_projection.npz"
OUT = HERE / "data" / "qkt_single_8x4"

for key, value in BASELINE_ENV.items():
    os.environ.setdefault(key, value)
# Lock the tested Row-Stationary mode even if the terminal has an old override.
os.environ["DATAFLOW"] = "row_stationary"

with np.load(RAW_PATH, allow_pickle=False) as data:
    q_raw = data["q_raw"].astype(np.int64)
    k_raw = data["k_raw"].astype(np.int64)

q8, q_shift = requantize_pow2(q_raw)
k8, k_shift = requantize_pow2(k_raw)
assert q8.shape == (4, 8), f"Unexpected Q8 shape: {q8.shape}"
assert k8.shape == (4, 8), f"Unexpected K8 shape: {k8.shape}"

params = oep.get_oep(1)
mapper = AttentionProjectionMapper(params, k_pad=8, n_pad=4)
sections = mapper.make_pass(q8[0], k8)
stream = [int(word) for name in SECTION_NAMES for word in sections[name]]
expected = (q8[0].astype(np.int64) @ k8.astype(np.int64).T)

assert len(stream) == 677, f"Unexpected DMA word count: {len(stream)}"
assert expected.shape == (4,), f"Unexpected output shape: {expected.shape}"

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "qkt_t0_dma.mem").write_text(
    "".join(f"{word:016x}\n" for word in stream), encoding="ascii"
)
np.savez(
    OUT / "qkt_t0_reference.npz",
    Q8=q8,
    K8=k8,
    expected=expected,
    q_shift=np.int64(q_shift),
    k_shift=np.int64(k_shift),
)

print("PASS: Prepared one QKT GEMV (K=8, N=4, Row-Stationary baseline)")
print("DMA words:", len(stream))
print("Sections:", {name: len(sections[name]) for name in SECTION_NAMES})
print("Expected scores:", expected.tolist())
print("Q/K shifts:", int(q_shift), int(k_shift))
print("Saved under:", OUT)
print("NOTE: This is offline preparation only; hardware has NOT yet been tested.")
