"""Verify dynamic RTL shifts, correct pipeline writeback, and independent Q8/K8 reads."""
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.triggers import Timer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from open_eye.attention_scheduler import requantize_pow2  # noqa: E402


def signed(raw, bits):
    raw &= (1 << bits) - 1
    return raw - (1 << bits) if raw & (1 << (bits-1)) else raw


async def tick(dut):
    dut.clk_i.value = 0
    await Timer(5, unit="ns")
    dut.clk_i.value = 1
    await Timer(5, unit="ns")


async def load_and_run(dut, q, k, label):
    q = np.asarray(q, dtype=np.int64).reshape(4, 8)
    k = np.asarray(k, dtype=np.int64).reshape(4, 8)
    assert np.max(np.abs(q)) <= (1 << 19)
    assert np.max(np.abs(k)) <= (1 << 19)
    for idx in range(32):
        dut.raw_wr_index_i.value = idx
        dut.raw_wr_q_data_i.value = int(q.flat[idx]) & ((1 << 20)-1)
        dut.raw_wr_k_data_i.value = int(k.flat[idx]) & ((1 << 20)-1)
        dut.raw_wr_en_i.value = 1
        await tick(dut)
    dut.raw_wr_en_i.value = 0

    golden_q, qshift = requantize_pow2(q)
    golden_k, kshift = requantize_pow2(k)
    dut.start_i.value = 1
    await tick(dut)
    dut.start_i.value = 0
    assert int(dut.busy_o.value) == 1
    completed = False
    for cycle in range(120):
        await tick(dut)
        if int(dut.done_o.value):
            completed = True
            break
    assert completed, f"{label}: controller did not finish within 120 clocks"
    assert int(dut.busy_o.value) == 0
    assert int(dut.q_shift_o.value) == qshift, f"{label}: Q shift"
    assert int(dut.k_shift_o.value) == kshift, f"{label}: K shift"

    for qt in range(4):
        for qf in range(8):
            # Intentionally read unrelated Q and K addresses in the same cycle.
            kt, kf = (qt + 1) % 4, (qf + 3) % 8
            dut.q8_rd_token_i.value = qt
            dut.q8_rd_feature_i.value = qf
            dut.k8_rd_token_i.value = kt
            dut.k8_rd_feature_i.value = kf
            await Timer(1, unit="ns")
            qseen = signed(int(dut.q8_data_o.value), 8)
            kseen = signed(int(dut.k8_data_o.value), 8)
            assert qseen == golden_q[qt, qf], (
                f"{label} Q[{qt},{qf}] expected={golden_q[qt,qf]} got={qseen}"
            )
            assert kseen == golden_k[kt, kf], (
                f"{label} K[{kt},{kf}] expected={golden_k[kt,kf]} got={kseen}"
            )

    dut._log.info("PASS %s: Q-shift=%s, K-shift=%s, cycles=%s, 64/64 values match",
                  label, qshift, kshift, cycle + 1)

    if label == "real-projection":
        s_ref = q.astype(np.float64) @ k.astype(np.float64).T
        s_rec = (golden_q @ golden_k.T).astype(np.float64) * (2**(qshift+kshift))
        rel = np.linalg.norm(s_rec-s_ref)/np.linalg.norm(s_ref)*100
        dut._log.info("Real QK^T score reconstructed relative Frobenius error = %.6f%%", rel)
        assert abs(rel - 0.45688) < 0.001, f"unexpected baseline: {rel}"


@cocotb.test()
async def test_existing_openeye_quantizer_with_rtl_controller(dut):
    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.raw_wr_en_i.value = 0
    dut.raw_wr_index_i.value = 0
    dut.raw_wr_q_data_i.value = 0
    dut.raw_wr_k_data_i.value = 0
    dut.q8_rd_token_i.value = 0
    dut.q8_rd_feature_i.value = 0
    dut.k8_rd_token_i.value = 0
    dut.k8_rd_feature_i.value = 0
    await tick(dut)
    dut.rst_ni.value = 1

    data_file = Path(__file__).resolve().parent / "data" / "qk_raw_projection.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Please run your Projection test first: {data_file}")
    with np.load(data_file, allow_pickle=False) as data:
        await load_and_run(dut, data["q_raw"], data["k_raw"], "real-projection")

    # Shift=0; no rounding or writeback address can be incorrect.
    q = np.arange(-16, 16, dtype=np.int64).reshape(4, 8)
    k = -q
    await load_and_run(dut, q, k, "shift-zero")

    # Signed-20 extreme and small entries; catch abs(-524288) overflow.
    q = np.array([-524288, 524287, 1, -1, 31, -31, 32, -32] * 4).reshape(4, 8)
    k = np.array([524287, -524288, -7, 7, 63, -63, 64, -64] * 4).reshape(4, 8)
    await load_and_run(dut, q, k, "20-bit-extremes")

    # Zero-only input must give zero shift and zero output, without stale data.
    z = np.zeros((4, 8), dtype=np.int64)
    await load_and_run(dut, z, z, "all-zero")

    # Different shifts for Q and K, deliberately placed maxima at LAST address.
    q = np.array([3, -6, 18, -27] * 8, dtype=np.int64).reshape(4,8)
    k = np.array([30, -50, 210, -300] * 8, dtype=np.int64).reshape(4,8)
    q[-1,-1] = 127
    k[-1,-1] = 8129
    await load_and_run(dut, q, k, "independent-shifts-last-element")
