
import random
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from softmax_exp_reference import exp_row


def pack_scores(scores):
    # Pack four signed 20-bit scores into 80 bits.
    return sum(
        (int(score) & 0xFFFFF) << (20 * i)
        for i, score in enumerate(scores)
    )


def reference_row(scores, q_shift, k_shift):
    # Bit-exact exponential reference.
    exp_values = exp_row(scores, q_shift, k_shift)
    total = sum(exp_values)

    if total == 0:
        return [0] * 4

    # Same integer rounding as the divider RTL.
    return [
        min(32768, (exp * 32768 + total // 2) // total)
        for exp in exp_values
    ]


async def check_row(dut, scores, q_shift, k_shift):
    expected = reference_row(scores, q_shift, k_shift)

    # Start a new row.
    await FallingEdge(dut.clk_i)

    dut.scores_i.value = pack_scores(scores)
    dut.q_shift_i.value = q_shift
    dut.k_shift_i.value = k_shift
    dut.start_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    assert int(dut.busy_o.value) == 1, "Row did not start"

    # Remove start and change external inputs.
    # The row module must use its registered values.
    await FallingEdge(dut.clk_i)

    dut.start_i.value = 0
    dut.scores_i.value = 0
    dut.q_shift_i.value = 0
    dut.k_shift_i.value = 0

    # Wait for complete row normalization.
    for cycle in range(40):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.done_o.value) == 1:
            packed = int(dut.probabilities_o.value)

            actual = [
                (packed >> (16 * i)) & 0xFFFF
                for i in range(4)
            ]

            assert actual == expected, (
                f"Scores: {scores}\n"
                f"Shifts: {q_shift}, {k_shift}\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

            assert int(dut.busy_o.value) == 0

            # done_o must last exactly one clock cycle.
            await RisingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            assert int(dut.done_o.value) == 0
            return

    raise AssertionError(
        f"Softmax row timeout: scores={scores}"
    )


@cocotb.test()
async def test_softmax_row(dut):
    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.scores_i.value = 0
    dut.q_shift_i.value = 0
    dut.k_shift_i.value = 0

    await Timer(25, unit="ns")
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # Reuse existing QKT reference data.
    ref_path = (
        Path(__file__).resolve().parent.parent
        / "cocotb_attention_qkt"
        / "data"
        / "qkt_single_8x4"
        / "qkt_t0_reference.npz"
    )

    with np.load(ref_path, allow_pickle=False) as f:
        matrix = (
            f["Q8"].astype(np.int64)
            @ f["K8"].astype(np.int64).T
        )
        q_shift = int(f["q_shift"])
        k_shift = int(f["k_shift"])

    # Test all 4 rows from the real QKT test data.
    for row in matrix:
        await check_row(
            dut, [int(x) for x in row], q_shift, k_shift
        )

    # Boundary cases
    cases = [
        ([0, 0, 0, 0], 6, 6),
        ([100, 100, 100, 100], 6, 6),
        ([100, -100000, -200000, -300000], 6, 6),
        ([-524288, -500000, 524287, 0], 13, 13),
        ([-1, -2, -3, -4], 0, 0),
        ([524287, -524288, 7, -8], 5, 9),
    ]

    for scores, q, k in cases:
        await check_row(dut, scores, q, k)

    # Random input rows and quantization shifts.
    rng = random.Random(42)

    for _ in range(100):
        scores = [
            rng.randint(-(1 << 19), (1 << 19) - 1)
            for _ in range(4)
        ]

        q = rng.randint(0, 13)
        k = rng.randint(0, 13)

        await check_row(dut, scores, q, k)

    dut._log.info("All 110 Softmax row tests passed")
