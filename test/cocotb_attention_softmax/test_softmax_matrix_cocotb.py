
import random
from pathlib import Path

import cocotb
import numpy as np

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from softmax_exp_reference import exp_row


def reference_row(scores, q_shift, k_shift):
    exp_values = [
        int(v) for v in exp_row(scores, q_shift, k_shift)
    ]

    total = sum(exp_values)

    if total == 0:
        return [0] * 4

    return [
        min(32768, (e * 32768 + total // 2) // total)
        for e in exp_values
    ]


async def score_buffer_model(dut, model):
    """
    Emulate the existing combinational score read port.
    One score is available per valid clock cycle.
    """
    cycle = 0

    while True:
        await FallingEdge(dut.clk_i)

        row = int(dut.score_rd_token_o.value)
        col = int(dut.score_rd_index_o.value)

        score = int(model["matrix"][row][col])

        # Signed INT20 represented as 20-bit two's complement
        dut.score_rd_data_i.value = score & 0xFFFFF

        cycle += 1
        stall_every = model["stall_every"]

        valid = not (
            stall_every and cycle % stall_every == 0
        )

        dut.score_rd_valid_i.value = int(valid)


async def check_matrix(dut, model, matrix, q_shift, k_shift,
                       stall_every=0):

    expected = [
        reference_row(row, q_shift, k_shift)
        for row in matrix
    ]

    model["matrix"] = matrix
    model["stall_every"] = stall_every

    # One start for the entire 4x4 matrix
    await FallingEdge(dut.clk_i)

    dut.q_shift_i.value = q_shift
    dut.k_shift_i.value = k_shift
    dut.start_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    assert int(dut.busy_o.value) == 1
    assert int(dut.complete_o.value) == 0

    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    # Change external shifts to verify internal registers.
    dut.q_shift_i.value = 0
    dut.k_shift_i.value = 0

    # Wait for all four rows to finish.
    for cycle in range(300):

        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.done_o.value) == 1:
            assert int(dut.complete_o.value) == 1
            assert int(dut.busy_o.value) == 0
            break

    else:
        raise AssertionError("Full Softmax matrix timeout")

    # done_o must be a one-cycle pulse.
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    assert int(dut.done_o.value) == 0
    assert int(dut.complete_o.value) == 1

    # Read and check all 16 stored probabilities.
    actual = []

    for row in range(4):
        actual_row = []

        for col in range(4):
            await FallingEdge(dut.clk_i)

            dut.prob_rd_token_i.value = row
            dut.prob_rd_index_i.value = col

            await Timer(1, unit="ns")

            assert int(dut.prob_rd_valid_o.value) == 1, (
                f"Probability missing at [{row}][{col}]"
            )

            actual_row.append(
                int(dut.prob_rd_data_o.value)
            )

        actual.append(actual_row)

    assert actual == expected, (
        f"Expected:\n{expected}\n"
        f"Actual:\n{actual}"
    )

    dut._log.info(
        "Matrix PASS (q_shift=%d, k_shift=%d, stalls=%d)",
        q_shift, k_shift, stall_every
    )


@cocotb.test()
async def test_full_softmax_matrix(dut):

    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.q_shift_i.value = 0
    dut.k_shift_i.value = 0
    dut.score_rd_data_i.value = 0
    dut.score_rd_valid_i.value = 0
    dut.prob_rd_token_i.value = 0
    dut.prob_rd_index_i.value = 0

    await Timer(25, unit="ns")
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    model = {
        "matrix": [[0] * 4 for _ in range(4)],
        "stall_every": 0,
    }

    cocotb.start_soon(score_buffer_model(dut, model))

    # Test 1: Existing QKT reference data
    ref_path = (
        Path(__file__).resolve().parent.parent
        / "cocotb_attention_qkt"
        / "data"
        / "qkt_single_8x4"
        / "qkt_t0_reference.npz"
    )

    with np.load(ref_path, allow_pickle=False) as f:
        scores = (
            f["Q8"].astype(np.int64)
            @ f["K8"].astype(np.int64).T
        ).tolist()

        q_shift = int(f["q_shift"])
        k_shift = int(f["k_shift"])

    await check_matrix(
        dut, model, scores, q_shift, k_shift
    )

    # Test 2: All-equal scores
    equal_matrix = [[100] * 4 for _ in range(4)]

    await check_matrix(
        dut, model, equal_matrix, 6, 6
    )

    # Test 3: Signed 20-bit boundary values
    boundary_matrix = [
        [-524288, -500000, 524287, 0],
        [524287, -524288, 10, -10],
        [0, 0, 0, 0],
        [100, 200, 300, 400],
    ]

    await check_matrix(
        dut, model, boundary_matrix, 13, 13
    )

    # Test 4: Random matrix with read-valid stalls
    rng = random.Random(42)

    random_matrix = [
        [rng.randint(-20000, 20000) for _ in range(4)]
        for _ in range(4)
    ]

    await check_matrix(
        dut, model, random_matrix, 3, 8,
        stall_every=5
    )

    dut._log.info("All full-matrix Softmax tests passed")
