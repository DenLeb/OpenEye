# Multi-GEMV integration test without resetting the hardware.
# Executes all 12 Q/K/V projections using token and projection selectors.
# Verifies DMA streams, GEMV results, and selector stability.

from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


@cocotb.test()
async def test_twelve_projection_gemvs(dut):

    data_dir = (
    Path(__file__).resolve().parent
    / "data"
    / "attention_mapper_v1"
    )

    # Reference data for both GEMV operations
    gemv_cases = [
        (proj_id, token, f"verification_streams/{kind}_t{token}.mem")
        for proj_id, kind in enumerate(("q", "k", "v"))
        for token in range(4)
    ]

    with np.load(data_dir / "inputs_used.npz") as tensors:
        x = tensors["X"].astype(np.int64)
        weights = {
            "q": tensors["WQ"].astype(np.int64),
            "k": tensors["WK"].astype(np.int64),
            "v": tensors["WV"].astype(np.int64),
        }

    # Initialize hardware only once
    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.token_select_i.value = 0
    dut.projection_select_i.value = 0

    # Activation RAM write interface
    dut.iact_we_i.value = 0
    dut.iact_wr_addr_i.value = 0
    dut.iact_wr_data_i.value = 0

    # Weight RAM write interface
    dut.wght_we_i.value = 0
    dut.wght_wr_addr_i.value = 0
    dut.wght_wr_data_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 20, unit="ns").start())

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    dut.rst_ni.value = 1

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    # Execute three independent GEMVs without another reset
    for run_id, (proj_id, token, input_file) in enumerate(gemv_cases):

        input_words = [
            int(line, 16)
            for line in (data_dir / input_file).read_text().splitlines()
            if line.strip()
        ]

        kind = ("q", "k", "v")[proj_id]

        expected = (
            x[token] @ weights[kind]
        ).tolist()

        assert len(input_words) == 939
        assert len(expected) == 8

        # Continue with the existing ready-wait loop...

        # Allow controller to return to IDLE
        for _ in range(3):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

        # Wait until OpenEye can accept a new stream
        for _ in range(200):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            if (int(dut.u_openeye.ready_dma_o.value) == 1
                    and int(dut.busy_o.value) == 0):
                break
        else:
            raise AssertionError(
                f"GEMV {run_id + 1}: OpenEye not ready"
            )

        dut._log.info(
            "Starting %s[%d] - GEMV %d/12",
            kind.upper(), token, run_id + 1
        )

        dut.token_select_i.value = token
        dut.projection_select_i.value = proj_id

        dut.start_i.value = 1

        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        dut.start_i.value = 0

        sent_count = 0
        raw_results = []

        # Monitor this GEMV
        for cycle in range(2500):

            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            if int(dut.u_openeye.enable_dma_i.value):

                assert sent_count < 939, (
                    f"{kind.upper()}[{token}]: Extra input word"
                )

                actual_word = int(dut.u_openeye.data_dma_i.value)

                assert actual_word == input_words[sent_count], (
                    f"{kind.upper()}[{token}]: "
                    f"Input mismatch at word {sent_count}"
                )

                sent_count += 1

                # Change external selections during transmission
                if sent_count == 100:
                    dut.token_select_i.value = (token + 1) % 4
                    dut.projection_select_i.value = (proj_id + 1) % 3

                # Restore original selections
                if sent_count == 200:
                    dut.token_select_i.value = token
                    dut.projection_select_i.value = proj_id

            if int(dut.result_write_o.value):

                address = int(dut.result_addr_o.value)
                word = int(dut.result_data_o.value)

                assert address == len(raw_results), (
                    f"GEMV {run_id + 1}: Result address mismatch"
                )

                raw_results.append(word)

            if int(dut.done_o.value):
                break

        else:
            raise AssertionError(
                f"GEMV {run_id + 1} timeout: "
                f"sent={sent_count}/939, "
                f"results={len(raw_results)}"
            )

        assert sent_count == 939
        assert len(raw_results) == 32

        # Decode signed 20-bit PSUM values
        decoded = []

        for word in raw_results:
            value = word & ((1 << 20) - 1)

            if value & (1 << 19):
                value -= 1 << 20

            decoded.append(value)

        # Restore logical output ordering
        actual = [0] * 32

        for beat, value in enumerate(decoded):
            filter_index = (beat % 2) * 16 + (beat // 2)
            actual[filter_index] = value

        assert actual[:8] == expected, (
            f"{kind.upper()}[{token}]: "
            f"Expected {expected}, got {actual[:8]}"
        )

        assert actual[8:] == [0] * 24, (
            f"{kind.upper()}[{token}]: Padding mismatch"
        )

        dut._log.info(
            "%s[%d] PASSED: cycles=%d, inputs=%d, outputs=%d",
            kind.upper(), token, cycle + 1,
            sent_count, len(raw_results)
        )

    dut._log.info(
        "All 12 Q/K/V projection GEMVs PASSED without reset"
    )