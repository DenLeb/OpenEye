# Multi-GEMV integration test without resetting the hardware.
# Executes Q[0] and Q[1] using selectable ROM banks.
# Verifies input streams, output results, and bank selection stability.

from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


@cocotb.test()
async def test_two_distinct_gemvs(dut):

    data_dir = Path(__file__).resolve().parent / "data"

# Reference data for both GEMV operations
    gemv_cases = [
        (
            0,
            "gemv_reference.mem",
            "gemv_expected_output.txt",
        ),
        (
            1,
            "gemv_q1_reference.mem",
            "gemv_q1_expected_output.txt",
        ),
    ]

    # Initialize hardware only once
    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.bank_select_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 20, unit="ns").start())

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    dut.rst_ni.value = 1

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    # Execute three independent GEMVs without another reset
    for run_id, (bank, input_file, output_file) in enumerate(gemv_cases):

        input_words = [
            int(line, 16)
            for line in (data_dir / input_file).read_text().splitlines()
            if line.strip()
        ]

        expected = [
            int(line)
            for line in (data_dir / output_file).read_text().splitlines()
            if line.strip()
        ]

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

        dut._log.info("Starting GEMV %d", run_id + 1)

        # Select the ROM bank before starting the GEMV
        dut.bank_select_i.value = bank

        # Send one start pulse
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
                    f"GEMV {run_id + 1}: Extra input word"
                )

                actual_word = int(dut.u_openeye.data_dma_i.value)

                assert actual_word == input_words[sent_count], (
                    f"GEMV {run_id + 1}: "
                    f"Input mismatch at word {sent_count}"
                )

                sent_count += 1

                # Try changing the external bank during transmission
                if sent_count == 100:
                    dut.bank_select_i.value = 1 - bank

                if sent_count == 200:
                    dut.bank_select_i.value = bank

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
            f"GEMV {run_id + 1}: "
            f"Expected {expected}, got {actual[:8]}"
        )

        assert actual[8:] == [0] * 24, (
            f"GEMV {run_id + 1}: Padding mismatch"
        )

        dut._log.info(
            "GEMV %d PASSED: cycles=%d, inputs=%d, outputs=%d",
            run_id + 1, cycle + 1, sent_count, len(raw_results)
        )

    dut._log.info("Both Q[0] and Q[1] GEMVs PASSED")
