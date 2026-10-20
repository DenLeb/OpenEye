# Single-GEMV integration test for the real OpenEye accelerator.
# Verifies DMA input streaming and Q[0] results against golden references.

from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


@cocotb.test()
async def test_one_real_gemv(dut):

    data_dir = Path(__file__).resolve().parent / "data"

    # Golden input stream
    input_words = [
        int(line, 16)
        for line in
        (data_dir / "gemv_reference.mem").read_text().splitlines()
        if line.strip()
    ]

    # Golden mathematical output
    expected = [
        int(line)
        for line in
        (data_dir / "gemv_expected_output.txt").read_text().splitlines()
        if line.strip()
    ]

    assert len(input_words) == 939
    assert len(expected) == 8

    # Initialize hardware
    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0

    cocotb.start_soon(
        Clock(dut.clk_i, 20, unit="ns").start()
    )

    # Reset
    for _ in range(6):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    dut.rst_ni.value = 1

    # Allow OpenEye reset synchronizer to settle
    for _ in range(4):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    # Start one GEMV
    dut.start_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    dut.start_i.value = 0

    sent_count = 0
    raw_results = []

    # Wait for the GEMV to finish
    for cycle in range(2500):

        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        # Monitor input words entering OpenEye
        if int(dut.u_openeye.enable_dma_i.value):

            assert sent_count < len(input_words), (
                "More than 939 input words transmitted"
            )

            actual_word = int(dut.u_openeye.data_dma_i.value)
            expected_word = input_words[sent_count]

            assert actual_word == expected_word, (
                f"Input mismatch at word {sent_count}: "
                f"expected {expected_word:016X}, "
                f"got {actual_word:016X}"
            )

            sent_count += 1

        # Capture results accepted by our DMA adapter
        if int(dut.result_write_o.value):

            address = int(dut.result_addr_o.value)
            word = int(dut.result_data_o.value)

            assert address == len(raw_results), (
                f"Unexpected result address: {address}"
            )

            raw_results.append(word)

        if int(dut.done_o.value):
            break

    else:
        raise AssertionError(
            f"GEMV timeout: sent={sent_count}/939, "
            f"results={len(raw_results)}"
        )

    assert sent_count == 939, (
        f"Only {sent_count} input words transmitted"
    )

    # The 32 padded output values use one DMA word each
    assert len(raw_results) == 32, (
        f"Expected 32 result words, got {len(raw_results)}"
    )

    # Decode signed 20-bit PSUM values
    decoded = []

    for word in raw_results:
        raw = word & ((1 << 20) - 1)

        if raw & (1 << 19):
            raw -= 1 << 20

        decoded.append(raw)

    # OpenEye interleaves its two cluster columns
    actual = [0] * 32

    for beat, value in enumerate(decoded):
        filter_index = (beat % 2) * 16 + (beat // 2)
        actual[filter_index] = value

    assert actual[:8] == expected, (
        f"GEMV mismatch:\n"
        f"Expected: {expected}\n"
        f"Actual:   {actual[:8]}"
    )

    assert actual[8:] == [0] * 24, (
        "Nonzero values found in padded outputs"
    )

    dut._log.info("Input words transmitted: %d", sent_count)
    dut._log.info("DMA result words received: %d", len(raw_results))
    dut._log.info("GEMV result: %s", actual[:8])
    dut._log.info("Real OpenEye GEMV integration PASSED")