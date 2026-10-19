from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def tick(dut):
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_rom_adapter_stream(dut):

    # Load expected DMA words
    test_dir = Path(__file__).resolve().parent
    mem_file = test_dir / "data" / "gemv_reference.mem"

    expected_words = [
        int(line.strip(), 16)
        for line in mem_file.read_text().splitlines()
        if line.strip()
    ]

    assert len(expected_words) == 939

    # Initialize signals
    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.dma_ready_i.value = 0

    dut.dma_result_data_i.value = 0
    dut.dma_result_valid_i.value = 0
    dut.dma_result_last_i.value = 0

    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    # Reset
    await tick(dut)
    await tick(dut)

    dut.rst_ni.value = 1
    await tick(dut)

    # Start one GEMV
    dut.start_i.value = 1
    await tick(dut)
    dut.start_i.value = 0

    # DMA is not ready yet
    for _ in range(4):
        await tick(dut)
        assert int(dut.dma_valid_o.value) == 0

    # Allow transmission
    dut.dma_ready_i.value = 1
    await tick(dut)

    # Verify all 939 transmitted words
    for address, expected_word in enumerate(expected_words):

        actual_address = int(dut.stream_addr_o.value)
        actual_word = int(dut.dma_data_o.value)
        valid = int(dut.dma_valid_o.value)

        assert valid == 1, f"DMA not valid at word {address}"

        assert actual_address == address, (
            f"Address mismatch: expected {address}, "
            f"got {actual_address}"
        )

        assert actual_word == expected_word, (
            f"Data mismatch at address {address}: "
            f"expected {expected_word:016X}, "
            f"got {actual_word:016X}"
        )

        await tick(dut)

    # Input transfer must be finished
    assert int(dut.dma_valid_o.value) == 0

    # Adapter should now be waiting for OpenEye results
    assert int(dut.dma_result_ready_o.value) == 1
    assert int(dut.done_o.value) == 0

    dut._log.info(
        "ROM + DMA Adapter PASSED: %d words transmitted",
        len(expected_words)
    )