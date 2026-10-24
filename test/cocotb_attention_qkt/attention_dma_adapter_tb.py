import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def tick(dut):
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_dma_adapter(dut):

    # --------------------------------------------------
    # 1. Initialize inputs and reset
    # --------------------------------------------------

    dut.clk_i.value = 0
    dut.rst_ni.value = 0

    dut.gemv_start_i.value = 0
    dut.stream_word_count_i.value = 3
    dut.stream_data_i.value = 0

    dut.dma_ready_i.value = 0

    dut.dma_result_data_i.value = 0
    dut.dma_result_valid_i.value = 0
    dut.dma_result_last_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())

    await tick(dut)
    dut.rst_ni.value = 1

    # --------------------------------------------------
    # 2. Request one GEMV transaction
    # --------------------------------------------------

    dut.gemv_start_i.value = 1
    await tick(dut)
    dut.gemv_start_i.value = 0

    assert int(dut.busy_o.value) == 1

    # Adapter must wait while OpenEye is not ready
    for _ in range(2):
        await tick(dut)
        assert int(dut.dma_valid_o.value) == 0

    # Simulate OpenEye becoming ready
    dut.dma_ready_i.value = 1
    await tick(dut)

    # --------------------------------------------------
    # 3. Send three DMA words
    # --------------------------------------------------

    input_words = [0x11, 0x22, 0x33]

    for address, word in enumerate(input_words):

        # Simulate a memory returning the requested word
        dut.stream_data_i.value = word
        await Timer(1, unit="ns")

        assert int(dut.stream_addr_o.value) == address
        assert int(dut.dma_valid_o.value) == 1
        assert int(dut.dma_data_o.value) == word

        await tick(dut)

    # All input words have now been sent
    assert int(dut.dma_valid_o.value) == 0
    assert int(dut.dma_result_ready_o.value) == 1

    # --------------------------------------------------
    # 4. Simulate two output words from OpenEye
    # --------------------------------------------------

    results = {}

    for address, word in enumerate([0xAA, 0xBB]):

        dut.dma_result_data_i.value = word
        dut.dma_result_valid_i.value = 1
        dut.dma_result_last_i.value = int(address == 1)

        await Timer(1, unit="ns")

        assert int(dut.result_write_o.value) == 1
        assert int(dut.result_addr_o.value) == address
        assert int(dut.result_data_o.value) == word

        # Simulate a result memory accepting the word
        results[address] = int(dut.result_data_o.value)

        await tick(dut)

        dut.dma_result_valid_i.value = 0
        dut.dma_result_last_i.value = 0

    # --------------------------------------------------
    # 5. Verify stored values and completion
    # --------------------------------------------------

    assert results == {0: 0xAA, 1: 0xBB}

    assert int(dut.gemv_done_o.value) == 1
    assert int(dut.busy_o.value) == 0

    await tick(dut)

    assert int(dut.gemv_done_o.value) == 0

    dut._log.info("DMA adapter unit test PASSED")