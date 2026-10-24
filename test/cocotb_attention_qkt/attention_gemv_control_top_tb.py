import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def tick(dut):
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_gemv_integration(dut):

    # ==================================================
    # 1. Initialize signals
    # ==================================================

    dut.clk_i.value = 0
    dut.rst_ni.value = 0

    dut.start_i.value = 0

    dut.stream_word_count_i.value = 3
    dut.stream_data_i.value = 0

    dut.dma_ready_i.value = 0

    dut.dma_result_data_i.value = 0
    dut.dma_result_valid_i.value = 0
    dut.dma_result_last_i.value = 0

    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    await tick(dut)

    # Release reset
    dut.rst_ni.value = 1
    await tick(dut)

    assert int(dut.busy_o.value) == 0
    assert int(dut.adapter_busy_o.value) == 0

    # ==================================================
    # 2. Start the main controller
    # ==================================================

    dut.start_i.value = 1
    await tick(dut)
    dut.start_i.value = 0

    assert int(dut.busy_o.value) == 1

    # Controller generates GEMV start pulse
    await tick(dut)

    # Adapter receives the pulse on the next edge
    await tick(dut)

    assert int(dut.adapter_busy_o.value) == 1

    # ==================================================
    # 3. Test WAIT_READY
    # ==================================================

    # OpenEye is not ready yet
    for _ in range(3):
        await tick(dut)

        assert int(dut.dma_valid_o.value) == 0
        assert int(dut.adapter_busy_o.value) == 1

    # Simulate OpenEye becoming ready
    dut.dma_ready_i.value = 1
    await tick(dut)

    # ==================================================
    # 4. Test DMA input transmission
    # ==================================================

    input_words = [0x11, 0x22, 0x33]

    for address, word in enumerate(input_words):

        # Simulated combinational word source
        dut.stream_data_i.value = word
        await Timer(1, unit="ns")

        assert int(dut.stream_addr_o.value) == address
        assert int(dut.dma_valid_o.value) == 1
        assert int(dut.dma_data_o.value) == word

        await tick(dut)

    assert int(dut.dma_valid_o.value) == 0
    assert int(dut.dma_result_ready_o.value) == 1

    # Main controller must still be waiting
    assert int(dut.busy_o.value) == 1

    # ==================================================
    # 5. Simulate results from OpenEye
    # ==================================================

    received = {}

    for address, word in enumerate([0xAA, 0xBB]):

        dut.dma_result_data_i.value = word
        dut.dma_result_valid_i.value = 1
        dut.dma_result_last_i.value = int(address == 1)

        await Timer(1, unit="ns")

        assert int(dut.result_write_o.value) == 1
        assert int(dut.result_addr_o.value) == address
        assert int(dut.result_data_o.value) == word

        received[address] = int(dut.result_data_o.value)

        await tick(dut)

        dut.dma_result_valid_i.value = 0
        dut.dma_result_last_i.value = 0

    assert received == {0: 0xAA, 1: 0xBB}

    # Adapter has completed, but the main controller
    # has not yet processed the completion signal.
    assert int(dut.adapter_busy_o.value) == 0
    assert int(dut.busy_o.value) == 1
    assert int(dut.done_o.value) == 0

    # ==================================================
    # 6. Verify main controller completion
    # ==================================================

    # Main controller detects the adapter's done pulse
    await tick(dut)

    assert int(dut.done_o.value) == 0

    # Main controller executes its DONE state
    await tick(dut)

    assert int(dut.done_o.value) == 1
    assert int(dut.busy_o.value) == 0
    assert int(dut.adapter_busy_o.value) == 0

    # Completion must last exactly one clock
    await tick(dut)

    assert int(dut.done_o.value) == 0

    dut._log.info("GEMV controller + DMA adapter integration PASSED")