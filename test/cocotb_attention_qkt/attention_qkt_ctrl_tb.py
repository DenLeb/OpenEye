import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def tick(dut):
    """Wait for one rising clock edge and updated RTL outputs."""
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_controller(dut):

    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.gemv_done_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())

    # Test 1: Reset
    await tick(dut)
    assert int(dut.busy_o.value) == 0
    assert int(dut.done_o.value) == 0
    assert int(dut.gemv_start_o.value) == 0

    dut.rst_ni.value = 1

    # Test 2: Start a GEMV operation
    dut.start_i.value = 1
    await tick(dut)

    assert int(dut.busy_o.value) == 1

    dut.start_i.value = 0
    await tick(dut)

    assert int(dut.gemv_start_o.value) == 1

    # Test 3: Wait for GEMV completion
    await tick(dut)
    assert int(dut.gemv_start_o.value) == 0
    assert int(dut.busy_o.value) == 1

    for _ in range(3):
        await tick(dut)
        assert int(dut.done_o.value) == 0

    # Test 4: Simulate GEMV completion
    dut.gemv_done_i.value = 1
    await tick(dut)
    dut.gemv_done_i.value = 0

    await tick(dut)

    assert int(dut.done_o.value) == 1
    assert int(dut.busy_o.value) == 0

    # Test 5: Completion pulse must end
    await tick(dut)
    assert int(dut.done_o.value) == 0

    # Test 6: Controller accepts another operation
    dut.start_i.value = 1
    await tick(dut)

    dut.start_i.value = 0
    await tick(dut)

    assert int(dut.gemv_start_o.value) == 1

    dut._log.info("Attention QKT controller FSM test PASSED")