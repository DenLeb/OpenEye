import cocotb

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.utils import get_sim_time

@cocotb.test()
async def test_pe_matrix(dut):

    # ---------------------------------
    # 1. Define our matrices
    # ---------------------------------

    A = [
        [1, 2],
        [3, 4]
    ]

    W = [
        [5, 6, 7,8],
        [9,10,11,12]
    ]

    # Expected output matrix
    Y = [
        [0, 0 ,0, 0],
        [0, 0, 0, 0]
    ]

    # Calculate the expected result
    for row in range(2):
        for col in range(len(W[row])):
            for k in range(2):

                Y[row][col] += A[row][k] * W[k][col]

    assert Y == [[23, 26, 29, 32], [51, 58, 65, 72]]

    dut._log.info(f"Expected matrix: {Y}")

    # ---------------------------------
    # 2. Initialize all inputs
    # ---------------------------------

    dut.clk_i.value = 0
    dut.rst_ni.value = 0

    inputs = [
        "iact_select_i",
        "iact_data_i",
        "iact_enable_i",
        "wght_data_i",
        "wght_enable_i",
        "psum_data_i",
        "psum_enable_i",
        "psum_ready_i",
        "compute_i",
        "enable_stream_i",
        "data_stream_i",
        "iact_pass_data_i",
        "iact_pass_enable_i",
        "iact_pass_ready_i",
    ]

    for name in inputs:
        getattr(dut, name).value = 0

    # ---------------------------------
    # 3. Start clock
    # ---------------------------------

    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    # ---------------------------------
    # 4. Reset PE
    # ---------------------------------

    for _ in range(3):
        await RisingEdge(dut.clk_i)

    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # ---------------------------------
    # 5. Send configuration
    # ---------------------------------

    # One row has two activations.
    # C0 = 1, B = 2, M0 = 4

    for value in [2, 1, 4]:

        await FallingEdge(dut.clk_i)

        dut.enable_stream_i.value = 1
        dut.data_stream_i.value = value

        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    await FallingEdge(dut.clk_i)

    dut.enable_stream_i.value = 0
    dut.data_stream_i.value = 0

    await RisingEdge(dut.clk_i)  
    await Timer(1, unit="ns")

    # ---------------------------------
    # 6. Check configuration
    # ---------------------------------

    assert int(dut.iact_addr_max_reg.value) == 2
    assert int(dut.channel_reg_C0.value) == 1
    assert int(dut.filters_reg_M0.value) == 4

    dut._log.info("PASS: Matrix definition, reset and configuration")

    # ---------------------------------
    # 7. Send first row of Input activations
    # ---------------------------------

    # Select activation input lane 0
    dut.iact_select_i.value = 0    

    dut.iact_data_i.value = (2 << 8) | 1

    # Enable activation lane 0
    dut.iact_enable_i.value = 0b001

    # PE extracts activation 1
    # -------------------------------------------------
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    # PE writes 1 into SPad
    # -------------------------------------------------
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    # ---------------------------------
    # 8. Stop sending activation data
    # ---------------------------------

    await FallingEdge(dut.clk_i)

    dut.iact_enable_i.value = 0
    dut.iact_data_i.value = 0

    # ---------------------------------
    # 9. Check First activation is loaded into SPad
    # ---------------------------------

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    dut._log.info("Activation row [1, 2] loaded")

    # ---------------------------------
    # 10. Send weights - PARALLEL_MACS = 2
    # ---------------------------------

    #activation 1:
    #filter group 0 → [5, 6]
    weight_word = (6 << 8) | 5

    await FallingEdge(dut.clk_i)
    dut.wght_data_i.value = weight_word
    dut.wght_enable_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    #filter group 1 → [7, 8]
    weight_word = (8 << 8) | 7

    await FallingEdge(dut.clk_i)
    dut.wght_data_i.value = weight_word

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    #activation 2:
    #filter group 0 → [9, 10]
    weight_word = (10 << 8) | 9
    
    await FallingEdge(dut.clk_i)
    dut.wght_data_i.value = weight_word
    dut.wght_enable_i.value = 1
    
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    #filter group 1 → [11, 12]
    weight_word = (12 << 8) | 11
    
    await FallingEdge(dut.clk_i)
    dut.wght_data_i.value = weight_word
    dut.wght_enable_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")
    # ---------------------------------
    # 11. Allow the registered SPad writes to complete.
    # ---------------------------------

    await FallingEdge(dut.clk_i)

    dut.wght_enable_i.value = 0
    dut.wght_data_i.value = 0

    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)

    await Timer(1, unit="ns")

    dut._log.info("Weight loading sequence completed")

    # ---------------------------------
    # 12. Computation.
    # ---------------------------------

    # Check that the PE is ready
    assert int(dut.iact_set.value) == 1
    assert int(dut.wght_set.value) == 1

    await FallingEdge(dut.clk_i)

    dut.compute_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    assert int(dut.current_state_computing.value) == 1

    await FallingEdge(dut.clk_i)

    dut.compute_i.value = 0

    # ---------------------------------
    # 13. Psum ready and data
    # ---------------------------------
    
    # Wait until the PE finishes computing
    for _ in range(30):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.current_state_computing.value) == 7:
            break

    assert int(dut.current_state_computing.value) == 7

    dut._log.info("PE computation completed")

    # ---------------------------------
    # 14. PSUM handshake
    # ---------------------------------

    dut.psum_ready_i.value = 1

    await RisingEdge(dut.psum_ready_o)
    await Timer(10, unit="ns")

    async def send_zero_biases (): # M0 = 4 -> send four zero biases
        for _ in range(4):

            await FallingEdge(dut.clk_i)

            dut.psum_data_i.value = 0
            dut.psum_enable_i.value = 1

            await RisingEdge(dut.clk_i)

        await FallingEdge(dut.clk_i)

        dut.psum_enable_i.value = 0
        dut.psum_data_i.value = 0

    # Start sending biases in parallel
    bias_task = cocotb.start_soon(send_zero_biases())

    await RisingEdge(dut.psum_enable_o)
    await Timer(1, unit="ns")

    # ---------------------------------
    # 15. Read outputs
    # ---------------------------------

    outputs = []

    while int(dut.psum_enable_o.value) == 1:

        data = dut.psum_data_o.value.to_signed()

        outputs.append(data)

        dut._log.info(f"PE output: {data}")
        await Timer(10, unit="ns")

    dut._log.info(f"Final outputs: {outputs}")

    assert outputs == [23, 26, 29, 32], \
        f"Expected [23, 26, 29, 32], got {outputs}"