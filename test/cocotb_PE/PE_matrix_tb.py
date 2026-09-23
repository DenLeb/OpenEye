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
        [5, 6],
        [7, 8]
    ]

    # Expected output matrix
    Y = [
        [0, 0],
        [0, 0]
    ]

    # Calculate the expected result
    for row in range(2):
        for col in range(2):
            for k in range(2):

                Y[row][col] += A[row][k] * W[k][col]

    assert Y == [[19, 22], [43, 50]]

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
    # C0 = 1, B = 2, M0 = 2

    for value in [2, 1, 2]:

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
    assert int(dut.filters_reg_M0.value) == 2

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
    # 10. send the weight
    # ---------------------------------

    weights = [5, 6, 7, 8]

    for weight in weights:

        await FallingEdge(dut.clk_i)

        dut.wght_data_i.value = weight
        dut.wght_enable_i.value = 1

        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    await FallingEdge(dut.clk_i)

    dut.wght_enable_i.value = 0
    dut.wght_data_i.value = 0

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    dut._log.info("Weight loading sequence completed")

    #---------------------------------
    ## Send the first weight row: [5, 7]
    #weight_word = (W[0][1] << 8) | W[0][0]

    #await FallingEdge(dut.clk_i)

    #dut.wght_data_i.value = weight_word
    #dut.wght_enable_i.value = 1

    #await RisingEdge(dut.clk_i)
    #await Timer(1, unit="ns")

    ## Send the second weight row: [6, 8]
    #await FallingEdge(dut.clk_i)

    #weight_word = (W[1][1] << 8) | W[1][0]

    #dut.wght_data_i.value = weight_word

    #await RisingEdge(dut.clk_i)
    #await Timer(1, unit="ns")
    #---------------------------------

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

    # M0 = 2 -> send two zero biases
    for _ in range(2):

        await FallingEdge(dut.clk_i)

        dut.psum_data_i.value = 0
        dut.psum_enable_i.value = 1

        await RisingEdge(dut.clk_i)

    await FallingEdge(dut.clk_i)

    dut.psum_enable_i.value = 0
    dut.psum_data_i.value = 0

    # ---------------------------------
    # 15. Read outputs
    # ---------------------------------

    await RisingEdge(dut.psum_enable_o)
    await Timer(10, unit="ns")

    outputs = []

    while int(dut.psum_enable_o.value) == 1:

        data = dut.psum_data_o.value.to_signed()

        outputs.append(data)

        dut._log.info(f"PE output: {data}")
        await Timer(10, unit="ns")

    dut._log.info(f"Final outputs: {outputs}")

    assert outputs == [19, 22], \
        f"Expected [19, 22], got {outputs}"