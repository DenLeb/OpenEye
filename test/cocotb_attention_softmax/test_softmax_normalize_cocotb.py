import random
import cocotb

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


def reference_probability(exp, total):
    if total == 0 or exp == 0:
        return 0

    if exp >= total:
        return 32768

    return (exp * 32768 + total // 2) // total


async def check_row(dut, exp_values):
    total = sum(exp_values)

    expected = [
        reference_probability(exp, total)
        for exp in exp_values
    ]

    # Pack four 16-bit exponential values into 64 bits
    packed_input = sum(
        int(exp) << (16 * i)
        for i, exp in enumerate(exp_values)
    )

    await FallingEdge(dut.clk_i)

    dut.exp_values_i.value = packed_input
    dut.start_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    assert int(dut.busy_o.value) == 1, (
        "Normalizer did not start"
    )

    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    # Wait for all four dividers
    for cycle in range(30):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.done_o.value) == 1:

            actual_packed = int(dut.probabilities_o.value)

            actual = [
                (actual_packed >> (16 * i)) & 0xFFFF
                for i in range(4)
            ]

            assert actual == expected, (
                f"Input: {exp_values}\n"
                f"Sum: {total}\n"
                f"Expected: {expected}\n"
                f"Actual: {actual}"
            )

            assert int(dut.busy_o.value) == 0

            # Verify one-cycle done pulse
            await RisingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            assert int(dut.done_o.value) == 0

            return

    raise AssertionError(
        f"Normalization timeout: {exp_values}"
    )


@cocotb.test()
async def test_softmax_normalize(dut):

    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.exp_values_i.value = 0

    await Timer(25, unit="ns")
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # Real data from our first Softmax row
    await check_row(
        dut, [13660, 3914, 32768, 13660]
    )

    # Boundary and special cases
    cases = [
        [32768, 0, 0, 0],
        [0, 32768, 0, 0],
        [32768, 32768, 0, 0],
        [32768, 32768, 32768, 32768],
        [32768, 0, 2000, 5000],
        [32768, 1, 1, 1],
        [0, 0, 0, 0],
    ]

    for row in cases:
        await check_row(dut, row)

    # Random rows
    rng = random.Random(42)

    for _ in range(100):
        row = [
            rng.randint(0, 32768)
            for _ in range(4)
        ]

        # At least one exponential must equal exp(0)
        row[rng.randrange(4)] = 32768

        await check_row(dut, row)

    dut._log.info(
        "All 108 normalization test cases passed"
    )