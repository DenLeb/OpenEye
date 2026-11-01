import random
import cocotb

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer


def reference_probability(exp, total):
    if total == 0 or exp == 0:
        return 0

    if exp >= total:
        return 32768

    # UQ1.15, round-to-nearest, ties up
    return (exp * 32768 + total // 2) // total


async def check_division(dut, exp, total):

    expected = reference_probability(exp, total)

    # Apply inputs before the rising clock edge
    await FallingEdge(dut.clk_i)

    dut.exp_i.value = exp
    dut.sum_i.value = total
    dut.start_i.value = 1

    # Divider receives start
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    # Remove start after one clock
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    # Wait for division to finish
    for cycle in range(20):

        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.done_o.value) == 1:
            actual = int(dut.probability_o.value)

            assert actual == expected, (
                f"exp={exp}, sum={total}: "
                f"expected={expected}, actual={actual}"
            )

            # done_o must be a one-cycle pulse
            await RisingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            assert int(dut.done_o.value) == 0
            return

    raise AssertionError(
        f"Divider timeout: exp={exp}, sum={total}"
    )


@cocotb.test()
async def test_softmax_divider(dut):

    cocotb.start_soon(
        Clock(dut.clk_i, 10, unit="ns").start()
    )

    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.exp_i.value = 0
    dut.sum_i.value = 0

    await Timer(25, unit="ns")
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # Basic and boundary cases
    cases = [
        (3, 8),
        (1, 2),
        (1, 3),
        (2, 3),
        (0, 64002),
        (32768, 32768),
        (32768, 131072),
        (13660, 64002),
        (3914, 64002),
        (32768, 64002),
        (1, 0),       # Division by zero protection
        (10, 9),      # Invalid numerator > denominator
    ]

    for exp, total in cases:
        await check_division(dut, exp, total)

    # Random 4-element exponential rows
    rng = random.Random(42)

    for _ in range(25):
        row = [rng.randint(0, 32768) for _ in range(4)]

        # Max subtraction guarantees at least one exp(0) = 1
        row[rng.randrange(4)] = 32768

        total = sum(row)

        for exp in row:
            await check_division(dut, exp, total)

    dut._log.info("All divider test cases passed")