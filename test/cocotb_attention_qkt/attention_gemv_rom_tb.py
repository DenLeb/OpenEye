from pathlib import Path

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def test_gemv_rom(dut):

    # Read the original reference file
    test_dir = Path(__file__).resolve().parent
    mem_file = test_dir / "data" / "gemv_reference.mem"

    words = [
        int(line.strip(), 16)
        for line in mem_file.read_text().splitlines()
        if line.strip()
    ]

    assert len(words) == 939

    # Verify every ROM address
    for address, expected_word in enumerate(words):

        dut.addr_i.value = address

        await Timer(1, unit="ns")

        actual_word = int(dut.data_o.value)

        assert actual_word == expected_word, (
            f"ROM mismatch at address {address}: "
            f"expected {expected_word:016X}, "
            f"got {actual_word:016X}"
        )

    # Verify an out-of-range address returns zero
    dut.addr_i.value = len(words)
    await Timer(1, unit="ns")

    assert int(dut.data_o.value) == 0

    dut._log.info("ROM verification PASSED: 939 words checked")