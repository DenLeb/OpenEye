"""Characterization of the unchanged OpenEye quantization_unit.v.

Expected behavior: signed (psum + offset) * mantissa, arithmetic shift,
then 8-bit truncation (NOT rounding or saturation), with 3-stage latency.
"""

import cocotb
from cocotb.triggers import Timer


def signed_bits(bits: int, width: int) -> int:
    bits &= (1 << width) - 1
    return bits - (1 << width) if bits & (1 << (width - 1)) else bits


def reference(psum: int, mant: int, offset: int, shift: int) -> int:
    # Python's right shift of signed integers rounds toward -infinity,
    # like Verilog's arithmetic >>> on a signed operand.
    shifted = (mant * (psum + offset)) >> shift
    return signed_bits(shifted, 8)  # Actual RTL truncates to 8 bits.


async def tick(dut, *, psum=0, mant=1, offset=0, shift=0):
    """One complete clock cycle, safely sampling outputs after the edge."""
    dut.clk_i.value = 0
    dut.psum_data.value = psum & ((1 << 20) - 1)
    dut.quant_mant.value = mant & ((1 << 25) - 1)
    dut.quant_offset.value = offset & 0xFF
    dut.current_shift.value = shift
    await Timer(5, unit="ns")
    dut.clk_i.value = 1
    await Timer(5, unit="ns")
    return signed_bits(int(dut.quantized_value.value), 8)


@cocotb.test()
async def test_existing_quantizer_pipeline(dut):
    """Verify sign handling, scale, zeroing, wraparound, and 3-cycle pipeline."""
    dut.clk_i.value = 0
    dut.rst_n.value = 0
    dut.psum_data.value = 0
    dut.quant_mant.value = 0
    dut.quant_offset.value = 0
    dut.current_shift.value = 0
    await tick(dut)
    assert int(dut.quantized_value.value) == 0, "Reset did not clear output"
    dut.rst_n.value = 1

    # (signed 20-bit PSUM, signed 25-bit mantissa, signed 8-bit offset, shift)
    vectors = [
        (0, 1, 0, 0),        # Zero stays zero
        (3, 1, 0, 3),        # Small positive becomes ZERO
        (-2, 1, 0, 3),       # Arithmetic shift: -2 >>> 3 becomes -1
        (700, 1, 0, 3),      # 87 (truncation, not round-to-nearest 88)
        (-700, 1, 0, 3),     # -88
        (12, 5, -2, 1),     # Mantissa and signed offset: 25
        (200, 1, 0, 0),      # 200 wraps to -56 (no saturation)
        (-200, 1, 0, 0),     # -200 wraps to +56
        (127, 1, 0, 0),      # INT8 upper endpoint
        (-128, 1, 0, 0),     # INT8 lower endpoint
        (400, 2, 0, 4),      # More than one mantissa factor
        (-400, 2, 0, 4),
        (32, -3, 0, 3),      # Negative mantissa
        (524287, 1, 0, 13), # Max signed 20-bit PSUM
        (-524288, 1, 0, 13),# Min signed 20-bit PSUM
        (1, 1, 0, 7),        # Positive quantization-to-zero
        (-1, 1, 0, 7),       # Negative floors to -1
        (0, 1, 5, 0),        # Offset is added before scaling
        (0, 1, -5, 0),
        (700, 3, 0, 5),      # Fractional multiplier 3/32
        (-700, 3, 0, 5),
    ]

    # No valid_out in RTL, so associate outputs with the inputs from
    # two rising edges earlier (3-stage pipeline: S1/S2/S3).
    for cycle in range(len(vectors) + 2):
        sample = vectors[cycle] if cycle < len(vectors) else (0, 1, 0, 0)
        actual = await tick(dut, psum=sample[0], mant=sample[1],
                            offset=sample[2], shift=sample[3])
        if cycle >= 2:
            original_index = cycle - 2
            expected = reference(*vectors[original_index])
            assert actual == expected, (
                f"Mismatch for input #{original_index} "
                f"{vectors[original_index]}: expected {expected}, got {actual}"
            )

    dut._log.info("PASS: all %d signed/pipeline quantizer cases", len(vectors))
    dut._log.info("Confirmed: arithmetic shift, no rounding, no saturation")
