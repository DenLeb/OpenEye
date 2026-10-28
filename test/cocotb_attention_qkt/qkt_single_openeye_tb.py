"""Cocotb: run exactly one prepared K=8/N=4 Row-Stationary QK^T GEMV.

The 677 input words are loaded from the offline image, NOT regenerated here.
The DUT is the real OpenEye_FPGA RTL (not a mocked GEMV engine).
This is an isolated numerical integration test; it does not yet prove
hardware-controlled Q/K buffer feedback.
"""

import logging
import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import SimTimeoutError, with_timeout

import open_eye.DRAM as DRAM
import open_eye.layer_execution_state as les_mod
import open_eye.layer_parameters as lp
import open_eye.open_eye_parameters as oep
import open_eye.rtl_test_utils as rtl_test_utils
import open_eye.stream_dicts as strdic
import open_eye.timing_parameters as tp
from open_eye.attention_projection_mapper import _layer_shim

SECTION_WORDS = {"status": 19, "iact": 4, "wght": 10, "psum": 4, "quantize": 640}


def load_prepared_data():
    data_dir = Path(os.environ["QKT_SINGLE_DATA_DIR"])
    dma_path = data_dir / "qkt_t0_dma.mem"
    ref_path = data_dir / "qkt_t0_reference.npz"

    words = [int(line.strip(), 16) for line in dma_path.read_text(encoding="ascii").splitlines()
             if line.strip()]
    assert len(words) == 677, f"Prepared DMA image must have 677 words, got {len(words)}"
    assert all(0 <= word < (1 << 64) for word in words)

    stream = [None] * len(SECTION_WORDS)
    offset = 0
    for name, count in SECTION_WORDS.items():
        section_index = strdic.stream_parallel_dict[name]
        stream[section_index] = words[offset:offset + count]
        offset += count
    assert offset == len(words) and all(section is not None for section in stream)

    with np.load(ref_path, allow_pickle=False) as ref:
        q8 = ref["Q8"].astype(np.int64)
        k8 = ref["K8"].astype(np.int64)
        expected = ref["expected"].astype(np.int64)
        q_shift = int(ref["q_shift"])
        k_shift = int(ref["k_shift"])

    assert q8.shape == (4, 8) and k8.shape == (4, 8)
    assert expected.shape == (4,)
    assert expected.tolist() == [8975, -5255, 18736, 8865], "Reference file changed"
    assert np.array_equal(expected, q8[0] @ k8.T)
    assert (q_shift, k_shift) == (6, 6)
    return stream, q8, k8, expected


async def run_one_qkt_gemv(dut):
    stream, q8, k8, expected = load_prepared_data()
    params = oep.get_oep(1)
    assert params.SERIAL == 1 and params.DMA_BITWIDTH == 64
    assert params.DATAFLOW == "row_stationary", params.DATAFLOW

    layer = _layer_shim(8, 4)
    layer_params = lp.LayerParameters([0], layer, params, 0, 1)
    assert layer_params.needed_total_transmissions == 1
    assert layer_params.gemm_mode == 0, "Expected Dense / Row-Stationary"

    dram = DRAM.DRAMContents([layer], [layer_params])
    dram.fmap[0] = [int(x) for x in q8[0]]
    dram.weights[0] = [[int(x) for x in row] for row in k8]
    dram.bias[0] = [0] * 4

    port_timing = tp.PortTimingParameters()
    port_timing.initiate_params(
        20, "ns", 100, "ps", 100, "ps"
    )

    cocotb.start_soon(Clock(dut.clk_i, 20, unit="ns").start())
    await rtl_test_utils.reset_all_signals(port_timing, dut, params.SERIAL)

    dut._log.info("Sending PREPARED 677-word QKT stream (not regenerating it)")
    await cocotb.start_soon(
        rtl_test_utils.send_stream(port_timing, dut, stream, params, layer_params, 0)
    )
    await cocotb.start_soon(rtl_test_utils.await_enable_signal(port_timing, dut))

    output_state = les_mod.LayerExecutionState()
    await cocotb.start_soon(
        rtl_test_utils.compare_stream_Dense(
            port_timing, dut, 0, 0, layer_params, params,
            output_state, dram, logging.INFO
        )
    )

    actual = np.array([int(dram.fmap[1][j]) for j in range(4)], dtype=np.int64)
    dut._log.info("QKT expected: %s", expected.tolist())
    dut._log.info("QKT hardware: %s", actual.tolist())
    assert np.array_equal(actual, expected), (
        f"Bit-exact QKT mismatch: expected {expected.tolist()}, actual {actual.tolist()}"
    )
    dut._log.info("PASS: Real OpenEye RTL computed one QKT GEMV K=8 N=4")


@cocotb.test()
async def test_qkt_single_gemv(dut):
    try:
        await with_timeout(run_one_qkt_gemv(dut), 10_000_000, "ns")
    except SimTimeoutError as exc:
        raise AssertionError("OpenEye did not complete single QKT GEMV within timeout") from exc
