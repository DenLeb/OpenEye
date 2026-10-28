"""Single-command RTL test of four hardware-driven QK^T GEMVs on OpenEye.

Copy beside the already passing test_qkt_single_openeye.py; run with pytest.
This test begins at the existing quantized Q8/K8 interface (NOT at Projection).
It checks the REAL OpenEye_FPGA computations, Top FSM and 4x4 score buffer.
"""

import os
import shutil
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer, ClockCycles, with_timeout, SimTimeoutError


def test_qkt_full_openeye():
    """Pytest runner; generate a fresh regmap matching the earlier passing K8/N4 test."""
    import cocotb_test.simulator
    import open_eye.generator as generator
    import open_eye.test_utils_main as ptu
    import open_eye.vh_file_creator as vh_file_creator
    from open_eye import hdl_dir, test_dir

    settings = {
        "PARALLEL_MACS": "2", "CLUSTER_ROWS": "2", "CLUSTER_COLUMNS": "2",
        "NUM_GLB_IACT": "3", "NUM_GLB_PSUM": "4", "NUM_GLB_WGHT": "3",
        "DATAFLOW": "row_stationary", "BRANCHES": "1", "BUFFER_WIDTH": "12",
        "QUANT_AMOUNT": "1024", "RAM_CELLS": "32",
    }
    os.environ.update(settings)

    here = Path(__file__).resolve().parent
    data_dir = here / "data" / "qkt_single_8x4"
    image = data_dir / "qkt_t0_dma.mem"
    reference = data_dir / "qkt_t0_reference.npz"
    for path in (image, reference):
        assert path.is_file(), f"Missing test input: {path}"

    build = Path(test_dir) / ".temp" / "qkt_full_openeye_k8_n4"
    build.mkdir(parents=True, exist_ok=True)

    # Proven method from test_qkt_single_openeye.py. In particular, do NOT
    # compile the repository's potentially stale hdl/dma_storage.v.
    vh_file_creator.create_vh_file_from_envvars(
        str(build), hdl_dir + "/", toplevel="OpenEye_FPGA"
    )
    generator.create_regmap_params_vh_file(
        os.path.join(test_dir, "cocotb_fpga"), str(build), str(build)
    )
    assert (build / "dma_storage.v").is_file(), "Regmap generator did not create dma_storage.v"
    assert (build / "regmap_params.vh").is_file(), "Missing generated regmap_params.vh"
    shutil.copyfile(image, build / "qkt_t0_dma.mem")

    sources = [
        str(p) for p in ptu.get_verilog_sources(hdl_dir)
        if Path(p).name != "dma_storage.v"
    ]
    extra_modules = (
        "attention_qkt_openeye_top.v",
        "attention_qkt_q8_packer.v",
        "attention_qkt_k8_packer.v",
        "attention_qkt_dma_memory.v",
        "attention_qkt_score_buffer.v",
        "attention_dma_adapter.v",
    )
    existing = {str(Path(src).resolve()) for src in sources}
    for name in extra_modules:
        file = Path(hdl_dir) / name
        assert file.is_file(), f"Missing RTL: {file}"
        if str(file.resolve()) not in existing:
            sources.append(str(file))
    sources.append(str(build / "dma_storage.v"))

    cocotb_test.simulator.run(
        python_search=[str(here)],
        verilog_sources=sources,
        toplevel="attention_qkt_openeye_top",
        module="test_qkt_full_openeye",
        testcase="qkt_four_gemv",
        sim_build=str(build),
        includes=[str(build), str(Path(hdl_dir)), str(Path(hdl_dir) / "include")],
        defines={"NO_TRACE": "TRUE", "USE_INTERNAL_PARAMS_PE": "TRUE"},
        simulator="icarus",
        force_compile=True,
        waves=False,
        extra_env={
            **settings,
            "QKT_FULL_DATA_DIR": str(data_dir.resolve()),
            "CLOCK_LEN": "20", "CLOCK_UNIT": "ns",
            "CLOCK_DELAY_INPUT": "100", "CLOCK_DELAY_UNIT_INPUT": "ps",
            "CLOCK_DELAY_OUTPUT": "100", "CLOCK_DELAY_UNIT_OUTPUT": "ps",
            "LOGGER_LEVEL": "20",
        },
    )


async def _drive_qk_memory_read_ports(dut, q8, k8):
    """Cocotb stand-in for existing asynchronous Q8/K8 memories.

    No Python computation or stream scheduling is used at runtime: DUT packers,
    DMA adapter, OpenEye and FSM perform the entire four-GEMV calculation.
    """
    dut.q8_data_i.value = int(q8[0, 0]) & 0xff
    dut.k8_data_i.value = int(k8[0, 0]) & 0xff
    while True:
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")  # allow nonblocking address updates to settle
        qt = int(dut.q8_rd_token_o.value)
        qf = int(dut.q8_rd_feature_o.value)
        kt = int(dut.k8_rd_token_o.value)
        kf = int(dut.k8_rd_feature_o.value)
        assert 0 <= qt < 4 and 0 <= qf < 8, f"Bad Q8 address [{qt},{qf}]"
        assert 0 <= kt < 4 and 0 <= kf < 8, f"Bad K8 address [{kt},{kf}]"
        dut.q8_data_i.value = int(q8[qt, qf]) & 0xff
        dut.k8_data_i.value = int(k8[kt, kf]) & 0xff


@cocotb.test()
async def qkt_four_gemv(dut):
    data_dir = Path(os.environ["QKT_FULL_DATA_DIR"])
    with np.load(data_dir / "qkt_t0_reference.npz", allow_pickle=False) as d:
        q8 = d["Q8"].astype(np.int64)
        k8 = d["K8"].astype(np.int64)
        t0_golden = d["expected"].astype(np.int64)

    assert q8.shape == (4, 8) and k8.shape == (4, 8)
    expected = q8 @ k8.T
    assert expected.shape == (4, 4)
    assert np.array_equal(expected[0], t0_golden), "Reference input and golden disagree"

    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.q8_data_i.value = 0
    dut.k8_data_i.value = 0
    dut.score_rd_token_i.value = 0
    dut.score_rd_index_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 20, unit="ns").start())
    memory_task = cocotb.start_soon(_drive_qk_memory_read_ports(dut, q8, k8))
    await ClockCycles(dut.clk_i, 10)
    await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await ClockCycles(dut.clk_i, 20)  # OpenEye's reset synchronizer

    dut._log.info("Start: four QKT GEMVs through actual OpenEye RTL")
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 1
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    try:
        await with_timeout(RisingEdge(dut.done_o), 10_000_000, "ns")
    except SimTimeoutError as e:
        raise AssertionError(
            "QKT Top timed out (check its FSM, DMA adapter and OpenEye output-last). "
            f"busy={dut.busy_o.value}, token={dut.active_token_o.value}, "
            f"stream_addr={dut.stream_addr_o.value}"
        ) from e

    await FallingEdge(dut.clk_i)
    assert int(dut.scores_complete_o.value) == 1, "Top DONE but 16 score entries are not all valid"

    actual = np.zeros((4, 4), dtype=np.int64)
    for token in range(4):
        for score in range(4):
            dut.score_rd_token_i.value = token
            dut.score_rd_index_i.value = score
            await Timer(1, unit="ns")
            assert int(dut.score_rd_valid_o.value) == 1, (
                f"Score buffer missing S[{token},{score}]"
            )
            raw = int(dut.score_rd_data_o.value)
            actual[token, score] = raw - (1 << 20) if raw & (1 << 19) else raw

    dut._log.info("Expected S:\n%s", expected)
    dut._log.info("Hardware S:\n%s", actual)
    assert np.array_equal(actual, expected), (
        f"Bit-exact S mismatch\nExpected:\n{expected}\nActual:\n{actual}"
    )
    memory_task.cancel()
    dut._log.info("PASS: 4 real OpenEye GEMVs, all 16 QKT scores bit-exact")
