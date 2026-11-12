"""RTL end-to-end test: 12 Q/K/V Projection GEMVs -> INT8 Q/K -> 4 QK^T GEMVs.

Run from the OpenEye repository root after copying this file into
  test/cocotb_attention_qkt/test_attention_top_e2e.py

    python -m pytest test/cocotb_attention_qkt/test_attention_top_e2e.py -v -s

Only the pre-existing .mem files are loaded by $readmemh. Python DOES NOT
orchestrate any of the 16 GEMVs or inject Q8/K8 into the DUT. The DUT's
attention_top_scheduler, DMA adapter, OpenEye FPGA, quantizer and packers do
all of that. Python is used for the independent arithmetic golden reference.
"""

import os
import shutil
import pytest
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer, with_timeout, SimTimeoutError


@pytest.mark.parametrize(
    "stage, test_case",
    [
        (4, "check_complete_attention_dataflow"),
        (1, "check_stage1_projection"),
    ],
)
def test_attention_top_e2e(stage, test_case):
    import cocotb_test.simulator
    import open_eye.test_utils_main as ptu
    import open_eye.vh_file_creator as vh_file_creator
    import open_eye.generator as generator
    from open_eye import hdl_dir, test_dir

    here = Path(__file__).resolve().parent
    proj_dir = here / "data" / "attention_mapper_v1"
    qkt_dir = here / "data" / "qkt_single_8x4"
    files = {
        "common_rom.mem": proj_dir / "common_rom.mem",
        "activation_ram.mem": proj_dir / "activation_ram.mem",
        "weight_ram.mem": proj_dir / "weight_ram.mem",
        "qkt_t0_dma.mem": qkt_dir / "qkt_t0_dma.mem",
    }
    for name, src in files.items():
        assert src.is_file(), f"Missing {name}: {src}"
    assert (proj_dir / "inputs_used.npz").is_file(), (
        f"Missing reference tensors: {proj_dir / 'inputs_used.npz'}"
    )
    # Validate the memory-image geometry expected by the current RTL.
    for name, count in {
        "common_rom.mem": 691,
        "activation_ram.mem": 32,
        "weight_ram.mem": 720,
        "qkt_t0_dma.mem": 677,
    }.items():
        actual = sum(bool(line.strip()) for line in files[name].read_text().splitlines())
        assert actual == count, f"{name}: expected {count} lines, got {actual}"

    # Both passing row-stationary tests use the same OpenEye configuration.
    settings = {
        "PARALLEL_MACS": "2",
        "CLUSTER_ROWS": "2", "CLUSTER_COLUMNS": "2",
        "NUM_GLB_IACT": "3", "NUM_GLB_PSUM": "4", "NUM_GLB_WGHT": "3",
        "DATAFLOW": "row_stationary", "BRANCHES": "1", "BUFFER_WIDTH": "12",
        "QUANT_AMOUNT": "1024", "RAM_CELLS": "32",
    }
    os.environ.update(settings)

    build = (Path(test_dir)/ ".temp"/ f"attention_top_e2e_k8_n4_stage{stage}")
    build.mkdir(parents=True, exist_ok=True)
    vh_file_creator.create_vh_file_from_envvars(
        str(build), str(hdl_dir) + os.sep, toplevel="OpenEye_FPGA"
    )
    generator.create_regmap_params_vh_file(
        str(Path(test_dir) / "cocotb_fpga"), str(build), str(build)
    )
    for name, source in files.items():
        shutil.copyfile(source, build / name)
    generated = build / "dma_storage.v"
    assert generated.is_file(), "Regmap generation did not produce dma_storage.v"

    sources = [
        str(p) for p in ptu.get_verilog_sources(str(hdl_dir))
        if Path(p).name != "dma_storage.v"
    ]
    for name in (
        "attention_top.v",
        "attention_dma_adapter.v",
        "attention_projection_memory.v",
        "attention_projection_result_buffer.v",
        "attention_qk_quantization_controller.v",
        "attention_qkt_q8_packer.v",
        "attention_qkt_k8_packer.v",
        "attention_qkt_dma_memory.v",
        "attention_qkt_score_buffer.v",
        "attention_softmax_exp.v",
        "attention_softmax_divider.v",
        "attention_softmax_normalize.v",
        "attention_softmax_row.v",
        "attention_softmax.v",
    ):
        path = Path(hdl_dir) / name
        assert path.is_file(), f"Missing RTL module: {path}"
        if str(path.resolve()) not in {str(Path(src).resolve()) for src in sources}:
            sources.append(str(path))
    sources.append(str(generated))

    cocotb_test.simulator.run(
        python_search=[str(here), str(here.parent / "cocotb_attention_softmax")],
        verilog_sources=sources,
        toplevel="attention_top",
        module="test_attention_top_e2e",
        parameters={"INTEGRATION_STAGE": stage},
        sim_build=str(build),
        includes=[str(build), str(hdl_dir), str(Path(hdl_dir) / "include")],
        defines={"NO_TRACE": "TRUE", "USE_INTERNAL_PARAMS_PE": "TRUE"},
        simulator="icarus",
        force_compile=True,
        waves=False,
        extra_env={
            **settings,
            "ATTENTION_E2E_PROJECTION_DIR": str(proj_dir.resolve()),
            "COCOTB_TEST_FILTER": f"{test_case}$",
        },
    )

async def read_observe(dut, select, token, feature):
    dut.observe_req_i.value = 0

    # Set the requested address and selector
    dut.observe_select_i.value = select
    dut.observe_token_i.value = token
    dut.observe_feature_i.value = feature

    # Wait for the interface to become ready
    for _ in range(100):
        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.observe_ready_o.value):
            break
    else:
        raise AssertionError("Observation interface not ready")

    # Submit one read request
    dut.observe_req_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    immediate = int(dut.observe_valid_o.value)
    data = int(dut.observe_data_o.value) if immediate else None

    await FallingEdge(dut.clk_i)
    dut.observe_req_i.value = 0

    if immediate:
        return data

    # Wait for multi-cycle PSUM response
    await with_timeout(
        RisingEdge(dut.observe_valid_o),
        2000,
        "ns"
    )

    await Timer(1, unit="ns")

    return int(dut.observe_data_o.value)


@cocotb.test()
async def check_stage1_projection(dut):
    projection_dir = Path(
        os.environ["ATTENTION_E2E_PROJECTION_DIR"]
    )

    # Independent mathematical reference
    with np.load(
        projection_dir / "inputs_used.npz",
        allow_pickle=False
    ) as data:
        x = data["X"].astype(np.int64)
        wq = data["WQ"].astype(np.int64)
        wk = data["WK"].astype(np.int64)
        wv = data["WV"].astype(np.int64)

    golden = [
        x @ wq,
        x @ wk,
        x @ wv,
    ]

    for matrix in golden:
        assert matrix.shape == (4, 8)

    # Initialize all top-level inputs
    dut.rst_ni.value = 0
    dut.start_i.value = 0

    dut.iact_we_i.value = 0
    dut.iact_wr_addr_i.value = 0
    dut.iact_wr_data_i.value = 0

    dut.wght_we_i.value = 0
    dut.wght_wr_addr_i.value = 0
    dut.wght_wr_data_i.value = 0

    dut.observe_select_i.value = 0
    dut.observe_token_i.value = 0
    dut.observe_feature_i.value = 0
    dut.observe_req_i.value = 0

    dut.score_rd_token_i.value = 0
    dut.score_rd_index_i.value = 0

    dut.prob_rd_token_i.value = 0
    dut.prob_rd_index_i.value = 0

    cocotb.start_soon(
        Clock(dut.clk_i, 20, unit="ns").start()
    )

    for _ in range(12):
        await FallingEdge(dut.clk_i)

    dut.rst_ni.value = 1

    for _ in range(20):
        await FallingEdge(dut.clk_i)

    ready = str(dut.observe_ready_o.value)

    assert ready in ("0", "1"), (
        f"Observation ready is unresolved before execution: {ready}"
    )
    # Monitor actual DMA results
    projection_beats = 0
    seen_projections = set()
    qkt_beats = 0

    async def monitor():
        nonlocal projection_beats, qkt_beats

        while True:
            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            if int(dut.dma_result_write_o.value):
                mode = int(dut.qkt_mode_o.value)

                if mode == 0:
                    projection_beats += 1
                    seen_projections.add((
                        int(dut.active_projection_o.value),
                        int(dut.active_token_o.value),
                    ))
                else:
                    qkt_beats += 1

    monitor_task = cocotb.start_soon(monitor())

    dut._log.info("START: INTEGRATION_STAGE=1")

    # Start one complete Projection sequence
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 1

    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    try:
        await with_timeout(
            RisingEdge(dut.done_o),
            15_000_000,
            "ns"
        )
    except SimTimeoutError as exc:
        raise AssertionError(
            "Stage 1 timeout: "
            f"beats={projection_beats}, "
            f"seen={seen_projections}, "
            f"token={dut.active_token_o.value}, "
            f"projection={dut.active_projection_o.value}"
        ) from exc

    await FallingEdge(dut.clk_i)
    monitor_task.cancel()

    # Verify Scheduler execution
    expected_projections = {
        (projection, token)
        for projection in range(3)
        for token in range(4)
    }

    assert seen_projections == expected_projections, (
        f"Missing projections: "
        f"{expected_projections - seen_projections}"
    )

    assert projection_beats == 384, (
        f"Expected 384 DMA beats, got {projection_beats}"
    )

    assert qkt_beats == 0, (
        f"QKT must not execute in Stage 1: {qkt_beats}"
    )

    assert int(dut.busy_o.value) == 0
    assert int(dut.qkt_mode_o.value) == 0

    # Check disabled stages
    assert int(dut.scores_complete_o.value) == 0
    assert int(dut.score_rd_valid_o.value) == 0

    assert int(dut.softmax_complete_o.value) == 0
    assert int(dut.prob_rd_valid_o.value) == 0

    dut._log.info(
        "PSUM DEBUG: busy=%s adapter_busy=%s "
        "observe_ready=%s pending=%s core_ready=%s "
        "attention_active=%s",
        dut.busy_o.value,
        dut.adapter_busy.value,
        dut.observe_ready_o.value,
        dut.observe_pending.value,
        dut.core_psum_rd_ready.value,
        dut.u_openeye.psum_attention_active.value,
    )

    # Read and compare all 96 Q/K/V values
    def signed20(raw):
        raw = int(raw)
        return raw - (1 << 20) if raw & (1 << 19) else raw

    names = ["Q", "K", "V"]

    for select in range(3):
        actual = np.zeros((4, 8), dtype=np.int64)

        dut.observe_select_i.value = select

        for token in range(4):
            for feature in range(8):
                dut.observe_token_i.value = token
                dut.observe_feature_i.value = feature

                raw = await read_observe(
                    dut,
                    select,
                    token,
                    feature
                )

                actual[token, feature] = signed20(raw)

        dut._log.info(
            "%s expected:\n%s",
            names[select],
            golden[select]
        )

        dut._log.info(
            "%s actual:\n%s",
            names[select],
            actual
        )

        assert np.array_equal(actual, golden[select]), (
            f"{names[select]} projection mismatch\n"
            f"Expected:\n{golden[select]}\n"
            f"Actual:\n{actual}"
        )

    dut._log.info(
        "PASS Stage 1: 12 GEMVs, 384 DMA beats, "
        "all 96 Q/K/V values bit-exact, "
        "no QKT or Softmax execution"
    )

def _reference_quantize(values):
    """Bit-exact per-tensor pow2 rounding from passing Q/K quantization TB."""
    values = np.asarray(values, dtype=np.int64)
    m = int(np.max(np.abs(values)))
    shift = 0
    while m > (127 << shift):
        shift += 1
    if shift:
        half = 1 << (shift - 1)
        values = np.where(
            values >= 0,
            (values + half) >> shift,
            -((-values + half) >> shift),
        )
    return np.clip(values, -128, 127).astype(np.int64), shift


@cocotb.test()
async def check_complete_attention_dataflow(dut):
    from softmax_exp_reference import exp_row
    projection_dir = Path(os.environ["ATTENTION_E2E_PROJECTION_DIR"])
    with np.load(projection_dir / "inputs_used.npz", allow_pickle=False) as t:
        x = t["X"].astype(np.int64)
        wq = t["WQ"].astype(np.int64)
        wk = t["WK"].astype(np.int64)
        wv = t["WV"].astype(np.int64)
    q_raw, k_raw, v_raw = x @ wq, x @ wk, x @ wv
    assert q_raw.shape == (4, 8) and k_raw.shape == (4, 8)
    assert v_raw.shape == (4, 8)
    q8, q_shift = _reference_quantize(q_raw)
    k8, k_shift = _reference_quantize(k_raw)
    golden_scores = q8 @ k8.T

    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.iact_we_i.value = 0
    dut.iact_wr_addr_i.value = 0
    dut.iact_wr_data_i.value = 0
    dut.wght_we_i.value = 0
    dut.wght_wr_addr_i.value = 0
    dut.wght_wr_data_i.value = 0
    dut.score_rd_token_i.value = 0
    dut.score_rd_index_i.value = 0
    dut.prob_rd_token_i.value = 0
    dut.prob_rd_index_i.value = 0
    dut.observe_select_i.value = 0
    dut.observe_token_i.value = 0
    dut.observe_feature_i.value = 0
    dut.observe_req_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 20, unit="ns").start())
    for _ in range(12):
        await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    for _ in range(20):
        await FallingEdge(dut.clk_i)

    # Count actual result DMA handshakes by phase (not testbench starts).
    counts = {"projection": 0, "qkt": 0}
    seen_proj = set()
    seen_qkt = set()
    finished = False

    async def monitor():
        nonlocal finished
        while not finished:
            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")
            if int(dut.dma_result_write_o.value):
                token = int(dut.active_token_o.value)
                mode = int(dut.qkt_mode_o.value)
                if mode:
                    counts["qkt"] += 1
                    seen_qkt.add(token)
                else:
                    counts["projection"] += 1
                    seen_proj.add((int(dut.active_projection_o.value), token))

    monitor_task = cocotb.start_soon(monitor())
    dut._log.info("RTL E2E START: 12 Projection -> Quantization -> 4 QKT")
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 1
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    try:
        await with_timeout(RisingEdge(dut.done_o), 15_000_000, "ns")
    except SimTimeoutError as exc:
        raise AssertionError(
            "attention_top timed out: "
            f"busy={dut.busy_o.value} "
            f"qkt_mode={dut.qkt_mode_o.value} "
            f"projection={dut.active_projection_o.value} "
            f"token={dut.active_token_o.value} "
            f"stream_addr={dut.stream_addr_o.value} "
            f"writes={counts}"
        ) from exc
    await FallingEdge(dut.clk_i)
    finished = True
    monitor_task.cancel()
    # Debug Q8/K8 after the complete attention operation
    for name, select, expected in [
        ("Q8", 3, q8),
        ("K8", 4, k8),
    ]:
        actual = np.zeros((4, 8), dtype=np.int64)

        for token in range(4):
            for feature in range(8):
                raw = await read_observe(
                    dut, select, token, feature
                )

                value = raw & 0xFF
                actual[token, feature] = (
                    value - 256 if value & 0x80 else value
                )

        dut._log.info(
            "%s DEBUG:\nExpected:\n%s\nActual:\n%s",
            name, expected, actual
        )

    for name, select in [("Q shift", 5), ("K shift", 6)]:
        shift = await read_observe(dut, select, 0, 0)
        dut._log.info("%s DEBUG: %d", name, shift)
    assert int(dut.scores_complete_o.value) == 1, (
        "Global DONE but QKT score buffer is incomplete"
    )
    assert seen_proj == {(p, t) for p in range(3) for t in range(4)}, (
        f"Missing projection GEMV output(s): {seen_proj}"
    )
    assert seen_qkt == {0, 1, 2, 3}, f"Missing QKT GEMV(s): {seen_qkt}"
    assert counts["projection"] == 384, (
        f"Expected 12*32 projection DMA result beats, got {counts['projection']}"
    )
    assert counts["qkt"] == 16, (
        f"Expected 4*4 QKT DMA result beats, got {counts['qkt']}"
    )

    actual_scores = np.empty((4, 4), dtype=np.int64)
    for token in range(4):
        for j in range(4):
            dut.score_rd_token_i.value = token
            dut.score_rd_index_i.value = j
            await Timer(1, unit="ns")
            assert int(dut.score_rd_valid_o.value) == 1, (
                f"Missing score S[{token},{j}]"
            )
            raw = int(dut.score_rd_data_o.value)
            actual_scores[token, j] = raw - (1 << 20) if (raw & (1 << 19)) else raw

    dut._log.info("Expected quantization shifts: Q=%s K=%s", q_shift, k_shift)
    dut._log.info("Expected score matrix:\n%s", golden_scores)
    dut._log.info("Actual score matrix:\n%s", actual_scores)
    assert np.array_equal(actual_scores, golden_scores), (
        f"End-to-end bit-exact mismatch\n"
        f"Golden Q8:\n{q8}\nGolden K8:\n{k8}\n"
        f"Expected:\n{golden_scores}\nActual:\n{actual_scores}"
    )
    # Reference Softmax using the existing fixed-point EXP LUT model
    golden_probabilities = []

    for row in golden_scores:
        exp_values = [
            int(v) for v in exp_row(
                [int(score) for score in row],
                q_shift,
                k_shift
            )
        ]

        total = sum(exp_values)
        assert total > 0

        probabilities = [
            (exp * 32768 + total // 2) // total
            for exp in exp_values
        ]

        golden_probabilities.append(probabilities)

    golden_probabilities = np.asarray(
        golden_probabilities, dtype=np.int64
    )

    # Global DONE must imply that Softmax is complete.
    assert int(dut.softmax_complete_o.value) == 1, (
        "Global DONE but Softmax probability buffer is incomplete"
    )

    actual_probabilities = np.empty((4, 4), dtype=np.int64)

    for token in range(4):
        for j in range(4):
            dut.prob_rd_token_i.value = token
            dut.prob_rd_index_i.value = j

            await Timer(1, unit="ns")

            assert int(dut.prob_rd_valid_o.value) == 1, (
                f"Missing probability P[{token},{j}]"
            )

            actual_probabilities[token, j] = int(
                dut.prob_rd_data_o.value
            )

    dut._log.info(
        "PASS: single OpenEye and DMA adapter, 12 projections, "
        "hardware Q/K quantization, 4 QKT GEMVs, all 16 scores bit-exact"
    )
    dut._log.info(
        "Expected Softmax probabilities (UQ1.15):\n%s",
        golden_probabilities
    )

    dut._log.info(
        "Actual RTL Softmax probabilities (UQ1.15):\n%s",
        actual_probabilities
    )

    assert np.array_equal(
        actual_probabilities,
        golden_probabilities
    ), (
        f"Softmax End-to-End mismatch\n"
        f"Expected:\n{golden_probabilities}\n"
        f"Actual:\n{actual_probabilities}"
    )

    # Each normalized row should sum to approximately 1.0.
    for token in range(4):
        row_sum = int(actual_probabilities[token].sum())

        assert abs(row_sum - 32768) <= 2, (
            f"Invalid probability sum in row {token}: {row_sum}"
        )

    dut._log.info(
        "PASS: 12 projections, quantization, 4 QKT GEMVs, "
        "16 scores and 16 Softmax probabilities"
    )