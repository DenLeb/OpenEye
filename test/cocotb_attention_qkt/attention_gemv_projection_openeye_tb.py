# Multi-GEMV integration test without resetting the hardware.
# Executes all 12 Q/K/V projections using token and projection selectors.
# Verifies DMA streams, GEMV results, and selector stability.

from pathlib import Path
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, Timer

def reference_quantize(raw):

    values = np.asarray(raw, dtype=np.int64)

    max_abs = int(np.max(np.abs(values)))

    shift = 0
    while max_abs > (127 << shift):
        shift += 1

    if shift == 0:
        quantized = values
    else:
        half = 1 << (shift - 1)

        quantized = np.where(
            values >= 0,
            (values + half) >> shift,
            -((-values + half) >> shift)
        )

    quantized = np.clip(quantized, -128, 127)

    return quantized.astype(np.int64), shift

async def check_dense_mode(dut):
    fpga = dut.u_openeye

    # OpenEye calculation-related FSM states
    COMPUTE_STATES = (7, 8, 9, 10)

    while True:
        await RisingEdge(dut.clk_i)
        await ReadOnly()

        state = int(fpga.fsm_current_state.value)

        if state in COMPUTE_STATES:
            mode = int(fpga.gemm_mode.value)

            assert mode == 0, (
                f"Dense Mode violation: "
                f"gemm_mode={mode}, FSM state={state}"
            )

@cocotb.test()
async def test_twelve_projection_gemvs(dut):

    data_dir = (
    Path(__file__).resolve().parent
    / "data"
    / "attention_mapper_v1"
    )

    # Reference data for both GEMV operations
    gemv_cases = [
        (proj_id, token, f"verification_streams/{kind}_t{token}.mem")
        for proj_id, kind in enumerate(("q", "k", "v"))
        for token in range(4)
    ]

    with np.load(data_dir / "inputs_used.npz") as tensors:
        x = tensors["X"].astype(np.int64)
        weights = {
            "q": tensors["WQ"].astype(np.int64),
            "k": tensors["WK"].astype(np.int64),
            "v": tensors["WV"].astype(np.int64),
        }

    # Initialize hardware only once
    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.token_select_i.value = 0
    dut.projection_select_i.value = 0

    # Activation RAM write interface
    dut.iact_we_i.value = 0
    dut.iact_wr_addr_i.value = 0
    dut.iact_wr_data_i.value = 0

    # Weight RAM write interface
    dut.wght_we_i.value = 0
    dut.wght_wr_addr_i.value = 0
    dut.wght_wr_data_i.value = 0

    # Q/K/V Buffer read interface
    dut.q_rd_token_i.value = 0
    dut.q_rd_feature_i.value = 0

    dut.k_rd_token_i.value = 0
    dut.k_rd_feature_i.value = 0

    dut.v_rd_token_i.value = 0
    dut.v_rd_feature_i.value = 0

    # Quantization controller
    dut.quant_start_i.value = 0

    # Q8 read interface
    dut.q8_rd_token_i.value = 0
    dut.q8_rd_feature_i.value = 0

    # K8 read interface
    dut.k8_rd_token_i.value = 0
    dut.k8_rd_feature_i.value = 0

    cocotb.start_soon(check_dense_mode(dut))
    cocotb.start_soon(Clock(dut.clk_i, 20, unit="ns").start())

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    dut.rst_ni.value = 1

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    # Execute three independent GEMVs without another reset
    for run_id, (proj_id, token, input_file) in enumerate(gemv_cases):

        input_words = [
            int(line, 16)
            for line in (data_dir / input_file).read_text().splitlines()
            if line.strip()
        ]

        kind = ("q", "k", "v")[proj_id]

        expected = (
            x[token] @ weights[kind]
        ).tolist()

        assert len(input_words) == 939
        assert len(expected) == 8

        # Continue with the existing ready-wait loop...

        # Allow controller to return to IDLE
        for _ in range(3):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

        # Wait until OpenEye can accept a new stream
        for _ in range(200):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            if (int(dut.u_openeye.ready_dma_o.value) == 1
                    and int(dut.busy_o.value) == 0):
                break
        else:
            raise AssertionError(
                f"GEMV {run_id + 1}: OpenEye not ready"
            )

        dut._log.info(
            "Starting %s[%d] - GEMV %d/12",
            kind.upper(), token, run_id + 1
        )

        dut.token_select_i.value = token
        dut.projection_select_i.value = proj_id

        dut.start_i.value = 1

        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        dut.start_i.value = 0

        sent_count = 0
        raw_results = []

        # Monitor this GEMV
        for cycle in range(2500):

            await FallingEdge(dut.clk_i)
            await Timer(1, unit="ns")

            if int(dut.u_openeye.enable_dma_i.value):

                assert sent_count < 939, (
                    f"{kind.upper()}[{token}]: Extra input word"
                )

                actual_word = int(dut.u_openeye.data_dma_i.value)

                assert actual_word == input_words[sent_count], (
                    f"{kind.upper()}[{token}]: "
                    f"Input mismatch at word {sent_count}"
                )

                sent_count += 1

                # Change external selections during transmission
                if sent_count == 100:
                    dut.token_select_i.value = (token + 1) % 4
                    dut.projection_select_i.value = (proj_id + 1) % 3

                # Restore original selections
                if sent_count == 200:
                    dut.token_select_i.value = token
                    dut.projection_select_i.value = proj_id

            if int(dut.result_write_o.value):

                address = int(dut.result_addr_o.value)
                word = int(dut.result_data_o.value)

                assert address == len(raw_results), (
                    f"GEMV {run_id + 1}: Result address mismatch"
                )

                raw_results.append(word)

            if int(dut.done_o.value):
                break

        else:
            raise AssertionError(
                f"GEMV {run_id + 1} timeout: "
                f"sent={sent_count}/939, "
                f"results={len(raw_results)}"
            )

        assert sent_count == 939
        assert len(raw_results) == 32

        # Decode signed 20-bit PSUM values
        decoded = []

        for word in raw_results:
            value = word & ((1 << 20) - 1)

            if value & (1 << 19):
                value -= 1 << 20

            decoded.append(value)

        # Restore logical output ordering
        actual = [0] * 32

        for beat, value in enumerate(decoded):
            filter_index = (beat % 2) * 16 + (beat // 2)
            actual[filter_index] = value

        assert actual[:8] == expected, (
            f"{kind.upper()}[{token}]: "
            f"Expected {expected}, got {actual[:8]}"
        )

        assert actual[8:] == [0] * 24, (
            f"{kind.upper()}[{token}]: Padding mismatch"
        )

        dut._log.info(
            "%s[%d] PASSED: cycles=%d, inputs=%d, outputs=%d",
            kind.upper(), token, cycle + 1,
            sent_count, len(raw_results)
        )

    dut._log.info(
        "All 12 Q/K/V projection GEMVs PASSED without reset"
    )
    # =====================================================
    # Verify complete Q/K/V matrices stored in FPGA buffers
    # =====================================================

    buffer_ports = {
        "q": dut.q_data_o,
        "k": dut.k_data_o,
        "v": dut.v_data_o,
    }

    for kind in ("q", "k", "v"):

        expected_matrix = x @ weights[kind]

        for token in range(4):
            for feature in range(8):

                getattr(dut, f"{kind}_rd_token_i").value = token
                getattr(dut, f"{kind}_rd_feature_i").value = feature

                await Timer(1, unit="ns")

                raw = int(buffer_ports[kind].value)

                # Signed 20-bit conversion
                actual = raw
                if actual & (1 << 19):
                    actual -= 1 << 20

                expected = int(expected_matrix[token, feature])

                assert actual == expected, (
                    f"{kind.upper()} Buffer mismatch: "
                    f"token={token}, feature={feature}, "
                    f"expected={expected}, actual={actual}"
                )

        dut._log.info(
            "%s Buffer PASSED: all 32 matrix elements verified",
            kind.upper()
        )
    
    dut._log.info(
        "All Q/K/V buffers verified: 96 matrix elements PASSED"
    )
    # Verify independent simultaneous Q/K/V reads
    dut.q_rd_token_i.value = 1
    dut.q_rd_feature_i.value = 3

    dut.k_rd_token_i.value = 2
    dut.k_rd_feature_i.value = 5

    dut.v_rd_token_i.value = 3
    dut.v_rd_feature_i.value = 1

    await Timer(1, unit="ns")

    cases = [
        ("Q", dut.q_data_o, weights["q"], 1, 3),
        ("K", dut.k_data_o, weights["k"], 2, 5),
        ("V", dut.v_data_o, weights["v"], 3, 1),
    ]

    for name, signal, weight, token, feature in cases:
        raw = int(signal.value)
        actual = raw - (1 << 20) if raw & (1 << 19) else raw
        expected = int((x @ weight)[token, feature])

        assert actual == expected, (
            f"{name} independent read failed: "
            f"expected={expected}, actual={actual}"
        )

    dut._log.info("Independent Q/K/V address test PASSED")


    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        data_dir / "qk_raw_projection.npz",
        q_raw=np.asarray(x @ weights["q"], dtype=np.int64),
        k_raw=np.asarray(x @ weights["k"], dtype=np.int64),
    )

    dut._log.info("Q/K raw matrices exported for quantization analysis")

    # =====================================================
    # Q/K Quantization Integration Test
    # =====================================================

    dut._log.info("Starting Q/K hardware quantization")

    # Python reference
    q_raw = x @ weights["q"]
    k_raw = x @ weights["k"]

    q8_expected, q_shift_expected = reference_quantize(q_raw)
    k8_expected, k_shift_expected = reference_quantize(k_raw)

    # Start the RTL quantization controller
    assert int(dut.busy_o.value) == 0
    assert int(dut.adapter_busy_o.value) == 0

    dut.quant_start_i.value = 1

    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ns")

    dut.quant_start_i.value = 0

    assert int(dut.quant_busy_o.value) == 1, (
        "Quantization controller did not start"
    )

    # Wait until quantization finishes
    for cycle in range(160):

        await RisingEdge(dut.clk_i)
        await Timer(1, unit="ns")

        if int(dut.quant_done_o.value):
            break

    else:
        raise AssertionError("Q/K Quantization timeout")

    dut._log.info(
        "Quantization finished after %d cycles",
        cycle + 1
    )

    # =====================================================
    # Verify Q/K shifts
    # =====================================================

    q_shift_actual = int(dut.q_shift_o.value)
    k_shift_actual = int(dut.k_shift_o.value)

    assert q_shift_actual == q_shift_expected
    assert k_shift_actual == k_shift_expected

    dut._log.info(
        "Shift verification PASSED: Q=%d, K=%d",
        q_shift_actual,
        k_shift_actual
    )

    # =====================================================
    # Verify all 64 Q8/K8 elements
    # =====================================================

    q8_actual = np.zeros((4, 8), dtype=np.int64)
    k8_actual = np.zeros((4, 8), dtype=np.int64)

    for token in range(4):
        for feature in range(8):

            dut.q8_rd_token_i.value = token
            dut.q8_rd_feature_i.value = feature

            dut.k8_rd_token_i.value = token
            dut.k8_rd_feature_i.value = feature

            await Timer(1, unit="ns")

            q_value = int(dut.q8_data_o.value) & 0xFF
            k_value = int(dut.k8_data_o.value) & 0xFF

            # Convert 8-bit Two's Complement
            if q_value & 0x80:
                q_value -= 256

            if k_value & 0x80:
                k_value -= 256

            q8_actual[token, feature] = q_value
            k8_actual[token, feature] = k_value

            assert q_value == q8_expected[token, feature], (
                f"Q8 mismatch: token={token}, feature={feature}, "
                f"expected={q8_expected[token, feature]}, "
                f"actual={q_value}"
            )

            assert k_value == k8_expected[token, feature], (
                f"K8 mismatch: token={token}, feature={feature}, "
                f"expected={k8_expected[token, feature]}, "
                f"actual={k_value}"
            )

    dut._log.info(
        "Q8/K8 Buffer PASSED: all 64 elements verified"
    )

    # =====================================================
    # Quantization accuracy evaluation
    # =====================================================

    original_scores = q_raw @ k_raw.T

    quantized_scores = q8_actual @ k8_actual.T

    reconstructed_scores = (
        quantized_scores *
        (1 << (q_shift_actual + k_shift_actual))
    )

    relative_error = (
        np.linalg.norm(
            (reconstructed_scores - original_scores).astype(np.float64)
        )
        /
        np.linalg.norm(original_scores.astype(np.float64))
    ) * 100

    dut._log.info(
        "QK^T relative quantization error: %.6f%%",
        relative_error
    )

    dut._log.info(
        "FULL PROJECTION -> Q/K BUFFER -> QUANTIZATION PASSED"
    )