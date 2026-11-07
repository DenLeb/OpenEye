"""Single-port PSUM page-preservation test across all twelve Q/K/V GEMVs.

Used by test_psum_paging_12.py. Existing 64-bit DMA output is checked against
NumPy after each GEMV. The actual bank-0 RAM_SP contents are read via the
*same* physical single port only while OpenEye is idle: once immediately after
creation, and again for all 12 pages after the final GEMV, without any reset.
"""
import csv
import json
import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

STREAM_WORDS = 942
OUTPUT_BEATS = 32
FEATURES = 8
PAGE_STRIDE = 32
SIM_CYCLES = int(os.getenv('PSUM_PAGING_12_MAX_CYCLES', '4000'))
OPERATIONS = [(kind, projection, token, 4 * projection + token)
              for projection, kind in enumerate(('q', 'k', 'v'))
              for token in range(4)]


def value(handle):
    return int(handle.value)


def signed20(raw):
    raw &= (1 << 20) - 1
    return raw - (1 << 20) if (raw & (1 << 19)) else raw


async def await_idle(dut):
    """Ensure physical single-port RAM is not being used for a computation."""
    for _ in range(60):
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
        if (not value(dut.busy_o)
                and not value(dut.adapter_busy_o)
                and value(dut.u_openeye.fsm_psum_current_state) == 0
                and not value(dut.u_openeye.psum_buffer_en_w)):
            return
    raise AssertionError('Accelerator/PSUM did not become idle for physical readback')


async def read_page_bank0(dut, kind, token, projection, page, expected, phase):
    """Read actual PSUM RAM_SP contents using its original physical port.

    Test-only mux takes over *while idle*. RAM_SP_generic with Pipelined=1
    provides output after its internal memory read and output registers;
    wait three rising edges per address to sample a stable 40-bit RAM word.
    """
    assert not value(dut.busy_o), 'Do not debug-read RAM during GEMV'
    assert not value(dut.adapter_busy_o), 'DMA still active during debug read'
    assert not value(dut.u_openeye.psum_buffer_en_w), 'PSUM write active during debug read'
    assert value(dut.u_openeye.fsm_psum_current_state) == 0, 'PSUM FSM not idle'

    dut.psum_test_debug_read_i.value = 1
    rows = []
    try:
        for feature in range(FEATURES):
            physical = page * PAGE_STRIDE + feature
            await FallingEdge(dut.clk_i)
            await Timer(1, unit='ns')
            dut.psum_test_debug_addr_i.value = physical
            for _ in range(3):
                await RisingEdge(dut.clk_i)
                await Timer(1, unit='ns')
            raw = value(dut.psum_test_debug_data_o)
            observed = signed20(raw)
            want = int(expected[feature])
            rows.append({
                'check_phase': phase,
                'projection': kind.upper(),
                'token': token,
                'page': page,
                'bank': 0,
                'feature': feature,
                'physical_address': physical,
                'expected_signed20': want,
                'observed_signed20': observed,
                'match': observed == want,
                'raw_word_hex': f'{raw:010x}',
            })
            if observed != want:
                raise AssertionError(
                    f'{phase}: {kind.upper()}[{token}][{feature}] at bank 0 '
                    f'physical address {physical}: expected {want}, got '
                    f'{observed}, word 0x{raw:010x}'
                )
    finally:
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
        dut.psum_test_debug_read_i.value = 0
    return rows


def save_readbacks(rows, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'check_phase', 'projection', 'token', 'page', 'bank',
            'feature', 'physical_address', 'expected_signed20',
            'observed_signed20', 'match', 'raw_word_hex'])
        writer.writeheader()
        writer.writerows(rows)


async def run_gemv(dut, data_dir, kind, projection, token, page, expected):
    name = f'{kind.upper()}[{token}]'
    stream_path = data_dir / 'verification_streams' / f'{kind}_t{token}.mem'
    stream = [int(line, 16) for line in stream_path.read_text().splitlines()
              if line.strip()]
    assert len(stream) == STREAM_WORDS, f'{name}: wrong input DMA length'

    for _ in range(3):
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
    for _ in range(200):
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
        if (value(dut.u_openeye.ready_dma_o) and not value(dut.busy_o)
                and not value(dut.adapter_busy_o)):
            break
    else:
        raise AssertionError(f'{name}: OpenEye not ready for next GEMV')
    assert not value(dut.psum_test_debug_read_i)

    dut._log.info('PSUM PAGING 12: start %s, page=%d, addresses=%d..%d',
                  name, page, page * PAGE_STRIDE, page * PAGE_STRIDE + 31)
    dut.token_select_i.value = token
    dut.projection_select_i.value = projection
    dut.start_i.value = 1
    await FallingEdge(dut.clk_i)
    await Timer(1, unit='ns')
    dut.start_i.value = 0

    sent = 0
    received = 0
    results = []
    saw_psum_write = False
    done = False
    for cycle in range(SIM_CYCLES):
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
        fpga = dut.u_openeye
        assert value(fpga.psum_test_page_i) == page, (
            f'{name}: page selection changed during GEMV')
        if value(fpga.psum_buffer_en_w):
            saw_psum_write = True
        if value(fpga.enable_dma_i):
            assert sent < STREAM_WORDS, f'{name}: extra DMA input word'
            actual = value(fpga.data_dma_i)
            if actual != stream[sent]:
                raise AssertionError(
                    f'{name}: DMA input word {sent}: 0x{actual:016x} != '
                    f'0x{stream[sent]:016x}'
                )
            sent += 1
            # Deliberately perturb LIVE selectors, proving that the page
            # metadata is latched on GEMV start and not sampled mid-stream.
            if sent == 100:
                dut.token_select_i.value = (token + 1) % 4
                dut.projection_select_i.value = (projection + 1) % 3
            if sent == 200:
                dut.token_select_i.value = token
                dut.projection_select_i.value = projection
        if value(dut.result_write_o):
            assert received < OUTPUT_BEATS, f'{name}: too many output beats'
            address = value(dut.result_addr_o)
            assert address == received, (
                f'{name}: output DMA address {address}, expected {received}')
            results.append(value(dut.result_data_o))
            received += 1
        if value(dut.done_o):
            done = True
            break
        if cycle % 1000 == 0:
            dut._log.info('%s: cycle=%d sent=%d/%d outputs=%d/%d',
                          name, cycle, sent, STREAM_WORDS, received, OUTPUT_BEATS)

    assert done, f'{name}: timed out at {SIM_CYCLES} cycles'
    assert sent == STREAM_WORDS, f'{name}: sent {sent}/{STREAM_WORDS} words'
    assert received == OUTPUT_BEATS, f'{name}: received {received}/{OUTPUT_BEATS}'
    assert saw_psum_write, f'{name}: no PSUM RAM write observed'

    ordered = [0] * OUTPUT_BEATS
    for beat, word in enumerate(results):
        logical_index = (beat % 2) * 16 + beat // 2
        ordered[logical_index] = signed20(word)
    assert ordered[:FEATURES] == list(expected), (
        f'{name}: wrong GEMV result; got {ordered[:FEATURES]}, '
        f'expected {list(expected)}')
    assert ordered[FEATURES:] == [0] * (OUTPUT_BEATS - FEATURES), (
        f'{name}: nonzero DMA padding: {ordered[FEATURES:]}')
    dut._log.info('%s: DMA matched NumPy (8 features + 24 zeros, 32 beats)', name)
    await await_idle(dut)


@cocotb.test()
async def all_twelve_qkv_pages_survive(dut):
    data_dir = Path(os.environ.get('PSUM_PAGING_12_DATA_DIR',
        str(Path(__file__).resolve().parent / 'data' / 'attention_mapper_v1_fixed')))
    with np.load(data_dir / 'inputs_used.npz') as tensors:
        x = tensors['X'].astype(np.int64)
        matrices = {
            'q': tensors['WQ'].astype(np.int64),
            'k': tensors['WK'].astype(np.int64),
            'v': tensors['WV'].astype(np.int64),
        }
    assert x.shape == (4, 8), f'Expected X[4,8], got {x.shape}'
    assert all(w.shape == (8, 8) for w in matrices.values()), (
        f'Unexpected weight shapes: { {k: w.shape for k, w in matrices.items()} }')
    expected = {(kind, token): (x[token] @ matrices[kind]).tolist()
                for kind, projection, token, page in OPERATIONS}

    dut.clk_i.value = 0
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.token_select_i.value = 0
    dut.projection_select_i.value = 0
    dut.iact_we_i.value = 0
    dut.iact_wr_addr_i.value = 0
    dut.iact_wr_data_i.value = 0
    dut.wght_we_i.value = 0
    dut.wght_wr_addr_i.value = 0
    dut.wght_wr_data_i.value = 0
    dut.q_rd_token_i.value = 0
    dut.q_rd_feature_i.value = 0
    dut.k_rd_token_i.value = 0
    dut.k_rd_feature_i.value = 0
    dut.v_rd_token_i.value = 0
    dut.v_rd_feature_i.value = 0
    dut.quant_start_i.value = 0
    dut.q8_rd_token_i.value = 0
    dut.q8_rd_feature_i.value = 0
    dut.k8_rd_token_i.value = 0
    dut.k8_rd_feature_i.value = 0
    dut.psum_test_debug_read_i.value = 0
    dut.psum_test_debug_addr_i.value = 0

    cocotb.start_soon(Clock(dut.clk_i, 20, unit='ns').start())
    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit='ns')
    dut.rst_ni.value = 1
    for _ in range(8):
        await RisingEdge(dut.clk_i)
        await Timer(1, unit='ns')

    readbacks = []
    results = []
    test_dir = Path(__file__).resolve().parent
    csv_path = test_dir / 'psum_paging_12_readback.csv'
    save_readbacks(readbacks, csv_path)
    for kind, projection, token, page in OPERATIONS:
        assert page == 4 * projection + token
        assert page < 12, 'Page ID exceeds 4-bit page selector'
        await run_gemv(dut, data_dir, kind, projection, token, page,
                       expected[(kind, token)])
        # Read each newly written page immediately to localize failures.
        rows = await read_page_bank0(dut, kind, token, projection, page,
                                     expected[(kind, token)], 'immediate')
        readbacks.extend(rows)
        save_readbacks(readbacks, csv_path)  # checkpoint, useful on later failure
        results.append({'projection': kind.upper(), 'token': token,
                        'page': page, 'dma_verified': True,
                        'immediate_ram_verified': True})
        dut._log.info('PSUM RAM PAGE PASS: %s[%d] bank0 addr %d..%d',
                      kind.upper(), token, 32 * page, 32 * page + 7)

    # This is the key preservation test: re-read every page AFTER all 12
    # GEMVs have run, with no intervening reset or external result storage.
    for kind, projection, token, page in OPERATIONS:
        rows = await read_page_bank0(dut, kind, token, projection, page,
                                     expected[(kind, token)], 'final')
        readbacks.extend(rows)
        save_readbacks(readbacks, csv_path)  # retain completed rows if later page fails
        dut._log.info('FINAL RAM PRESERVATION PASS: %s[%d] page=%d, bank0 %d..%d',
                      kind.upper(), token, page,
                      PAGE_STRIDE * page, PAGE_STRIDE * page + 7)

    save_readbacks(readbacks, csv_path)
    summary_path = test_dir / 'psum_paging_12_summary.json'
    summary_path.write_text(json.dumps({
        'status': 'PASS',
        'execution_order': [f'{kind.upper()}[{token}]' for kind, _, token, _ in OPERATIONS],
        'page_stride': PAGE_STRIDE,
        'physical_ram_type': 'RAM_SP (unchanged single-port)',
        'gemvs_verified': len(results),
        'dma_beats_verified': len(results) * OUTPUT_BEATS,
        'logical_projection_values_final_verified': 12 * FEATURES,
        'ram_readbacks_verified': len(readbacks),
        'readback_csv': str(csv_path),
        'pages': results,
    }, indent=2), encoding='utf-8')
    dut._log.info('PSUM FULL QKV PAGE PRESERVATION PASSED: 12 GEMVs, '
                  '384 DMA beats, 96 final RAM values, 192 total RAM readbacks; '
                  'original single-port RAM unchanged; CSV=%s', csv_path)
