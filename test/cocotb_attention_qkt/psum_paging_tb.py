"""Verify two GEMVs use distinct physical PSUM RAM pages (single port only).

Use with test_psum_paging.py. Stores psum_paging_readback.csv with actual RAM
contents read via the existing single port when the accelerator is idle.
No RAM_SP, psum_pipeline, baseline test, or Q/K/V buffer changes.
"""
import csv
import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

STREAM_WORDS = 942
SIM_CYCLES = int(os.getenv('PSUM_PAGING_MAX_CYCLES', '4000'))
PAGE_STRIDE = 32


def value(h):
    return int(h.value)


def signed20(raw):
    raw &= (1 << 20) - 1
    return raw - (1 << 20) if raw & (1 << 19) else raw


async def tick():
    await RisingEdge(cocotb.top.clk_i)
    await Timer(1, unit='ns')


async def read_page_bank0(dut, token, expected):
    """Read actual RAM_SP data through its existing shared address port.

    Called ONLY between GEMVs, while OpenEye is idle; write is disabled by
    the staging mux. Pipelined=1 uses memout and q registers (2 rising edges).
    """
    assert not value(dut.busy_o), 'Cannot debug-read PSUM during a GEMV'
    assert not value(dut.u_openeye.psum_buffer_en_w), 'Unexpected PSUM write active'
    assert value(dut.u_openeye.fsm_psum_current_state) == 0, 'PSUM FSM not idle'

    dut.psum_test_debug_read_i.value = 1
    records = []
    for feature in range(8):
        physical = token * PAGE_STRIDE + feature
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
        dut.psum_test_debug_addr_i.value = physical
        # In RAM_SP_generic, a read updates memout; the following edge updates
        # q when Pipelined=1. Three rising edges give a robust stable sample.
        for _ in range(3):
            await RisingEdge(dut.clk_i)
            await Timer(1, unit='ns')
        physical_data = value(dut.psum_test_debug_data_o)
        actual = signed20(physical_data)
        want = int(expected[feature])
        records.append({
            'token': token, 'bank': 0, 'physical_address': physical,
            'expected_signed20': want, 'observed_signed20': actual,
            'raw_word_hex': f'{physical_data:010x}',
        })
        if actual != want:
            raise AssertionError(
                f'Q[{token}][{feature}] RAM BANK 0 @ {physical}: '
                f'expected {want}, found {actual}; raw=0x{physical_data:010x}'
            )
    await FallingEdge(dut.clk_i)
    await Timer(1, unit='ns')
    dut.psum_test_debug_read_i.value = 0
    return records


@cocotb.test()
async def q0_q1_distinct_psum_pages(dut):
    data_dir = Path(os.environ.get('PSUM_PAGING_DATA_DIR',
        str(Path(__file__).resolve().parent / 'data' / 'attention_mapper_v1_fixed')))
    with np.load(data_dir / 'inputs_used.npz') as tensors:
        x = tensors['X'].astype(np.int64)
        wq = tensors['WQ'].astype(np.int64)
    expected = {token: (x[token] @ wq).tolist() for token in (0, 1)}
    assert all(len(v) == 8 for v in expected.values())

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
    for token in (0, 1):
        stream_path = data_dir / 'verification_streams' / f'q_t{token}.mem'
        stream = [int(s, 16) for s in stream_path.read_text().splitlines() if s.strip()]
        assert len(stream) == STREAM_WORDS
        for _ in range(3):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit='ns')
        for _ in range(200):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit='ns')
            if value(dut.u_openeye.ready_dma_o) and not value(dut.busy_o):
                break
        else:
            raise AssertionError('OpenEye did not become ready')
        assert not value(dut.psum_test_debug_read_i)
        dut._log.info('PSUM PAGE TEST Q[%d]: start, page=%d', token, token)
        dut.token_select_i.value = token
        dut.projection_select_i.value = 0
        dut.start_i.value = 1
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')
        dut.start_i.value = 0

        sent = 0
        received = 0
        raw_results = []
        completed = False
        saw_write = False
        for cycle in range(SIM_CYCLES):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit='ns')
            fpga = dut.u_openeye
            assert value(fpga.psum_test_page_i) == token, (
                f'Physical page changed during Q[{token}]')
            if value(fpga.psum_buffer_en_w):
                saw_write = True
            if value(fpga.enable_dma_i):
                assert sent < STREAM_WORDS, 'Unexpected DMA input after end of stream'
                actual_dma = value(fpga.data_dma_i)
                if actual_dma != stream[sent]:
                    raise AssertionError(
                        f'Q[{token}]: DMA input mismatch at word {sent}: '
                        f'0x{actual_dma:016x} != 0x{stream[sent]:016x}')
                sent += 1
                # Deliberately alter live selectors mid-GEMV, as in passing
                # baseline. The page selector must stay latched.
                if sent == 100:
                    dut.token_select_i.value = (token + 1) % 4
                    dut.projection_select_i.value = 1
                if sent == 200:
                    dut.token_select_i.value = token
                    dut.projection_select_i.value = 0
            if value(dut.result_write_o):
                assert received < 32
                addr = value(dut.result_addr_o)
                assert addr == received, f'Output address {addr}, expected {received}'
                raw_results.append(value(dut.result_data_o))
                received += 1
            if value(dut.done_o):
                completed = True
                break
            if cycle % 500 == 0:
                dut._log.info('Q[%d] cycle=%d sent=%d/%d outputs=%d/32',
                              token, cycle, sent, STREAM_WORDS, received)

        assert completed, f'Q[{token}] timed out after {SIM_CYCLES} cycles'
        assert sent == STREAM_WORDS and received == 32, (
            f'Q[{token}]: sent={sent}/{STREAM_WORDS}, received={received}/32')
        assert saw_write, f'Q[{token}]: no PSUM writes'
        decoded = [signed20(word) for word in raw_results]
        ordered = [0] * 32
        for beat, output in enumerate(decoded):
            feature = (beat % 2) * 16 + (beat // 2)
            ordered[feature] = output
        assert ordered[:8] == expected[token], (
            f'Q[{token}] DMA incorrect: got {ordered[:8]}, expected {expected[token]}')
        assert ordered[8:] == [0] * 24, (
            f'Q[{token}]: nonzero padding in DMA result')
        dut._log.info('Q[%d]: DMA output matched NumPy; 32 beats verified', token)

        # Wait until both state machines and RAM write enables become idle.
        for _ in range(30):
            await FallingEdge(dut.clk_i)
            await Timer(1, unit='ns')
            if (not value(dut.busy_o) and
                value(dut.u_openeye.fsm_psum_current_state) == 0 and
                not value(dut.u_openeye.psum_buffer_en_w)):
                break
        else:
            raise AssertionError('PSUM did not become idle for physical readback')

        # After Q0, validate its own page. After Q1, validate BOTH Q0 and Q1.
        for preserved in range(token + 1):
            records = await read_page_bank0(dut, preserved, expected[preserved])
            readbacks.extend(records)
            dut._log.info('RAM READBACK PASS: Q[%d], bank 0, physical addresses %d..%d',
                          preserved, preserved * PAGE_STRIDE,
                          preserved * PAGE_STRIDE + 7)

    out_path = Path(__file__).resolve().parent / 'psum_paging_readback.csv'
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'token', 'bank', 'physical_address', 'expected_signed20',
            'observed_signed20', 'raw_word_hex'])
        writer.writeheader()
        writer.writerows(readbacks)
    dut._log.info('PSUM PAGE PRESERVATION PASSED: Q0 survived Q1 with distinct pages; '
                  'readback CSV: %s', out_path)
