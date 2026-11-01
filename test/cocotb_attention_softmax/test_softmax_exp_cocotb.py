"""Standalone Cocotb test of attention_softmax_exp (not the full Attention RTL)."""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from softmax_exp_reference import exp_row


def pack_row(values):
    return sum((int(v) & ((1 << 20) - 1)) << (20*i) for i,v in enumerate(values))


@cocotb.test()
async def check_scaling_and_exp(dut):
    dut.rst_ni.value = 0
    dut.valid_i.value = 0
    dut.scores_i.value = 0
    dut.q_shift_i.value = 0
    dut.k_shift_i.value = 0
    cocotb.start_soon(Clock(dut.clk_i, 10, unit='ns').start())
    for _ in range(3):
        await FallingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    from pathlib import Path
    import numpy as np
    ref_path = Path(__file__).resolve().parent.parent / "cocotb_attention_qkt"/ "data"/ "qkt_single_8x4"/ "qkt_t0_reference.npz"
    with np.load(ref_path, allow_pickle=False) as f:
        matrix = f['Q8'].astype(np.int64) @ f['K8'].astype(np.int64).T
        qs,ks = int(f['q_shift']), int(f['k_shift'])
    cases = [(list(map(int,row)),qs,ks) for row in matrix]
    cases += [
        ([0,0,0,0],6,6),
        ([100,-100000,-200000,-300000],6,6),
        ([-524288,-500000,524287,0],13,13),
        ([-1,-2,-3,-4],0,0),
        ([1,1,1,1],0,0),
        ([524287,-524288,7,-8],5,9),
    ]
    rng = random.Random(37)
    cases += [([rng.randrange(-(1<<19),1<<19) for _ in range(4)],
               rng.randrange(14),rng.randrange(14)) for _ in range(100)]

    for row,q,k in cases:
        await FallingEdge(dut.clk_i)
        dut.scores_i.value = pack_row(row)
        dut.q_shift_i.value = q
        dut.k_shift_i.value = k
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
        await Timer(1,unit='ns')
        assert int(dut.valid_o.value) == 1
        got = int(dut.exp_values_o.value)
        actual = [(got >> (16*i)) & 0xFFFF for i in range(4)]
        expected = exp_row(row,q,k)
        assert actual == expected, f'row={row}, shift={q},{k}: {actual} != {expected}'

    await FallingEdge(dut.clk_i)
    dut.valid_i.value = 0
    await RisingEdge(dut.clk_i)
    await Timer(1,unit='ns')
    assert int(dut.valid_o.value) == 0
    dut._log.info('PASS: 110 scaling/EXP cases including baseline, extremes and random')
