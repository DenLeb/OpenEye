"""Run with: python -m pytest run_softmax_exp.py -v -s"""
from pathlib import Path
from open_eye import hdl_dir
import os


def test_softmax_exp():
    from cocotb_test.simulator import run
    here = Path(__file__).resolve().parent
    run(verilog_sources=[str(Path(hdl_dir) / 'attention_softmax_exp.v')],
        toplevel='attention_softmax_exp',
        module='test_softmax_exp_cocotb',
        python_search=[str(here)],
        sim_build=str(here / 'sim_build'),
        simulator='icarus',
        force_compile=True)
