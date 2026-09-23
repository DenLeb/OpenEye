from pathlib import Path

import cocotb_test.simulator

from open_eye import hdl_dir
import pe_test_utils as ptu


def test_matrix():

    test_dir = Path(__file__).parent

    cocotb_test.simulator.run(

        verilog_sources=ptu.get_verilog_sources(str(hdl_dir)),

        toplevel="PE",
        module="PE_matrix_tb",

        python_search=[str(test_dir)],

        parameters={
            "SPARSITY_EN": 0,
            "USE_DSP": 0,
            "PARALLEL_MACS": 1,
        },

        defines={
            "USE_INTERNAL_PARAMS_PE": "1",
        },

        simulator="icarus",

        sim_build=str(test_dir / ".temp" / "pe_matrix"),

        extra_env={
            "IVERILOG_DUMPER": "fst",
        },

        waves=True,
        force_compile=True,
    )