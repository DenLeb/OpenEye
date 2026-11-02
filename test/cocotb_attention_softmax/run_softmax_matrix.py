
from pathlib import Path
from cocotb_test.simulator import run


def test_softmax_matrix():

    here = Path(__file__).resolve().parent
    project_root = here.parent.parent

    run(
        verilog_sources=[
            str(project_root / "hdl" / "attention_softmax_exp.v"),
            str(project_root / "hdl" / "attention_softmax_divider.v"),
            str(project_root / "hdl" / "attention_softmax_normalize.v"),
            str(project_root / "hdl" / "attention_softmax_row.v"),
            str(project_root / "hdl" / "attention_softmax.v"),
        ],
        toplevel="attention_softmax",
        module="test_softmax_matrix_cocotb",
        python_search=[str(here)],
        sim_build=str(here / "sim_build_matrix"),
        simulator="icarus",
        waves=True,
        force_compile=True,
    )
