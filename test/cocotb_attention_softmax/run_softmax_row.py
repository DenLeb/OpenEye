
from pathlib import Path
from cocotb_test.simulator import run


def test_softmax_row():
    here = Path(__file__).resolve().parent
    project_root = here.parent.parent

    run(
        verilog_sources=[
            str(project_root / "hdl" / "attention_softmax_exp.v"),
            str(project_root / "hdl" / "attention_softmax_divider.v"),
            str(project_root / "hdl" / "attention_softmax_normalize.v"),
            str(project_root / "hdl" / "attention_softmax_row.v"),
        ],
        toplevel="attention_softmax_row",
        module="test_softmax_row_cocotb",
        python_search=[str(here)],
        sim_build=str(here / "sim_build_row"),
        simulator="icarus",
        force_compile=True,
    )
