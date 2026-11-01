from pathlib import Path
from cocotb_test.simulator import run


def test_softmax_divider():

    here = Path(__file__).resolve().parent
    project_root = here.parent.parent

    run(
        verilog_sources=[
            str(project_root / "hdl" / "attention_softmax_divider.v")
        ],
        toplevel="attention_softmax_divider",
        module="test_softmax_divider_cocotb",
        python_search=[str(here)],
        sim_build=str(here / "sim_build_divider"),
        simulator="icarus",
        force_compile=True
    )