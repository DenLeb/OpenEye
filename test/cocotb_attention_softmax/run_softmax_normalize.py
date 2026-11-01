from pathlib import Path
from cocotb_test.simulator import run


def test_softmax_normalize():

    here = Path(__file__).resolve().parent
    project_root = here.parent.parent

    run(
        verilog_sources=[
            str(project_root / "hdl" / "attention_softmax_divider.v"),
            str(project_root / "hdl" / "attention_softmax_normalize.v"),
        ],
        toplevel="attention_softmax_normalize",
        module="test_softmax_normalize_cocotb",
        python_search=[str(here)],
        sim_build=str(here / "sim_build_normalize"),
        simulator="icarus",
        force_compile=True,
    )