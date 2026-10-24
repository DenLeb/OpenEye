from pathlib import Path
import cocotb_test.simulator


def test_attention_qkt_ctrl():

    root = Path(__file__).resolve().parents[2]
    test_dir = Path(__file__).resolve().parent

    cocotb_test.simulator.run(
        verilog_sources=[
            str(root / "hdl" / "attention_qkt_ctrl.v")
        ],
        toplevel="attention_qkt_ctrl",
        module="attention_qkt_ctrl_tb",
        python_search=[str(test_dir)],
        sim_build=str(test_dir / ".sim_build"),
        simulator="icarus",
        force_compile=True,
    )