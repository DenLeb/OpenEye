from pathlib import Path
import cocotb_test.simulator


def test_attention_gemv_control_top():

    root = Path(__file__).resolve().parents[2]
    test_dir = Path(__file__).resolve().parent

    cocotb_test.simulator.run(
        verilog_sources=[
            str(root / "hdl" / "attention_qkt_ctrl.v"),
            str(root / "hdl" / "attention_dma_adapter.v"),
            str(root / "hdl" / "attention_gemv_control_top.v"),
        ],

        toplevel="attention_gemv_control_top",
        module="attention_gemv_control_top_tb",

        python_search=[str(test_dir)],
        sim_build=str(test_dir / ".sim_build"),

        simulator="icarus",
        force_compile=True,
        waves=True,
    )
