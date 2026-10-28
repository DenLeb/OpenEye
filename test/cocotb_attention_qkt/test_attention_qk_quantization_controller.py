"""Independent RTL controller+existing quantization_unit end-to-end test."""
from pathlib import Path
from cocotb_test.simulator import run


def test_qk_quantization_controller():
    test_dir = Path(__file__).resolve().parent
    root = test_dir.parents[1]
    run(
        verilog_sources=[
            str(root / "hdl" / "quantization_unit.v"),
            str(root / "hdl" / "attention_qk_quantization_controller.v"),
            str(test_dir / "attention_qk_quantization_controller_test_top.v"),
        ],
        toplevel="attention_qk_quantization_controller_test_top",
        module="attention_qk_quantization_controller_tb",
        python_search=[str(test_dir)],
        simulator="icarus",
        sim_build=str(test_dir / ".sim_build_qk_controller"),
        waves=False,
    )
