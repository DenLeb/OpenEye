"""Standalone pytest runner for OpenEye's *existing* quantization_unit.v."""

from pathlib import Path

from cocotb_test.simulator import run


def test_quantization_unit():
    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent.parent

    run(
        verilog_sources=[str(repo_root / "hdl" / "quantization_unit.v")],
        toplevel="quantization_unit",
        module="quantization_unit_tb",
        python_search=[str(test_dir)],
        simulator="icarus",
        parameters={
            "PSUM_WIDTH": 20,
            "MANT_WIDTH": 25,
            "OFFSET_WIDTH": 8,
            "SHIFT_WIDTH": 7,
            "OUT_WIDTH": 8,
        },
        sim_build=str(test_dir / ".sim_build_quantization_unit"),
        waves=False,
    )
