from pathlib import Path
import shutil

import cocotb_test.simulator


def test_attention_gemv_rom_top():

    root = Path(__file__).resolve().parents[2]
    test_dir = Path(__file__).resolve().parent

    build_dir = test_dir / ".sim_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(
        test_dir / "data" / "gemv_reference.mem",
        build_dir / "gemv_reference.mem"
    )

    cocotb_test.simulator.run(
        verilog_sources=[
            str(root / "hdl" / "attention_qkt_ctrl.v"),
            str(root / "hdl" / "attention_dma_adapter.v"),
            str(root / "hdl" / "attention_gemv_control_top.v"),
            str(root / "hdl" / "attention_gemv_rom.v"),
            str(root / "hdl" / "attention_gemv_rom_top.v"),
        ],

        toplevel="attention_gemv_rom_top",
        module="attention_gemv_rom_top_tb",
        python_search=[str(test_dir)],
        sim_build=str(build_dir),

        simulator="icarus",
        force_compile=True,
        waves=True,
    )