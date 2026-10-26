"""One and only one Cocotb/Pytest case: prepared 8x4 Row-Stationary QKT."""

import os
from pathlib import Path

import cocotb_test.simulator

import open_eye.generator as generator
import open_eye.test_utils_main as ptu
import open_eye.vh_file_creator as vh_file_creator
from open_eye import hdl_dir, test_dir


def test_qkt_single_openeye():
    # Freeze the exact environment of the successful K8-N4 row_stationary test.
    settings = {
        "PARALLEL_MACS": "2", "CLUSTER_ROWS": "2", "CLUSTER_COLUMNS": "2",
        "NUM_GLB_IACT": "3", "NUM_GLB_PSUM": "4", "NUM_GLB_WGHT": "3",
        "DATAFLOW": "row_stationary", "BRANCHES": "1", "BUFFER_WIDTH": "12",
        "QUANT_AMOUNT": "1024", "RAM_CELLS": "32",
    }
    os.environ.update(settings)

    this_dir = Path(__file__).resolve().parent
    image_dir = this_dir / "data" / "qkt_single_8x4"
    for filename in ("qkt_t0_dma.mem", "qkt_t0_reference.npz"):
        assert (image_dir / filename).is_file(), f"Missing prepared file: {image_dir / filename}"

    build_dir = Path(test_dir) / ".temp" / "qkt_single_row_stationary_k8_n4"
    build_dir.mkdir(parents=True, exist_ok=True)

    vh_file_creator.create_vh_file_from_envvars(
        str(build_dir), hdl_dir + "/", toplevel="OpenEye_FPGA"
    )
    generator.create_regmap_params_vh_file(
        os.path.join(test_dir, "cocotb_fpga"), str(build_dir), str(build_dir)
    )

    sources = [source for source in ptu.get_verilog_sources(hdl_dir)
               if os.path.basename(source) != "dma_storage.v"]
    sources.append(str(build_dir / "dma_storage.v"))

    cocotb_test.simulator.run(
        python_search=[str(this_dir)],
        verilog_sources=sources,
        toplevel="OpenEye_FPGA",
        module="qkt_single_openeye_tb",
        testcase="test_qkt_single_gemv",
        sim_build=str(build_dir),
        defines={"NO_TRACE": "TRUE", "USE_INTERNAL_PARAMS_PE": "TRUE"},
        simulator="icarus",
        force_compile=True,
        waves=False,
        extra_env={
            **settings,
            "QKT_SINGLE_DATA_DIR": str(image_dir.resolve()),
            "CLOCK_LEN": "20", "CLOCK_UNIT": "ns",
            "CLOCK_DELAY_INPUT": "100", "CLOCK_DELAY_UNIT_INPUT": "ps",
            "CLOCK_DELAY_OUTPUT": "100", "CLOCK_DELAY_UNIT_OUTPUT": "ps",
            "LOGGER_LEVEL": "20",
        },
    )
