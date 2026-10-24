"""Run: python -m pytest -s test/cocotb_attention_qkt/test_attention_projection_memory.py

Place attention_projection_memory.v in hdl/ and
attention_projection_memory_tb.py in test/cocotb_attention_qkt/.
The Python Mapper output must exist at data/attention_mapper_v1/.
"""

import os
import shutil
from pathlib import Path

import pytest
from cocotb_test.simulator import run

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data" / "attention_mapper_v1"


@pytest.mark.parametrize("simulator", ["icarus"])
def test_attention_projection_memory(simulator):
    build_dir = TEST_DIR / ".sim_build_projection_memory"
    build_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("common_rom.mem", "activation_ram.mem", "weight_ram.mem"):
        source = DATA_DIR / filename
        assert source.exists(), f"Missing {source}; run the Python Mapper first"
        shutil.copyfile(source, build_dir / filename)

    run(
        simulator=simulator,
        verilog_sources=[str(ROOT / "hdl" / "attention_projection_memory.v")],
        toplevel="attention_projection_memory",
        module="attention_projection_memory_tb",
        python_search=[str(TEST_DIR)],
        sim_build=str(build_dir),
        extra_env={"ATTENTION_MEMORY_DATA_DIR": str(DATA_DIR)},
        force_compile=True,
    )
