from pathlib import Path
import cocotb_test.simulator


def test_attention_dma_adapter():

    root = Path(__file__).resolve().parents[2]
    test_dir = Path(__file__).resolve().parent

    cocotb_test.simulator.run(
        verilog_sources=[
            str(root / "hdl" / "attention_dma_adapter.v")
        ],
        toplevel="attention_dma_adapter",
        module="attention_dma_adapter_tb",
        python_search=[str(test_dir)],
        sim_build=str(test_dir / ".sim_build_dma"),
        simulator="icarus",
        force_compile=True,
        waves=True,
    )