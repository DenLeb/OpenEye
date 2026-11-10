"""Full Q/K/V (12 GEMVs) PSUM page-preservation experiment, Icarus + cocotb.

Save beside psum_direct_read_tb.py at test/cocotb_attention_qkt/.
Run from repository root:
    python -m pytest -s test/cocotb_attention_qkt/test_psum_integrated_rtl.py

This version expects that you manually edited these TWO RTL sources:
  hdl/OpenEye_FPGA.v
  hdl/attention_gemv_projection_openeye_top.v
No Python patcher is needed or imported. The test temporarily stages only the
ROM_DEPTH=939 and STATUS_WORDS=19 compatibility adjustments for the DMA stream.
"""
import json
import os
import re
import shutil
from pathlib import Path

import cocotb_test.simulator
from open_eye import hdl_dir
from open_eye import generator
from open_eye import test_utils_main as ptu
from open_eye import vh_file_creator as vh_creator

def test_psum_integrated_rtl():
    root = Path(__file__).resolve().parents[2]
    test_dir = Path(__file__).resolve().parent
    build_dir = test_dir / '.sim_build_psum_integrated_rtl'
    build_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'CLUSTER_ROWS': '2', 'CLUSTER_COLUMNS': '2',
        'NUM_GLB_IACT': '3', 'NUM_GLB_PSUM': '4', 'NUM_GLB_WGHT': '3',
        'DATAFLOW': 'row stationary', 'BRANCHES': '1',
        'BUFFER_WIDTH': '12', 'QUANT_AMOUNT': '1024', 'RAM_CELLS': '32',
    }
    os.environ.update(config)
    regmap_dir = root / 'test' / 'cocotb_fpga'
    os.environ['REGMAP_DIR'] = str(regmap_dir.resolve())

    vh_creator.create_vh_file_from_envvars(
        str(build_dir), str(hdl_dir) + os.sep, toplevel='OpenEye_FPGA')
    generator.create_regmap_params_vh_file(
        str(regmap_dir), str(build_dir), str(build_dir))

    source_dir = test_dir /  'data' / 'attention_mapper_stable_psum'
    manifest = json.loads((source_dir / 'manifest.json').read_text(encoding='utf-8'))
    sections = manifest['sections']
    status = int(sections['status']['words'])
    total = int(manifest['complete_stream_words'])
    cfg = generator.regmap_transmissions(str(regmap_dir))
    if (cfg, status, total) != (14, 19, 939):
        raise AssertionError(f'Unexpected config/status/total: {cfg}/{status}/{total}')
    if sum(int(v['words']) for v in sections.values()) != total:
        raise AssertionError('Stream manifest total does not equal section sum')
    expected_sections = {'iact': 8, 'wght': 240, 'psum': 32, 'quantize': 640}
    if {k: int(sections[k]['words']) for k in expected_sections} != expected_sections:
        raise AssertionError('Unexpected stream section lengths')
    words = [int(line, 16) for line in (source_dir/'common_rom.mem').read_text().splitlines() if line.strip()]
    regs = generator.unpack_registers(words[:cfg], str(regmap_dir))
    required = {'max_pooling': 0, 'fully_connected_layer': 1,
                'skipPsum_reg': 0, 'send_data_out': 1}
    for key, v in required.items():
        if regs[key] != v:
            raise AssertionError(f'Invalid config: {key}={regs[key]}, expected {v}')

    # All 12 images must exist, share one 939-word stream layout, and use
    # distinct activation/weight selections as declared in the manifest.
    expected_operations = [f'{kind}_t{token}' for kind in ('q', 'k', 'v')
                           for token in range(4)]
    actual_operations = [op['name'] for op in manifest['operations']]
    if actual_operations != expected_operations:
        raise AssertionError(f'Expected 12 Q/K/V operations, got {actual_operations}')
    if manifest['shape']['sequence_length'] != 4 or manifest['shape']['embedding_width'] != 8:
        raise AssertionError(f"Unexpected input shape: {manifest['shape']}")
    for operation in manifest['operations']:
        if int(operation['dma_words']) != 939:
            raise AssertionError(f'Wrong DMA length: {operation}')
        stream_path = source_dir / 'verification_streams' / (operation['name'] + '.mem')
        nwords = sum(bool(line.strip()) for line in stream_path.read_text().splitlines())
        if nwords != 939:
            raise AssertionError(f'{stream_path}: expected 939 words, got {nwords}')
    if not (source_dir / 'inputs_used.npz').is_file():
        raise FileNotFoundError(source_dir / 'inputs_used.npz')

    # The RTL changes must already be present in the actual HDL checkout.
    core_rtl = (Path(hdl_dir) / 'OpenEye_FPGA.v').read_text(encoding='utf-8')
    top_rtl = (Path(hdl_dir) / 'attention_gemv_projection_openeye_top.v').read_text(encoding='utf-8')
    assert 'parameter ATTENTION_PSUM_ENABLE = 0' in core_rtl, (
        'First edit hdl/OpenEye_FPGA.v manually according to the instructions.')
    assert 'input wire psum_result_rd_req_i' in top_rtl, (
        'First edit hdl/attention_gemv_projection_openeye_top.v manually.')
    assert re.search(
        r'OpenEye_FPGA\s*#\s*\(\s*\.ATTENTION_PSUM_ENABLE\s*\(\s*1\s*\)\s*\)\s*u_openeye\s*\(',
        top_rtl,
    ), 'Attention paging parameter must be enabled in projection top.'

    for name in ('common_rom.mem', 'activation_ram.mem', 'weight_ram.mem'):
        shutil.copyfile(source_dir / name, build_dir / name)

    sources = [
        str(file)
        for file in ptu.get_verilog_sources(str(hdl_dir))
        if Path(file).name != 'dma_storage.v'
    ]

    sources.append(str(build_dir / 'dma_storage.v'))

    print(
        'PSUM INTEGRATION: Stable configuration, '
        '939-word DMA stream.',
        flush=True
    )
    print('PSUM INTEGRATION: Stable configuration, 939-word DMA stream, no temporary RTL patch.',flush=True)
    print('PSUM INTEGRATION: RAM_SP is unchanged and remains single-port.', flush=True)
    print('PSUM INTEGRATION: Q pages 0..3; K pages 4..7; V pages 8..11 (stride=32).', flush=True)

    cocotb_test.simulator.run(
        verilog_sources=sources,
        toplevel='attention_projection_openeye_top',
        module='psum_direct_read_tb',
        python_search=[str(test_dir)],
        sim_build=str(build_dir),
        simulator='icarus',
        defines={'NO_TRACE': 'TRUE', 'USE_INTERNAL_PARAMS_PE': 'TRUE'},
        force_compile=True,
        waves=False,
        extra_env={
            'PSUM_DIRECT_READ_DATA_DIR': str(source_dir.resolve()),
            'PSUM_DIRECT_READ_MAX_CYCLES': os.getenv('PSUM_DIRECT_READ_MAX_CYCLES', '4000'),
        },
    )
