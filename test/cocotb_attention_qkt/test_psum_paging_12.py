"""Full Q/K/V (12 GEMVs) PSUM page-preservation experiment, Icarus + cocotb.

Save beside psum_paging_12_tb.py at test/cocotb_attention_qkt/.
Run from repository root:
    python -m pytest -s test/cocotb_attention_qkt/test_psum_paging_12.py

All RTL modifications are staged inside .sim_build_psum_paging_12/. The normal
OpenEye sources, verified tests, RAM_SP, and QKV buffer are not modified.
The temporary RAM readback uses THE EXISTING single RAM port, only while idle.
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


def once(text, needle, replacement, name):
    n = text.count(needle)
    if n != 1:
        raise AssertionError(f"{name}: expected one anchor, found {n}: {needle[:90]}")
    return text.replace(needle, replacement, 1)


def patch_fpga(source: str) -> str:
    source = once(source,
        '    output reg last_data_o\n\n);',
        '''    output reg last_data_o,
    // TEST-ONLY: page selection and idle single-port readback; original RTL unchanged.
    input wire [3:0] psum_test_page_i,
    input wire psum_test_debug_read_i,
    input wire [BUFFER_WIDTH_PSUM-1:0] psum_test_debug_addr_i,
    output wire [PSUM_BUFFER_WIDTH-1:0] psum_test_debug_data_o

);''', 'OpenEye_FPGA test interface')
    source = once(source,
        '  wire rst_n; // Synchronized, active-low reset; driven by RST_SYNC below.',
        '''  wire rst_n; // Synchronized, active-low reset; driven by RST_SYNC below.
  // Address offset 32 * page. The local PSUM FSM address remains unchanged.
  wire [BUFFER_WIDTH_PSUM-1:0] psum_test_page_offset =
      {psum_test_page_i, 5'b0};''', 'page offset')
    # Observe actual RAM read data. Bank 0 is col=0,row=0,pair=0.
    pat = r'(wire\s+\[TRANS_BITWIDTH_PSUM\*CLUSTERS\*NUM_GLB_PSUM-1:0\]\s+psum_buffer_data_r\s*;[^\n]*)'
    source, n = re.subn(pat,
        r'\1\n  assign psum_test_debug_data_o = psum_buffer_data_r[0+:PSUM_BUFFER_WIDTH];',
        source, count=1)
    if n != 1:
        raise AssertionError('Cannot locate PSUM RAM read-data bus')
    # Change ONLY the physical RAM_SP instance wiring. The PSUM FSM and
    # original local-address bus are untouched.
    start = source.index('begin : PSUM_RAM_X')
    end = source.index('    /*assign debug_psum_re', start)
    block = source[start:end]
    old_rd = '.rd_en_i(psum_buffer_en_r[i_gen*((NUM_GLB_PSUM+1)/2)+j_gen*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen] & !psum_buffer_en_w[i_gen*((NUM_GLB_PSUM+1)/2)+j_gen*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen]),'
    new_rd = ".rd_en_i(psum_test_debug_read_i ? ((i_gen == 0) && (j_gen == 0) && (g_gen == 0)) : (psum_buffer_en_r[i_gen*((NUM_GLB_PSUM+1)/2)+j_gen*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen] & !psum_buffer_en_w[i_gen*((NUM_GLB_PSUM+1)/2)+j_gen*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen])),"
    block = once(block, old_rd, new_rd, 'PSUM RAM read enable')
    old_wr = '.wr_en_i(psum_buffer_en_w[i_gen*((NUM_GLB_PSUM+1)/2)+j_gen*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen]),'
    new_wr = ".wr_en_i(psum_test_debug_read_i ? 1'b0 : psum_buffer_en_w[i_gen*((NUM_GLB_PSUM+1)/2)+j_gen*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen]),"
    block = once(block, old_wr, new_wr, 'PSUM RAM write enable')
    old_addr = '.addr_i (psum_buffer_addr[i_gen*BUFFER_WIDTH_PSUM*((NUM_GLB_PSUM+1)/2)+j_gen*BUFFER_WIDTH_PSUM*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen*BUFFER_WIDTH_PSUM+:BUFFER_WIDTH_PSUM]),'
    new_addr = '''.addr_i (psum_test_debug_read_i ? psum_test_debug_addr_i :
                  (psum_buffer_addr[i_gen*BUFFER_WIDTH_PSUM*((NUM_GLB_PSUM+1)/2)+j_gen*BUFFER_WIDTH_PSUM*CLUSTER_COLUMNS*((NUM_GLB_PSUM+1)/2)+g_gen*BUFFER_WIDTH_PSUM+:BUFFER_WIDTH_PSUM] + psum_test_page_offset)),'''
    block = once(block, old_addr, new_addr, 'PSUM RAM physical address')
    source = source[:start] + block + source[end:]
    return source


def patch_top(source: str) -> str:
    source = once(source,
        'parameter integer ROM_DEPTH  = 939',
        'parameter integer ROM_DEPTH  = 942', 'top 942 words')
    source = once(source,
        '    output wire [ADDR_WIDTH-1:0] stream_addr_o\n);',
        '''    output wire [ADDR_WIDTH-1:0] stream_addr_o,
    // TEST-ONLY idle readback port; no extra physical RAM port is added.
    input wire psum_test_debug_read_i,
    input wire [13:0] psum_test_debug_addr_i,
    output wire [39:0] psum_test_debug_data_o
);''', 'top test readback ports')
    source = once(source,
        '    //==========================================================\n    // Real OpenEye accelerator\n    //==========================================================',
        '''    // TEST-ONLY: latch page ID on the GEMV start edge; must not track
    // changing live selectors while the input stream is being transferred.
    reg [1:0] psum_test_token_reg;
    reg [1:0] psum_test_projection_reg;
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            psum_test_token_reg <= 0;
            psum_test_projection_reg <= 0;
        end else if (projection_start) begin
            psum_test_token_reg <= token_select_i;
            psum_test_projection_reg <= projection_select_i;
        end
    end
    wire [3:0] psum_test_page_id =
        {psum_test_projection_reg, psum_test_token_reg};

    //==========================================================
    // Real OpenEye accelerator
    //==========================================================''', 'top page register')
    source = once(source,
        '    OpenEye_FPGA u_openeye (',
        '''    OpenEye_FPGA u_openeye (
        .psum_test_page_i(psum_test_page_id),
        .psum_test_debug_read_i(psum_test_debug_read_i),
        .psum_test_debug_addr_i(psum_test_debug_addr_i),
        .psum_test_debug_data_o(psum_test_debug_data_o),''', 'top FPGA test connections')
    return source


def test_psum_paging_12():
    root = Path(__file__).resolve().parents[2]
    test_dir = Path(__file__).resolve().parent
    build_dir = test_dir / '.sim_build_psum_paging_12'
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

    source_dir = test_dir / 'data' / 'attention_mapper_v1_fixed'
    manifest = json.loads((source_dir / 'manifest.json').read_text(encoding='utf-8'))
    sections = manifest['sections']
    status = int(sections['status']['words'])
    total = int(manifest['complete_stream_words'])
    cfg = generator.regmap_transmissions(str(regmap_dir))
    if (cfg, status, total) != (17, 22, 942):
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

    # All 12 images must exist, share one 942-word stream layout, and use
    # distinct activation/weight selections as declared in the manifest.
    expected_operations = [f'{kind}_t{token}' for kind in ('q', 'k', 'v')
                           for token in range(4)]
    actual_operations = [op['name'] for op in manifest['operations']]
    if actual_operations != expected_operations:
        raise AssertionError(f'Expected 12 Q/K/V operations, got {actual_operations}')
    if manifest['shape']['sequence_length'] != 4 or manifest['shape']['embedding_width'] != 8:
        raise AssertionError(f"Unexpected input shape: {manifest['shape']}")
    for operation in manifest['operations']:
        if int(operation['dma_words']) != 942:
            raise AssertionError(f'Wrong DMA length: {operation}')
        stream_path = source_dir / 'verification_streams' / (operation['name'] + '.mem')
        nwords = sum(bool(line.strip()) for line in stream_path.read_text().splitlines())
        if nwords != 942:
            raise AssertionError(f'{stream_path}: expected 942 words, got {nwords}')
    if not (source_dir / 'inputs_used.npz').is_file():
        raise FileNotFoundError(source_dir / 'inputs_used.npz')

    for name in ('common_rom.mem', 'activation_ram.mem', 'weight_ram.mem'):
        shutil.copyfile(source_dir / name, build_dir / name)

    replacements = {
        'attention_projection_memory.v': lambda text: once(text,
            'parameter integer STATUS_WORDS       = 19,',
            'parameter integer STATUS_WORDS       = 22,', 'memory status size'),
        'attention_gemv_projection_openeye_top.v': patch_top,
        'OpenEye_FPGA.v': patch_fpga,
    }
    sources = []
    patched = set()
    for file in ptu.get_verilog_sources(str(hdl_dir)):
        path = Path(file)
        if path.name == 'dma_storage.v':
            continue
        if path.name in replacements:
            staged = build_dir / path.name
            original = path.read_text(encoding='utf-8')
            changed = replacements[path.name](original)
            if changed == original:
                raise AssertionError(f'Staging made no change to {path.name}')
            staged.write_text(changed, encoding='utf-8')
            sources.append(str(staged))
            patched.add(path.name)
        else:
            sources.append(file)
    if patched != set(replacements):
        raise AssertionError(f'Not all required RTL files found: {set(replacements) - patched}')
    sources.append(str(build_dir / 'dma_storage.v'))
    print('PSUM PAGING 12: staged 3 temporary RTL source files only.', flush=True)
    print('PSUM PAGING 12: RAM_SP is unchanged and remains single-port.', flush=True)
    print('PSUM PAGING 12: Q pages 0..3; K pages 4..7; V pages 8..11 (stride=32).', flush=True)

    cocotb_test.simulator.run(
        verilog_sources=sources,
        toplevel='attention_projection_openeye_top',
        module='psum_paging_12_tb',
        python_search=[str(test_dir)],
        sim_build=str(build_dir),
        simulator='icarus',
        defines={'NO_TRACE': 'TRUE', 'USE_INTERNAL_PARAMS_PE': 'TRUE'},
        force_compile=True,
        waves=False,
        extra_env={
            'PSUM_PAGING_12_DATA_DIR': str(source_dir.resolve()),
            'PSUM_PAGING_12_MAX_CYCLES': os.getenv('PSUM_PAGING_12_MAX_CYCLES', '4000'),
        },
    )
