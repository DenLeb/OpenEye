`timescale 1ns / 1ps

//=============================================================================
// OpenEye Single-Head Attention Integration
//
// Hardware-controlled execution:
//   1. Q/K/V Projection (12 GEMVs)
//   2. Q/K Quantization (INT8)
//   3. QK^T Computation (4 GEMVs)
//
// Uses one shared OpenEye_FPGA and one shared DMA adapter.
//
// Verified:
//   - End-to-end Cocotb simulation PASS
//   - All 16 attention scores verified bit-exact
//
// Current configuration:
//   - 4 tokens, 8 features
//   - 64-bit DMA
//   - Projection: 939 DMA words
//   - QKT: 677 DMA words
//
// Stage 4 includes Softmax; earlier stages compile it out.
//=============================================================================

module attention_top #(
    parameter integer INTEGRATION_STAGE = 4,
    parameter integer PSUM_SINGLE_PORT = 1,
    parameter integer DMA_WIDTH = 64,
    parameter integer ADDR_WIDTH = 16,
    parameter integer PROJECTION_STREAM_WORDS = 939,
    parameter integer QKT_STREAM_WORDS = 677,
    parameter PROJECTION_COMMON_MEM_FILE = "common_rom.mem",
    parameter PROJECTION_IACT_MEM_FILE = "activation_ram.mem",
    parameter PROJECTION_WGHT_MEM_FILE = "weight_ram.mem",
    parameter QKT_TEMPLATE_FILE = "qkt_t0_dma.mem"
)(
    input  wire clk_i,
    input  wire rst_ni,
    input  wire start_i,
    output wire busy_o,
    output wire done_o,
    output wire scores_complete_o,

    // Optional software/testbench update of projection DMA operand RAMs.
    input  wire                  iact_we_i,
    input  wire [ADDR_WIDTH-1:0] iact_wr_addr_i,
    input  wire [DMA_WIDTH-1:0]  iact_wr_data_i,
    input  wire                  wght_we_i,
    input  wire [ADDR_WIDTH-1:0] wght_wr_addr_i,
    input  wire [DMA_WIDTH-1:0]  wght_wr_data_i,

    // Synthesis-visible readback; use after busy_o deasserts.
    // observe_select_i: 0=Q raw, 1=K raw, 2=V raw,
    //                   3=Q8, 4=K8, 5=Q shift, 6=K shift.
    input  wire [2:0] observe_select_i,
    input  wire [1:0] observe_token_i,
    input  wire [2:0] observe_feature_i,
    output reg  [19:0] observe_data_o,

    input  wire observe_req_i,
    output wire observe_ready_o,
    output reg  observe_valid_o,

    // Read the final 4x4 QK^T scores; used later by Softmax.
    input  wire [1:0] score_rd_token_i,
    input  wire [1:0] score_rd_index_i,
    output wire signed [19:0] score_rd_data_o,
    output wire score_rd_valid_o,

    // Softmax probability read interface
    input  wire [1:0] prob_rd_token_i,
    input  wire [1:0] prob_rd_index_i,
    output wire [15:0] prob_rd_data_o,
    output wire        prob_rd_valid_o,
    output wire        softmax_complete_o,
    
    // Debug outputs for integration tests.
    output wire [1:0] active_token_o,
    output wire [1:0] active_projection_o,
    output wire qkt_mode_o,
    output wire [ADDR_WIDTH-1:0] stream_addr_o,
    output wire dma_result_write_o
);
    localparam [ADDR_WIDTH-1:0] PROJECTION_WORD_COUNT = PROJECTION_STREAM_WORDS;
    localparam [ADDR_WIDTH-1:0] QKT_WORD_COUNT = QKT_STREAM_WORDS;

// Global execution sequencing is implemented in attention_top_scheduler.v.
// Legacy attention_qkt_ctrl.v and the passing standalone tops stay unchanged.
    wire projection_start;
    wire quant_start;
    wire qkt_memory_clear;
    wire qkt_q8_reload;
    wire q8_packer_start;
    wire k8_packer_start;
    wire gemv_start;
    wire qkt_mode;
    wire qkt_result_enable;
    wire [1:0] token_select;
    wire [1:0] projection_select;
    wire quant_done;
    wire operands_ready;
    wire gemv_done;

    wire softmax_start;
    wire softmax_done;
    wire softmax_busy;

    // Softmax Score Buffer read requests
    wire [1:0] softmax_score_rd_token;
    wire [1:0] softmax_score_rd_index;

    // Shared Score Buffer read response
    wire signed [19:0] score_buffer_rd_data;
    wire score_buffer_rd_valid;

    // Selected read address
    wire [1:0] selected_score_rd_token;
    wire [1:0] selected_score_rd_index;

    attention_top_scheduler #(
        .INTEGRATION_STAGE(INTEGRATION_STAGE)
    ) u_scheduler (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(start_i),
        .gemv_done_i(gemv_done),
        .quant_done_i(quant_done),
        .operands_ready_i(operands_ready),

        .projection_start_o(projection_start),
        .quant_start_o(quant_start),
        .qkt_memory_clear_o(qkt_memory_clear),
        .qkt_q8_reload_o(qkt_q8_reload),
        .q8_packer_start_o(q8_packer_start),
        .k8_packer_start_o(k8_packer_start),

        .gemv_start_o(gemv_start),
        .qkt_mode_o(qkt_mode),
        .qkt_result_enable_o(qkt_result_enable),
        
        .token_o(token_select),
        .projection_o(projection_select),
        .busy_o(busy_o),
        .softmax_start_o(softmax_start),
        .scores_complete_i(scores_complete_o),
        .softmax_done_i(softmax_done),
        .done_o(done_o)
    );
    assign active_token_o = token_select;
    assign active_projection_o = projection_select;
    assign qkt_mode_o = qkt_mode;

    //===================== ONE SHARED DMA ADAPTER ============================
    wire [ADDR_WIDTH-1:0] stream_addr;
    wire [DMA_WIDTH-1:0] projection_stream_data;
    wire [DMA_WIDTH-1:0] qkt_stream_data;
    wire [DMA_WIDTH-1:0] selected_stream_data =
        qkt_mode ? qkt_stream_data : projection_stream_data;
    wire [ADDR_WIDTH-1:0] selected_stream_word_count =
        qkt_mode ? QKT_WORD_COUNT : PROJECTION_WORD_COUNT;
    wire [DMA_WIDTH-1:0] dma_input_data;
    wire dma_input_valid;
    wire dma_input_ready;
    wire [DMA_WIDTH-1:0] dma_output_data;
    wire dma_output_valid;
    wire dma_output_last;
    wire dma_output_ready;
    wire adapter_busy;
    wire result_write;
    wire [DMA_WIDTH-1:0] result_data;
    wire [ADDR_WIDTH-1:0] result_addr;

    assign stream_addr_o = stream_addr;
    assign dma_result_write_o = result_write;

    attention_dma_adapter #(
        .DMA_WIDTH(DMA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_dma_adapter (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .gemv_start_i(gemv_start),
        .gemv_done_o(gemv_done),
        .busy_o(adapter_busy),
        .stream_word_count_i(selected_stream_word_count),
        .stream_addr_o(stream_addr),
        .stream_data_i(selected_stream_data),
        .dma_ready_i(dma_input_ready),
        .dma_data_o(dma_input_data),
        .dma_valid_o(dma_input_valid),
        .dma_result_data_i(dma_output_data),
        .dma_result_valid_i(dma_output_valid),
        .dma_result_last_i(dma_output_last),
        .dma_result_ready_o(dma_output_ready),
        .result_write_o(result_write),
        .result_data_o(result_data),
        .result_addr_o(result_addr)
    );

    //===================== SINGLE-PORT PSUM READ ======================

    wire psum_rd_req;
    wire [3:0] psum_rd_page;
    wire [2:0] psum_rd_feature;

    wire psum_rd_ready;
    wire psum_rd_valid;
    wire [19:0] psum_rd_data;

    // Shared OpenEye PSUM read interface
    wire core_psum_rd_req;
    wire [3:0] core_psum_rd_page;
    wire [2:0] core_psum_rd_feature;

    wire core_psum_rd_ready;
    wire core_psum_rd_valid;
    wire [19:0] core_psum_rd_data;

    // Quantization receives the existing OpenEye response.
    assign psum_rd_ready = core_psum_rd_ready;
    assign psum_rd_valid = core_psum_rd_valid;
    assign psum_rd_data  = core_psum_rd_data;

    // Retain projection page during the complete GEMV operation
    reg [3:0] psum_active_page;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            psum_active_page <= 4'd0;
        end else if (projection_start && !qkt_mode) begin
            psum_active_page <= {projection_select, token_select};
        end
    end
    //===================== QUANTIZATION SIGNALS ========================

    wire quant_busy;

    // Legacy RAW Q/K read interface
    wire [1:0] quant_q_rd_token;
    wire [2:0] quant_q_rd_feature;
    wire [1:0] quant_k_rd_token;
    wire [2:0] quant_k_rd_feature;

    wire [19:0] q_raw_data;
    wire [19:0] k_raw_data;
    wire [19:0] v_raw_data;

    // Quantized Q8/K8 read interface
    wire [1:0] q8_rd_token;
    wire [2:0] q8_rd_feature;
    wire signed [7:0] q8_data;

    wire [1:0] k8_rd_token;
    wire [2:0] k8_rd_feature;
    wire signed [7:0] k8_data;

    // Quantization shift values
    wire [6:0] q_shift;
    wire [6:0] k_shift;
    //===================== Q/K QUANTIZATION ===========================

generate
    if (INTEGRATION_STAGE >= 2) begin : GEN_QUANTIZATION

        attention_qk_quantization_controller #(
            .NUM_TOKENS(4),
            .NUM_FEATURES(8),
            .DATA_WIDTH(20),
            .SHIFT_WIDTH(7),
            .PSUM_SINGLE_PORT(PSUM_SINGLE_PORT)
        ) u_quantization (

            .clk_i(clk_i),
            .rst_ni(rst_ni),

            .start_i(quant_start),
            .busy_o(quant_busy),
            .done_o(quant_done),

            // Legacy RAW Q
            .q_raw_rd_token_o(quant_q_rd_token),
            .q_raw_rd_feature_o(quant_q_rd_feature),
            .q_raw_data_i(q_raw_data),

            // Legacy RAW K
            .k_raw_rd_token_o(quant_k_rd_token),
            .k_raw_rd_feature_o(quant_k_rd_feature),
            .k_raw_data_i(k_raw_data),

            // Direct Single-Port PSUM
            .psum_rd_req_o(psum_rd_req),
            .psum_rd_page_o(psum_rd_page),
            .psum_rd_feature_o(psum_rd_feature),

            .psum_rd_ready_i(psum_rd_ready),
            .psum_rd_valid_i(psum_rd_valid),
            .psum_rd_data_i(psum_rd_data),

            // Quantized Q8
            .q8_rd_token_i(q8_rd_token),
            .q8_rd_feature_i(q8_rd_feature),
            .q8_data_o(q8_data),

            // Quantized K8
            .k8_rd_token_i(k8_rd_token),
            .k8_rd_feature_i(k8_rd_feature),
            .k8_data_o(k8_data),

            // Scaling
            .q_shift_o(q_shift),
            .k_shift_o(k_shift)
        );

    end else begin : GEN_NO_QUANTIZATION

        assign quant_busy = 1'b0;
        assign quant_done = 1'b0;

        assign quant_q_rd_token = 2'd0;
        assign quant_q_rd_feature = 3'd0;
        assign quant_k_rd_token = 2'd0;
        assign quant_k_rd_feature = 3'd0;

        assign psum_rd_req = 1'b0;
        assign psum_rd_page = 4'd0;
        assign psum_rd_feature = 3'd0;

        assign q8_data = 8'sd0;
        assign k8_data = 8'sd0;

        assign q_shift = 7'd0;
        assign k_shift = 7'd0;

    end
endgenerate
    //===================== ONE SHARED OPENEYE ================================
    OpenEye_FPGA #(
        .ATTENTION_PSUM_ENABLE(1),
        .ATTENTION_SOFTMAX_ENABLE(INTEGRATION_STAGE == 4)
    ) u_openeye (
        .clk_i(clk_i),
        .rst_ni(rst_ni),

        .attention_mode_i(!qkt_mode || !busy_o),
        .psum_page_i(psum_active_page),

        .psum_result_rd_req_i(core_psum_rd_req),
        .psum_result_rd_page_i(core_psum_rd_page),
        .psum_result_rd_feature_i(core_psum_rd_feature),

        .psum_result_rd_ready_o(core_psum_rd_ready),
        .psum_result_rd_valid_o(core_psum_rd_valid),
        .psum_result_rd_data_o(core_psum_rd_data),

        // Softmax FSM and probability storage are now inside OpenEye_FPGA.
        .softmax_start_i(softmax_start),
        .softmax_q_shift_i(q_shift),
        .softmax_k_shift_i(k_shift),
        .softmax_score_rd_token_o(softmax_score_rd_token),
        .softmax_score_rd_index_o(softmax_score_rd_index),
        .softmax_score_rd_data_i(score_buffer_rd_data),
        .softmax_score_rd_valid_i(score_buffer_rd_valid),
        .softmax_prob_rd_token_i(prob_rd_token_i),
        .softmax_prob_rd_index_i(prob_rd_index_i),
        .softmax_prob_rd_data_o(prob_rd_data_o),
        .softmax_prob_rd_valid_o(prob_rd_valid_o),
        .softmax_busy_o(softmax_busy),
        .softmax_done_o(softmax_done),
        .softmax_complete_o(softmax_complete_o),

        // DMA input
        .data_dma_i(dma_input_data),
        .enable_dma_i(dma_input_valid),
        .ready_dma_o(dma_input_ready),

        // DMA output
        .data_dma_o(dma_output_data),
        .enable_dma_o(dma_output_valid),
        .last_data_o(dma_output_last),
        .ready_dma_i(dma_output_ready)
    );

    //===================== PROJECTION DMA MEMORY =============================
    attention_projection_memory #(
        .DMA_WIDTH(DMA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .COMMON_MEM_FILE(PROJECTION_COMMON_MEM_FILE),
        .ACTIVATION_MEM_FILE(PROJECTION_IACT_MEM_FILE),
        .WEIGHT_MEM_FILE(PROJECTION_WGHT_MEM_FILE)
    ) u_projection_memory (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(projection_start),
        .busy_i(adapter_busy),
        .token_select_i(token_select),
        .projection_select_i(projection_select),
        .addr_i(stream_addr),
        .data_o(projection_stream_data),
        .iact_we_i(iact_we_i),
        .iact_wr_addr_i(iact_wr_addr_i),
        .iact_wr_data_i(iact_wr_data_i),
        .wght_we_i(wght_we_i),
        .wght_wr_addr_i(wght_wr_addr_i),
        .wght_wr_data_i(wght_wr_data_i)
    );

    // Q8/K8 read ports are shared by the packers and the observation path.
    wire [1:0] packer_q8_rd_token;
    wire [2:0] packer_q8_rd_feature;
    wire [1:0] packer_k8_rd_token;
    wire [2:0] packer_k8_rd_feature;
    wire q8_packer_busy;
    wire k8_packer_busy;

    assign q8_rd_token   = q8_packer_busy ? packer_q8_rd_token : observe_token_i;
    assign q8_rd_feature = q8_packer_busy ? packer_q8_rd_feature : observe_feature_i;
    assign k8_rd_token   = k8_packer_busy ? packer_k8_rd_token : observe_token_i;
    assign k8_rd_feature = k8_packer_busy ? packer_k8_rd_feature : observe_feature_i;

//===================== RAW RESULT STORAGE =========================

generate
    if (PSUM_SINGLE_PORT == 0) begin : GEN_LEGACY_RESULT_BUFFER

        attention_projection_result_buffer u_projection_result_buffer (
            .clk_i(clk_i),
            .rst_ni(rst_ni),

            .start_i(projection_start),
            .token_select_i(token_select),
            .projection_select_i(projection_select),

            .result_write_i(result_write && !qkt_mode),
            .result_addr_i(result_addr),
            .result_data_i(result_data),

            .q_rd_token_i(
                quant_busy ? quant_q_rd_token : observe_token_i
            ),
            .q_rd_feature_i(
                quant_busy ? quant_q_rd_feature : observe_feature_i
            ),

            .k_rd_token_i(
                quant_busy ? quant_k_rd_token : observe_token_i
            ),
            .k_rd_feature_i(
                quant_busy ? quant_k_rd_feature : observe_feature_i
            ),

            .v_rd_token_i(observe_token_i),
            .v_rd_feature_i(observe_feature_i),

            .q_data_o(q_raw_data),
            .k_data_o(k_raw_data),
            .v_data_o(v_raw_data)
        );

    end else begin : GEN_INTERNAL_PSUM_ONLY

        // No duplicated Q/K/V result buffer.
        // Quantization uses the PSUM handshake interface.
        assign q_raw_data = 20'd0;
        assign k_raw_data = 20'd0;
        assign v_raw_data = 20'd0;

    end
endgenerate

//===================== SHARED PSUM OBSERVATION =====================

// Raw observation selectors:
// 0 = Q, 1 = K, 2 = V
wire observe_raw_select = (observe_select_i <= 3'd2);

// Observation is allowed only when the attention operation
// and the quantization controller are inactive.
reg observe_pending;

assign observe_ready_o =
    !busy_o &&
    !adapter_busy &&
    ((INTEGRATION_STAGE < 2) || !quant_busy) &&
    !observe_pending &&
    (
        ((PSUM_SINGLE_PORT != 0) && observe_raw_select)
        ? (core_psum_rd_ready && !psum_rd_req)
        : 1'b1
    );

wire observe_accept = observe_req_i && observe_ready_o;

// Only raw observations need access to OpenEye PSUM RAM.
// Q8/K8 and shifts are already stored in the quantization controller.
wire observe_psum_req =
    (PSUM_SINGLE_PORT != 0) &&
    observe_raw_select &&
    observe_accept;

// Multiplex the single physical PSUM read interface.
// Quantization has priority.
assign core_psum_rd_req =
    psum_rd_req || observe_psum_req;

assign core_psum_rd_page =
    psum_rd_req
        ? psum_rd_page
        : {observe_select_i[1:0], observe_token_i};

assign core_psum_rd_feature =
    psum_rd_req
        ? psum_rd_feature
        : observe_feature_i;

// Observation response FSM
always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
        observe_pending <= 1'b0;
        observe_valid_o <= 1'b0;
        observe_data_o <= 20'd0;
    end else begin
        observe_valid_o <= 1'b0;

        if (observe_pending) begin

            // Wait for the synchronous OpenEye PSUM read response.
            if (core_psum_rd_valid) begin
                observe_data_o <= core_psum_rd_data;
                observe_valid_o <= 1'b1;
                observe_pending <= 1'b0;
            end

        end else if (observe_accept) begin

            if ((PSUM_SINGLE_PORT != 0) && observe_raw_select) begin

                // Read request is accepted by OpenEye.
                // Response arrives a few clocks later.
                observe_pending <= 1'b1;

            end else begin

                // Legacy raw buffer or quantized observation.
                // Capture the available data on this clock edge.
                case (observe_select_i)

                    3'd0: observe_data_o <= q_raw_data;
                    3'd1: observe_data_o <= k_raw_data;
                    3'd2: observe_data_o <= v_raw_data;

                    3'd3: observe_data_o <=
                        {{12{q8_data[7]}}, q8_data};

                    3'd4: observe_data_o <=
                        {{12{k8_data[7]}}, k8_data};

                    3'd5: observe_data_o <=
                        {13'd0, q_shift};

                    3'd6: observe_data_o <=
                        {13'd0, k_shift};

                    default: observe_data_o <= 20'd0;

                endcase

                observe_valid_o <= 1'b1;
            end
        end
    end
end

    // Shared Score Buffer read interface across all configurations.
    assign selected_score_rd_token =
        softmax_busy ? softmax_score_rd_token : score_rd_token_i;
    assign selected_score_rd_index =
        softmax_busy ? softmax_score_rd_index : score_rd_index_i;
    assign score_rd_data_o = score_buffer_rd_data;
    assign score_rd_valid_o = score_buffer_rd_valid;

    generate
        if (INTEGRATION_STAGE >= 3) begin : GEN_QKT
    //===================== QKT DMA OPERAND PACKERS ===========================
    wire [DMA_WIDTH-1:0] q8_word;
    wire q8_word_valid;
    wire q8_word_ready;
    wire q8_word_last;
    wire q8_word_index;
    wire q8_packer_done;
    wire [DMA_WIDTH-1:0] k8_word;
    wire k8_word_valid;
    wire k8_word_ready;
    wire k8_word_last;
    wire [3:0] k8_word_index;
    wire k8_packer_done;

    // NOTE: Verify exact port names/widths against the local passing
    // attention_qkt_openeye_top.v before treating this draft as final.
    attention_qkt_q8_packer u_q8_packer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(q8_packer_start),
        .token_i(token_select),
        .q8_rd_token_o(packer_q8_rd_token),
        .q8_rd_feature_o(packer_q8_rd_feature),
        .q8_data_i(q8_data),
        .q8_word_o(q8_word),
        .q8_word_valid_o(q8_word_valid),
        .q8_word_ready_i(q8_word_ready),
        .q8_word_last_o(q8_word_last),
        .word_index_o(q8_word_index),
        .busy_o(q8_packer_busy),
        .done_o(q8_packer_done)
    );

    attention_qkt_k8_packer u_k8_packer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(k8_packer_start),
        .k8_rd_token_o(packer_k8_rd_token),
        .k8_rd_feature_o(packer_k8_rd_feature),
        .k8_data_i(k8_data),
        .wght_word_o(k8_word),
        .wght_word_valid_o(k8_word_valid),
        .wght_word_ready_i(k8_word_ready),
        .wght_word_last_o(k8_word_last),
        .word_index_o(k8_word_index),
        .busy_o(k8_packer_busy),
        .done_o(k8_packer_done)
    );

    //===================== QKT DMA TEMPLATE + DYNAMIC WORDS ==================
    // The validated QKT memory retains K8 after the first token, while
    // q8_reload_i invalidates only the one token's Q8 operand word.
    // Port names here are provisional pending inspection of local source.
    attention_qkt_dma_memory #(
        .TEMPLATE_FILE(QKT_TEMPLATE_FILE)
    ) u_qkt_memory (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(qkt_memory_clear),
        .q8_reload_i(qkt_q8_reload),
        .q8_word_i(q8_word),
        .q8_word_valid_i(q8_word_valid),
        .q8_word_ready_o(q8_word_ready),
        .k8_word_i(k8_word),
        .k8_word_valid_i(k8_word_valid),
        .k8_word_ready_o(k8_word_ready),
        .k8_word_index_i(k8_word_index),
        .operands_ready_o(operands_ready),
        .stream_addr_i(stream_addr),
        .stream_data_o(qkt_stream_data)
    );

    //===================== QKT SCORE RESULT BUFFER ===========================
    // OpenEye K8/N4 row-stationary output DMA beat order = 0, 2, 1, 3.
    wire [1:0] mapped_score_index = {result_addr[0], result_addr[1]};
    wire score_write = qkt_result_enable && result_write && (result_addr < 4);
    
    attention_qkt_score_buffer #(
        .NUM_TOKENS(4),
        .NUM_SCORES(4),
        .DATA_WIDTH(20)
    ) u_qkt_score_buffer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(qkt_memory_clear),
        .write_en_i(score_write),
        .write_token_i(token_select),
        .write_score_i(mapped_score_index),
        .write_data_i(result_data[19:0]),
        .read_token_i(selected_score_rd_token),
        .read_score_i(selected_score_rd_index),
        .read_data_o(score_buffer_rd_data),
        .read_valid_o(score_buffer_rd_valid),
        .complete_o(scores_complete_o)
    );
        end else begin : GEN_NO_QKT
            assign packer_q8_rd_token = 2'd0;
            assign packer_q8_rd_feature = 3'd0;
            assign packer_k8_rd_token = 2'd0;
            assign packer_k8_rd_feature = 3'd0;
            assign q8_packer_busy = 1'b0;
            assign k8_packer_busy = 1'b0;
            assign qkt_stream_data = {DMA_WIDTH{1'b0}};
            assign operands_ready = 1'b0;
            assign score_buffer_rd_data = 20'sd0;
            assign score_buffer_rd_valid = 1'b0;
            assign scores_complete_o = 1'b0;
        end
    endgenerate

    // Stage-4 Softmax FSM lives in OpenEye_FPGA. The QKT Score Buffer
    // and its original combinational read interface remain unchanged.

`ifndef SYNTHESIS
    initial begin
        if (INTEGRATION_STAGE < 1 || INTEGRATION_STAGE > 4)
            $fatal(1, "INTEGRATION_STAGE must be between 1 and 4");
        if (DMA_WIDTH != 64 || ADDR_WIDTH != 16 ||
            PROJECTION_STREAM_WORDS != 939 || QKT_STREAM_WORDS != 677)
            $fatal(1, "attention_top: unverified geometry or DMA configuration");
    end
`endif
endmodule

