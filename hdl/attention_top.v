`timescale 1ns / 1ps
//=============================================================================
// OpenEye single-head Q/K/V projection -> requantization -> Q*K^T
// Integration draft: exactly ONE OpenEye_FPGA and ONE attention_dma_adapter.
// Existing standalone projection/QKT top modules are intentionally NOT used.
//
// Supported geometry (first integration): 4 tokens, 8 features,
// projection stream 939 words, QKT stream 677 words, 64-bit DMA.
// NOTE: This is an unverified integration draft. See companion notes.
//=============================================================================
module attention_top #(
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

    // Read the final 4x4 QK^T scores; used later by Softmax.
    input  wire [1:0] score_rd_token_i,
    input  wire [1:0] score_rd_index_i,
    output wire signed [19:0] score_rd_data_o,
    output wire score_rd_valid_o,

    // Debug outputs for integration tests.
    output wire [1:0] active_token_o,
    output wire [1:0] active_projection_o,
    output wire qkt_mode_o,
    output wire [ADDR_WIDTH-1:0] stream_addr_o,
    output wire dma_result_write_o
);
    localparam [ADDR_WIDTH-1:0] PROJECTION_WORD_COUNT = PROJECTION_STREAM_WORDS;
    localparam [ADDR_WIDTH-1:0] QKT_WORD_COUNT = QKT_STREAM_WORDS;

    // IMPORTANT: The legacy attention_qkt_ctrl.v is a single-GEMV controller
    // and is instantiated by legacy attention_gemv_control_top.v. To preserve
    // its passing tests, integration sequencing is isolated in the temporary
    // attention_top_scheduler module below, until a backwards-compatible
    // full-scope attention_qkt_ctrl interface has been settled.
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

    attention_top_scheduler u_scheduler (
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

    //===================== ONE SHARED OPENEYE ================================
    OpenEye_FPGA u_openeye (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .data_dma_i(dma_input_data),
        .enable_dma_i(dma_input_valid),
        .ready_dma_o(dma_input_ready),
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

    //===================== Q/K/V RAW RESULT STORAGE ==========================
    wire [1:0] quant_q_rd_token;
    wire [2:0] quant_q_rd_feature;
    wire [1:0] quant_k_rd_token;
    wire [2:0] quant_k_rd_feature;
    wire [19:0] q_raw_data;
    wire [19:0] k_raw_data;
    wire [19:0] v_raw_data;
    wire quant_busy;

    attention_projection_result_buffer u_projection_result_buffer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(projection_start),
        .token_select_i(token_select),
        .projection_select_i(projection_select),
        .result_write_i(result_write && !qkt_mode),
        .result_addr_i(result_addr),
        .result_data_i(result_data),
        .q_rd_token_i(quant_q_rd_token),
        .q_rd_feature_i(quant_q_rd_feature),
        .k_rd_token_i(quant_k_rd_token),
        .k_rd_feature_i(quant_k_rd_feature),
        .v_rd_token_i(2'd0),
        .v_rd_feature_i(3'd0),
        .q_data_o(q_raw_data),
        .k_data_o(k_raw_data),
        .v_data_o(v_raw_data)
    );

    //===================== Q/K REQUANTIZATION ================================
    wire [1:0] q8_rd_token;
    wire [2:0] q8_rd_feature;
    wire signed [7:0] q8_data;
    wire [1:0] k8_rd_token;
    wire [2:0] k8_rd_feature;
    wire signed [7:0] k8_data;
    wire [6:0] q_shift;
    wire [6:0] k_shift;

    attention_qk_quantization_controller #(
        .NUM_TOKENS(4),
        .NUM_FEATURES(8),
        .DATA_WIDTH(20),
        .SHIFT_WIDTH(7)
    ) u_quantization (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(quant_start),
        .busy_o(quant_busy),
        .done_o(quant_done),
        .q_raw_rd_token_o(quant_q_rd_token),
        .q_raw_rd_feature_o(quant_q_rd_feature),
        .q_raw_data_i(q_raw_data),
        .k_raw_rd_token_o(quant_k_rd_token),
        .k_raw_rd_feature_o(quant_k_rd_feature),
        .k_raw_data_i(k_raw_data),
        .q8_rd_token_i(q8_rd_token),
        .q8_rd_feature_i(q8_rd_feature),
        .q8_data_o(q8_data),
        .k8_rd_token_i(k8_rd_token),
        .k8_rd_feature_i(k8_rd_feature),
        .k8_data_o(k8_data),
        .q_shift_o(q_shift),
        .k_shift_o(k_shift)
    );

    //===================== QKT DMA OPERAND PACKERS ===========================
    wire [DMA_WIDTH-1:0] q8_word;
    wire q8_word_valid;
    wire q8_word_ready;
    wire q8_word_last;
    wire [3:0] q8_word_index;
    wire q8_packer_busy;
    wire q8_packer_done;
    wire [DMA_WIDTH-1:0] k8_word;
    wire k8_word_valid;
    wire k8_word_ready;
    wire k8_word_last;
    wire [3:0] k8_word_index;
    wire k8_packer_busy;
    wire k8_packer_done;

    // NOTE: Verify exact port names/widths against the local passing
    // attention_qkt_openeye_top.v before treating this draft as final.
    attention_qkt_q8_packer u_q8_packer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(q8_packer_start),
        .token_i(token_select),
        .q8_rd_token_o(q8_rd_token),
        .q8_rd_feature_o(q8_rd_feature),
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
        .k8_rd_token_o(k8_rd_token),
        .k8_rd_feature_o(k8_rd_feature),
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
        .read_token_i(score_rd_token_i),
        .read_score_i(score_rd_index_i),
        .read_data_o(score_rd_data_o),
        .read_valid_o(score_rd_valid_o),
        .complete_o(scores_complete_o)
    );

`ifndef SYNTHESIS
    initial begin
        if (DMA_WIDTH != 64 || ADDR_WIDTH != 16 ||
            PROJECTION_STREAM_WORDS != 939 || QKT_STREAM_WORDS != 677)
            $fatal(1, "attention_top: unverified geometry or DMA configuration");
    end
`endif
endmodule

//=============================================================================
// Integration-only orchestration FSM (preserves legacy attention_qkt_ctrl).
// 12 projection GEMVs -> quantization -> 4 QKT GEMVs -> global DONE.
// No arithmetic, Python orchestration, second DMA adapter or second OpenEye.
//=============================================================================
module attention_top_scheduler (
    input  wire clk_i,
    input  wire rst_ni,
    input  wire start_i,
    input  wire gemv_done_i,
    input  wire quant_done_i,
    input  wire operands_ready_i,
    output wire projection_start_o,
    output wire quant_start_o,
    output wire qkt_memory_clear_o,
    output wire qkt_q8_reload_o,
    output wire q8_packer_start_o,
    output wire k8_packer_start_o,
    output wire gemv_start_o,
    output wire qkt_mode_o,
    output wire qkt_result_enable_o,
    output wire [1:0] token_o,
    output wire [1:0] projection_o,
    output wire busy_o,
    output reg done_o
);
    localparam [3:0]
        IDLE              = 4'd0,
        PROJ_START        = 4'd1,
        PROJ_WAIT         = 4'd2,
        QUANT_START       = 4'd3,
        QUANT_WAIT        = 4'd4,
        QKT_CLEAR         = 4'd5,
        QKT_PACK_START    = 4'd6,
        QKT_PACK_WAIT     = 4'd7,
        QKT_GEMV_START    = 4'd8,
        QKT_GEMV_WAIT     = 4'd9,
        QKT_RELOAD        = 4'd10,
        FINISH            = 4'd11;

    reg [3:0] state;
    reg [1:0] token;
    reg [1:0] projection;
    reg qkt_mode;

    assign token_o = token;
    assign projection_o = projection;
    assign qkt_mode_o = qkt_mode;
    assign busy_o = (state != IDLE);
    assign projection_start_o = (state == PROJ_START);
    assign quant_start_o = (state == QUANT_START);
    assign qkt_memory_clear_o = (state == QKT_CLEAR);
    assign qkt_q8_reload_o = (state == QKT_RELOAD);
    assign q8_packer_start_o = (state == QKT_PACK_START);
    assign k8_packer_start_o = (state == QKT_PACK_START) && (token == 2'd0);
    assign gemv_start_o = (state == PROJ_START) || (state == QKT_GEMV_START);
    assign qkt_result_enable_o = (state == QKT_GEMV_WAIT);

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= IDLE;
            token <= 2'd0;
            projection <= 2'd0;
            qkt_mode <= 1'b0;
            done_o <= 1'b0;
        end else begin
            done_o <= 1'b0;
            case (state)
                IDLE: begin
                    if (start_i) begin
                        token <= 2'd0;
                        projection <= 2'd0;
                        qkt_mode <= 1'b0;
                        state <= PROJ_START;
                    end
                end
                PROJ_START: state <= PROJ_WAIT;
                PROJ_WAIT: begin
                    if (gemv_done_i) begin
                        if (token != 2'd3) begin
                            token <= token + 2'd1;
                            state <= PROJ_START;
                        end else if (projection != 2'd2) begin
                            token <= 2'd0;
                            projection <= projection + 2'd1;
                            state <= PROJ_START;
                        end else begin
                            state <= QUANT_START;
                        end
                    end
                end
                QUANT_START: state <= QUANT_WAIT;
                QUANT_WAIT: if (quant_done_i) begin
                    qkt_mode <= 1'b1;
                    token <= 2'd0;
                    state <= QKT_CLEAR;
                end
                QKT_CLEAR: state <= QKT_PACK_START;
                QKT_PACK_START: state <= QKT_PACK_WAIT;
                QKT_PACK_WAIT: if (operands_ready_i) state <= QKT_GEMV_START;
                QKT_GEMV_START: state <= QKT_GEMV_WAIT;
                QKT_GEMV_WAIT: if (gemv_done_i) begin
                    if (token == 2'd3)
                        state <= FINISH;
                    else begin
                        token <= token + 2'd1;
                        state <= QKT_RELOAD;
                    end
                end
                QKT_RELOAD: state <= QKT_PACK_START;
                FINISH: begin
                    done_o <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
