
`timescale 1ns/1ps

module attention_qkt_openeye_top #(
    parameter integer NUM_TOKENS   = 4,
    parameter integer NUM_FEATURES = 8,
    parameter integer DMA_WIDTH    = 64,
    parameter integer SCORE_WIDTH  = 20,
    parameter integer ADDR_WIDTH   = 16,
    parameter integer STREAM_WORDS = 677,

    parameter TEMPLATE_FILE = "qkt_t0_dma.mem",

    parameter integer TOKEN_BITS =
        (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS),

    parameter integer FEATURE_BITS =
        (NUM_FEATURES <= 1) ? 1 : $clog2(NUM_FEATURES),

    parameter integer SCORE_BITS = TOKEN_BITS
)(
    input wire clk_i,
    input wire rst_ni,

    // Start after Q/K quantization has finished.
    input wire start_i,

    output wire busy_o,
    output reg done_o,
    output wire scores_complete_o,

    // Read interfaces to existing Q8/K8 memories.
    output wire [TOKEN_BITS-1:0] q8_rd_token_o,
    output wire [FEATURE_BITS-1:0] q8_rd_feature_o,
    input wire signed [7:0] q8_data_i,

    output wire [TOKEN_BITS-1:0] k8_rd_token_o,
    output wire [FEATURE_BITS-1:0] k8_rd_feature_o,
    input wire signed [7:0] k8_data_i,

    // Read interface for future Softmax.
    input wire [TOKEN_BITS-1:0] score_rd_token_i,
    input wire [SCORE_BITS-1:0] score_rd_index_i,
    output wire signed [SCORE_WIDTH-1:0] score_rd_data_o,
    output wire score_rd_valid_o,

    // Debug
    output wire [ADDR_WIDTH-1:0] stream_addr_o,
    output wire [TOKEN_BITS-1:0] active_token_o
);

    // ---------------------------------------------------------
    // Four consecutive GEMV executions
    // ---------------------------------------------------------

    localparam [2:0]
        IDLE       = 3'd0,
        CLEAR      = 3'd1,
        LOAD       = 3'd2,
        WAIT_PACK  = 3'd3,
        RUN_GEMV   = 3'd4,
        WAIT_GEMV  = 3'd5,
        RELOAD_Q8  = 3'd6;

    reg [2:0] state;
    reg [TOKEN_BITS-1:0] token_index;

    assign active_token_o = token_index;
    assign busy_o = (state != IDLE);

    wire clear_memory = (state == CLEAR);
    wire reload_q8 = (state == RELOAD_Q8);

    wire q8_start = (state == LOAD);
    wire k8_start = (state == LOAD) &&
                (token_index == {TOKEN_BITS{1'b0}});

    wire gemv_start = (state == RUN_GEMV);

    // ---------------------------------------------------------
    // Q8 Packer
    // ---------------------------------------------------------

    wire [63:0] q8_word;
    wire q8_valid;
    wire q8_ready;

    attention_qkt_q8_packer #(
        .NUM_TOKENS(NUM_TOKENS),
        .NUM_FEATURES(NUM_FEATURES)
    ) u_q8_packer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(q8_start),
        .token_i(token_index),

        .q8_rd_token_o(q8_rd_token_o),
        .q8_rd_feature_o(q8_rd_feature_o),
        .q8_data_i(q8_data_i),

        .q8_word_o(q8_word),
        .q8_word_valid_o(q8_valid),
        .q8_word_ready_i(q8_ready),
        .q8_word_last_o(),
        .word_index_o(),
        .busy_o(),
        .done_o()
    );

    // ---------------------------------------------------------
    // K8 Packer: Load weights only once
    // ---------------------------------------------------------

    wire [63:0] k8_word;
    wire k8_valid;
    wire k8_ready;
    wire [3:0] k8_word_index;

    attention_qkt_k8_packer u_k8_packer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .start_i(k8_start),

        .k8_rd_token_o(k8_rd_token_o),
        .k8_rd_feature_o(k8_rd_feature_o),
        .k8_data_i(k8_data_i),

        .wght_word_o(k8_word),
        .wght_word_valid_o(k8_valid),
        .wght_word_ready_i(k8_ready),
        .wght_word_last_o(),
        .word_index_o(k8_word_index),
        .busy_o(),
        .done_o()
    );

    // ---------------------------------------------------------
    // QKT DMA Memory
    // ---------------------------------------------------------

    wire operands_ready;
    wire [DMA_WIDTH-1:0] stream_data;

    attention_qkt_dma_memory #(
        .DMA_WIDTH(DMA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .TEMPLATE_FILE(TEMPLATE_FILE)
    ) u_qkt_memory (
        .clk_i(clk_i),
        .rst_ni(rst_ni),

        .clear_i(clear_memory),
        .q8_reload_i(reload_q8),

        .q8_word_i(q8_word),
        .q8_word_valid_i(q8_valid),
        .q8_word_ready_o(q8_ready),

        .k8_word_i(k8_word),
        .k8_word_valid_i(k8_valid),
        .k8_word_ready_o(k8_ready),
        .k8_word_index_i(k8_word_index),

        .operands_ready_o(operands_ready),

        .stream_addr_i(stream_addr_o),
        .stream_data_o(stream_data)
    );

    // ---------------------------------------------------------
    // Existing GEMV Controller
    // ---------------------------------------------------------

    wire gemv_done;
    wire gemv_busy;
    wire adapter_busy;

    wire [63:0] dma_input_data;
    wire dma_input_valid;
    wire dma_input_ready;

    wire [63:0] dma_output_data;
    wire dma_output_valid;
    wire dma_output_last;
    wire dma_output_ready;

    wire result_write;
    wire [63:0] result_data;
    wire [ADDR_WIDTH-1:0] result_addr;

   attention_dma_adapter #(
    .DMA_WIDTH(DMA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH)
    ) u_qkt_dma_adapter (
        .clk_i(clk_i),
        .rst_ni(rst_ni),

        .gemv_start_i(gemv_start),
        .gemv_done_o(gemv_done),
        .busy_o(adapter_busy),

        .stream_word_count_i(STREAM_WORDS),
        .stream_addr_o(stream_addr_o),
        .stream_data_i(stream_data),

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

    // ---------------------------------------------------------
    // Real OpenEye FPGA
    // ---------------------------------------------------------

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

    // ---------------------------------------------------------
    // Score Buffer
    // ---------------------------------------------------------

    // Dense output order for current K=8, N=4 mapping:
    // DMA beats: 0, 1, 2, 3
    // Scores:    0, 2, 1, 3

    localparam integer OUTPUT_COLUMNS = 2;

    localparam integer SCORES_PER_COLUMN =
        (NUM_TOKENS + OUTPUT_COLUMNS - 1) / OUTPUT_COLUMNS;

    localparam integer OUTPUT_BEATS =
        OUTPUT_COLUMNS * SCORES_PER_COLUMN;

    wire [ADDR_WIDTH-1:0] mapped_score_linear =
        (result_addr % OUTPUT_COLUMNS) * SCORES_PER_COLUMN +
        (result_addr / OUTPUT_COLUMNS);

    wire [SCORE_BITS-1:0] mapped_score_index =
        mapped_score_linear[SCORE_BITS-1:0];

    wire score_write =
        result_write &&
        (result_addr < OUTPUT_BEATS) &&
        (mapped_score_linear < NUM_TOKENS);

    attention_qkt_score_buffer #(
        .NUM_TOKENS(NUM_TOKENS),
        .NUM_SCORES(NUM_TOKENS),
        .DATA_WIDTH(SCORE_WIDTH)
    ) u_score_buffer (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .clear_i(clear_memory),

        .write_en_i(score_write),
        .write_token_i(token_index),
        .write_score_i(mapped_score_index),
        .write_data_i(result_data[SCORE_WIDTH-1:0]),

        .read_token_i(score_rd_token_i),
        .read_score_i(score_rd_index_i),
        .read_data_o(score_rd_data_o),
        .read_valid_o(score_rd_valid_o),
        .complete_o(scores_complete_o)
    );

    // ---------------------------------------------------------
    // Main sequencing FSM
    // ---------------------------------------------------------

    `ifndef SYNTHESIS
    initial begin
        if (NUM_TOKENS != 4 ||
            NUM_FEATURES != 8 ||
            DMA_WIDTH != 64 ||
            SCORE_WIDTH != 20 ||
            STREAM_WORDS != 677) begin

            $fatal(1,
                "QKT configuration not yet supported by all submodules");
        end
    end
    `endif

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= IDLE;
            token_index <= 0;
            done_o <= 0;
        end else begin
            done_o <= 0;

            case (state)

                IDLE: begin
                    if (start_i) begin
                        token_index <= 0;
                        state <= CLEAR;
                    end
                end

                CLEAR: begin
                    state <= LOAD;
                end

                LOAD: begin
                    // Start Q8; start K8 only for token 0.
                    state <= WAIT_PACK;
                end

                WAIT_PACK: begin
                    if (operands_ready)
                        state <= RUN_GEMV;
                end

                RUN_GEMV: begin
                    state <= WAIT_GEMV;
                end

                WAIT_GEMV: begin
                    if (gemv_done) begin
                        if (token_index == NUM_TOKENS - 1) begin
                            done_o <= 1'b1;
                            state <= IDLE;
                        end else begin
                            token_index <= token_index + 1'b1;
                            state <= RELOAD_Q8;
                        end
                    end
                end

                RELOAD_Q8: begin
                    // Preserve K8, replace Q8.
                    state <= LOAD;
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
