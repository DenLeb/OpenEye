
`timescale 1ns/1ps

module attention_qkt_k8_packer #(
    parameter integer NUM_TOKENS = 4,
    parameter integer NUM_FEATURES = 8,
    parameter integer DMA_BITWIDTH = 64,
    parameter integer TOKEN_BITS =
        (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS),
    parameter integer FEATURE_BITS =
        (NUM_FEATURES <= 1) ? 1 : $clog2(NUM_FEATURES)
)(
    input wire clk_i,
    input wire rst_ni,
    input wire start_i,

    // Existing K8 memory read interface
    output wire [TOKEN_BITS-1:0] k8_rd_token_o,
    output wire [FEATURE_BITS-1:0] k8_rd_feature_o,
    input wire signed [7:0] k8_data_i,

    // Packed DMA weight words
    output wire [DMA_BITWIDTH-1:0] wght_word_o,
    output wire wght_word_valid_o,
    input wire wght_word_ready_i,
    output wire wght_word_last_o,
    output reg [3:0] word_index_o,

    output reg busy_o,
    output reg done_o
);

    // Current verified DenseMapper configuration:
    // K8 shape: 4 x 8
    // 10 DMA words, organized in two groups of 5.
    localparam integer GROUP_WORDS = 5;
    localparam integer TOTAL_WORDS = 10;
    localparam integer GROUP_BITS =
        GROUP_WORDS * DMA_BITWIDTH;

    localparam [1:0]
        IDLE = 2'd0,
        READ = 2'd1,
        SEND = 2'd2;

    reg [1:0] state;

    reg [TOKEN_BITS-1:0] token_index;
    reg [FEATURE_BITS-1:0] feature_index;

    reg [GROUP_BITS-1:0] even_payload;
    reg [GROUP_BITS-1:0] odd_payload;

    assign k8_rd_token_o = token_index;
    assign k8_rd_feature_o = feature_index;

    // Bit position inside a 5-word group.
    // Mapping matches Dense Row-Stationary K8 [4 x 8],
    // SPARSITY_EN=1 and PARALLEL_MACS=2.
    function integer weight_bit_position;
        input integer token;
        input integer feature;
        integer pair;
        begin
            pair = feature / 2;

            if (token < 2)
                weight_bit_position =
                    pair * 24 + token * 12;

            else if (pair < 2)
                weight_bit_position =
                    144 + pair * 24 + (token - 2) * 12;

            else
                weight_bit_position =
                    192 + (pair - 2) * 24
                        + (token - 2) * 12;
        end
    endfunction

    wire [31:0] field_position =
        weight_bit_position(token_index, feature_index);

    // Five 64-bit words are stored as one continuous
    // bitfield, so an INT8 value can cross word boundaries.
    wire [GROUP_BITS-1:0] selected_payload =
        (word_index_o < GROUP_WORDS)
            ? even_payload : odd_payload;

    wire [3:0] group_word_index =
        (word_index_o < GROUP_WORDS)
            ? word_index_o
            : (word_index_o - GROUP_WORDS);

    // DenseMapper reverses the five words per group.
    assign wght_word_o =
        selected_payload[
            (GROUP_WORDS - 1 - group_word_index)
                * DMA_BITWIDTH +: DMA_BITWIDTH
        ];

    assign wght_word_valid_o = (state == SEND);

    assign wght_word_last_o =
        (state == SEND) &&
        (word_index_o == TOTAL_WORDS - 1);

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= IDLE;
            token_index <= 0;
            feature_index <= 0;
            word_index_o <= 0;
            even_payload <= 0;
            odd_payload <= 0;
            busy_o <= 0;
            done_o <= 0;

        end else begin
            done_o <= 0;

            case (state)
                IDLE: begin
                    if (start_i) begin
                        token_index <= 0;
                        feature_index <= 0;
                        word_index_o <= 0;
                        even_payload <= 0;
                        odd_payload <= 0;
                        busy_o <= 1;
                        state <= READ;
                    end
                end

                READ: begin
                    // Capture one K8 value per clock.
                    // Even and odd features are packed
                    // into separate DMA word groups.
                    if (feature_index[0])
                        odd_payload[field_position +: 8]
                            <= k8_data_i;
                    else
                        even_payload[field_position +: 8]
                            <= k8_data_i;

                    if (feature_index == NUM_FEATURES - 1) begin
                        feature_index <= 0;

                        if (token_index == NUM_TOKENS - 1)
                            state <= SEND;
                        else
                            token_index <= token_index + 1'b1;

                    end else begin
                        feature_index <= feature_index + 1'b1;
                    end
                end

                SEND: begin
                    // Hold output stable during backpressure.
                    if (wght_word_ready_i) begin
                        if (word_index_o == TOTAL_WORDS - 1) begin
                            busy_o <= 0;
                            done_o <= 1;
                            state <= IDLE;
                        end else begin
                            word_index_o <= word_index_o + 1'b1;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
