`timescale 1ns/1ps

module attention_qkt_q8_packer #(
    parameter integer NUM_TOKENS = 4,
    parameter integer NUM_FEATURES = 8,

    parameter integer TOKEN_BITS =
        (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS),

    parameter integer FEATURE_BITS =
        (NUM_FEATURES <= 1) ? 1 : $clog2(NUM_FEATURES),

    parameter integer WORDS_PER_TOKEN =
        (NUM_FEATURES + 7) / 8,

    parameter integer WORD_INDEX_BITS =
        (WORDS_PER_TOKEN <= 1) ? 1 : $clog2(WORDS_PER_TOKEN)
)(
    input wire clk_i,
    input wire rst_ni,
    input wire start_i,

    input wire [TOKEN_BITS-1:0] token_i,

    // Existing Q8 memory interface
    output wire [TOKEN_BITS-1:0] q8_rd_token_o,
    output wire [FEATURE_BITS-1:0] q8_rd_feature_o,
    input wire signed [7:0] q8_data_i,

    // 64-bit streaming interface
    output reg [63:0] q8_word_o,
    output reg q8_word_valid_o,
    input wire q8_word_ready_i,
    output reg q8_word_last_o,

    output reg [WORD_INDEX_BITS-1:0] word_index_o,

    output reg busy_o,
    output reg done_o
);

    localparam [1:0]
        IDLE = 2'd0,
        READ = 2'd1,
        SEND = 2'd2;

    reg [1:0] state;

    reg [TOKEN_BITS-1:0] selected_token;
    reg [FEATURE_BITS-1:0] feature_index;
    reg [2:0] byte_index;

    assign q8_rd_token_o = selected_token;
    assign q8_rd_feature_o = feature_index;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= IDLE;

            selected_token <= 0;
            feature_index <= 0;
            byte_index <= 0;

            q8_word_o <= 0;
            q8_word_valid_o <= 0;
            q8_word_last_o <= 0;

            word_index_o <= 0;

            busy_o <= 0;
            done_o <= 0;

        end else begin
            done_o <= 0;

            case (state)

                IDLE: begin
                    if (start_i) begin
                        selected_token <= token_i;

                        feature_index <= 0;
                        byte_index <= 0;
                        word_index_o <= 0;

                        q8_word_o <= 0;
                        q8_word_valid_o <= 0;
                        q8_word_last_o <= 0;

                        busy_o <= 1;
                        state <= READ;
                    end
                end

                READ: begin

                    // Pack one INT8 feature per clock
                    q8_word_o[byte_index*8 +: 8]
                        <= q8_data_i;

                    // Advance memory read address
                    if (feature_index != NUM_FEATURES-1)
                        feature_index <= feature_index + 1'b1;

                    // Word complete or final feature reached
                    if ((byte_index == 3'd7) ||
                        (feature_index == NUM_FEATURES-1)) begin

                        q8_word_valid_o <= 1;

                        q8_word_last_o <=
                            (feature_index == NUM_FEATURES-1);

                        state <= SEND;

                    end else begin
                        byte_index <= byte_index + 1'b1;
                    end
                end

                SEND: begin

                    // Hold the word until receiver accepts it
                    if (q8_word_valid_o && q8_word_ready_i) begin

                        q8_word_valid_o <= 0;
                        q8_word_last_o <= 0;

                        if (q8_word_last_o) begin
                            busy_o <= 0;
                            done_o <= 1;
                            state <= IDLE;

                        end else begin

                            // Prepare next 64-bit word
                            q8_word_o <= 0;
                            byte_index <= 0;

                            word_index_o <= word_index_o + 1'b1;

                            state <= READ;
                        end
                    end
                end

                default: state <= IDLE;

            endcase
        end
    end

endmodule