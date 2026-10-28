
`timescale 1ns/1ps

module attention_qkt_score_buffer #(
    parameter integer NUM_TOKENS = 4,
    parameter integer NUM_SCORES = 4,
    parameter integer DATA_WIDTH = 20,

    parameter integer TOKEN_BITS =
        (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS),

    parameter integer SCORE_BITS =
        (NUM_SCORES <= 1) ? 1 : $clog2(NUM_SCORES),

    parameter integer TOTAL = NUM_TOKENS * NUM_SCORES,

    parameter integer ADDR_BITS =
        (TOTAL <= 1) ? 1 : $clog2(TOTAL)
)(
    input wire clk_i,
    input wire rst_ni,
    input wire clear_i,

    // Write interface from OpenEye output controller
    input wire write_en_i,
    input wire [TOKEN_BITS-1:0] write_token_i,
    input wire [SCORE_BITS-1:0] write_score_i,
    input wire signed [DATA_WIDTH-1:0] write_data_i,

    // Read interface for future Softmax
    input wire [TOKEN_BITS-1:0] read_token_i,
    input wire [SCORE_BITS-1:0] read_score_i,

    output wire signed [DATA_WIDTH-1:0] read_data_o,
    output wire read_valid_o,

    // All score locations have been written
    output wire complete_o
);

    reg signed [DATA_WIDTH-1:0] score_mem [0:TOTAL-1];
    reg [TOTAL-1:0] valid_bits;

    wire [ADDR_BITS-1:0] write_addr =
        write_token_i * NUM_SCORES + write_score_i;

    wire [ADDR_BITS-1:0] read_addr =
        read_token_i * NUM_SCORES + read_score_i;

    wire write_in_range =
        (write_token_i < NUM_TOKENS) &&
        (write_score_i < NUM_SCORES);

    wire read_in_range =
        (read_token_i < NUM_TOKENS) &&
        (read_score_i < NUM_SCORES);

    assign read_valid_o =
        read_in_range && valid_bits[read_addr];

    assign read_data_o =
        read_valid_o ? score_mem[read_addr] : {DATA_WIDTH{1'b0}};

    assign complete_o = &valid_bits;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_bits <= {TOTAL{1'b0}};
        end else if (clear_i) begin
            valid_bits <= {TOTAL{1'b0}};
        end else if (write_en_i && write_in_range) begin
            score_mem[write_addr] <= write_data_i;
            valid_bits[write_addr] <= 1'b1;
        end
    end

endmodule
