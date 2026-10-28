`timescale 1ns/1ps
// Isolated test wrapper. Models the EXISTING result buffer's combinational
// read ports, and allows Cocotb to preload raw matrices without OpenEye GEMVs.
module attention_qk_quantization_controller_test_top (
    input wire clk_i,
    input wire rst_ni,
    input wire start_i,
    input wire raw_wr_en_i,
    input wire [4:0] raw_wr_index_i,
    input wire signed [19:0] raw_wr_q_data_i,
    input wire signed [19:0] raw_wr_k_data_i,
    input wire [1:0] q8_rd_token_i,
    input wire [2:0] q8_rd_feature_i,
    input wire [1:0] k8_rd_token_i,
    input wire [2:0] k8_rd_feature_i,
    output wire signed [7:0] q8_data_o,
    output wire signed [7:0] k8_data_o,
    output wire [6:0] q_shift_o,
    output wire [6:0] k_shift_o,
    output wire busy_o,
    output wire done_o
);
    reg signed [19:0] raw_q [0:31];
    reg signed [19:0] raw_k [0:31];
    wire [1:0] q_tok, k_tok;
    wire [2:0] q_feat, k_feat;
    wire signed [19:0] q_data = raw_q[{q_tok,q_feat}];
    wire signed [19:0] k_data = raw_k[{k_tok,k_feat}];

    always @(posedge clk_i) begin
        if (raw_wr_en_i) begin
            raw_q[raw_wr_index_i] <= raw_wr_q_data_i;
            raw_k[raw_wr_index_i] <= raw_wr_k_data_i;
        end
    end

    attention_qk_quantization_controller u_ctrl (
        .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start_i),
        .busy_o(busy_o), .done_o(done_o),
        .q_raw_rd_token_o(q_tok), .q_raw_rd_feature_o(q_feat),
        .k_raw_rd_token_o(k_tok), .k_raw_rd_feature_o(k_feat),
        .q_raw_data_i(q_data), .k_raw_data_i(k_data),
        .q8_rd_token_i(q8_rd_token_i), .q8_rd_feature_i(q8_rd_feature_i),
        .k8_rd_token_i(k8_rd_token_i), .k8_rd_feature_i(k8_rd_feature_i),
        .q8_data_o(q8_data_o), .k8_data_o(k8_data_o),
        .q_shift_o(q_shift_o), .k_shift_o(k_shift_o)
    );
endmodule
