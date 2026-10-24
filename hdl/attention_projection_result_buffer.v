`timescale 1ns / 1ps

module attention_projection_result_buffer (
    input wire clk_i,
    input wire rst_ni,

    // Capture operation metadata at start
    input wire start_i,
    input wire [1:0] token_select_i,
    input wire [1:0] projection_select_i,

    // DMA result interface
    input wire result_write_i,
    input wire [15:0] result_addr_i,
    input wire [63:0] result_data_i,

    // Independent Q read address
    input wire [1:0] q_rd_token_i,
    input wire [2:0] q_rd_feature_i,

    // Independent K read address
    input wire [1:0] k_rd_token_i,
    input wire [2:0] k_rd_feature_i,

    // Independent V read address
    input wire [1:0] v_rd_token_i,
    input wire [2:0] v_rd_feature_i,

    output wire [19:0] q_data_o,
    output wire [19:0] k_data_o,
    output wire [19:0] v_data_o
);

    reg [1:0] active_token;
    reg [1:0] active_projection;

    reg [19:0] q_mem [0:31];
    reg [19:0] k_mem [0:31];
    reg [19:0] v_mem [0:31];

    wire [4:0] write_addr;
    wire [4:0] read_addr;

    // Only even beats 0,2,...,14 contain
    // the eight logical projection outputs.
    wire valid_feature =
        result_write_i &&
        (result_addr_i < 16) &&
        (result_addr_i[0] == 1'b0);

    assign write_addr = {
        active_token,
        result_addr_i[3:1]
    };

    wire [4:0] q_read_addr;
    wire [4:0] k_read_addr;
    wire [4:0] v_read_addr;

    assign q_read_addr = {q_rd_token_i, q_rd_feature_i};
    assign k_read_addr = {k_rd_token_i, k_rd_feature_i};
    assign v_read_addr = {v_rd_token_i, v_rd_feature_i};

    assign q_data_o = q_mem[q_read_addr];
    assign k_data_o = k_mem[k_read_addr];
    assign v_data_o = v_mem[v_read_addr];

    integer i;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_token <= 0;
            active_projection <= 0;

            for (i = 0; i < 32; i = i + 1) begin
                q_mem[i] <= 0;
                k_mem[i] <= 0;
                v_mem[i] <= 0;
            end
        end else begin

            // Keep metadata stable throughout GEMV
            if (start_i) begin
                active_token <= token_select_i;
                active_projection <= projection_select_i;
            end

            if (valid_feature) begin
                case (active_projection)
                    2'd0:
                        q_mem[write_addr] <= result_data_i[19:0];

                    2'd1:
                        k_mem[write_addr] <= result_data_i[19:0];

                    2'd2:
                        v_mem[write_addr] <= result_data_i[19:0];

                    default: ;
                endcase
            end
        end
    end

endmodule