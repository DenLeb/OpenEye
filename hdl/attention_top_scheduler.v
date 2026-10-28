`timescale 1ns / 1ps

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
