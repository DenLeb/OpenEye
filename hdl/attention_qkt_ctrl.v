//==============================================================================
// Module: attention_qkt_ctrl
// Project: OpenEye - Single-Head Attention Accelerator
//==============================================================================
// Description:
//   Main finite state machine (FSM) for controlling the execution of
//   single-head self-attention operations using the existing OpenEye
//   hardware accelerator.
//
//   The controller is responsible for sequencing the following operations:
//     1. Query projection:  Q = X * W_Q
//     2. Key projection:    K = X * W_K
//     3. Value projection:  V = X * W_V
//     4. Attention scores:  S = Q * K^T
//
//   Matrix operations are executed by the existing OpenEye processing
//   elements (PEs). This module does not implement arithmetic units.
//
//   A separate DMA adapter handles the configuration and data transfer
//   between this controller and the OpenEye_FPGA module.
//
//   Intermediate tensor storage and requantization are coordinated by
//   this controller through dedicated hardware modules.
//
// Scope:
//   - Single-head attention only.
//   - Fully synthesizable RTL implementation.
//   - No runtime Python or processor-based orchestration.
//   - Softmax and subsequent attention operations are excluded.
//   - Architecture intended to allow future multi-head extensions.
//
// Development Status:
//   Initial version: Single GEMV execution controller.
//   Future versions: Complete Q, K, V and QK^T operation sequencing.
//
//==============================================================================

`timescale 1ns / 1ps

module attention_qkt_ctrl (
    input  wire clk_i,
    input  wire rst_ni,

    input  wire start_i,
    input  wire gemv_done_i,

    output reg  gemv_start_o,
    output reg  busy_o,
    output reg  done_o
);



//==========================================================================
// FSM State Definitions
//==========================================================================

    localparam [1:0] IDLE       = 2'd0;
    localparam [1:0] START_GEMV = 2'd1;
    localparam [1:0] WAIT_GEMV  = 2'd2;
    localparam [1:0] DONE       = 2'd3;

    reg [1:0] state_reg;

//==========================================================================
// Main FSM 
//==========================================================================    

    always @(posedge clk_i or negedge rst_ni) begin
            if (!rst_ni) begin
                state_reg <= IDLE;
                gemv_start_o <= 1'b0;
                busy_o <= 1'b0;
                done_o <= 1'b0;
            end 
            else begin

                // Default pulse outputs
                gemv_start_o <= 1'b0;
                done_o <= 1'b0;

                case (state_reg)

                    IDLE: begin
                        busy_o <= 1'b0;
        
                        if (start_i) begin
                            busy_o <= 1'b1;
                            state_reg <= START_GEMV;
                        end
                    end

                    START_GEMV: begin
                        gemv_start_o <= 1'b1; // Trigger GEMV operation
                        busy_o <= 1'b1;
                        state_reg <= WAIT_GEMV;
                    end

                    WAIT_GEMV: begin
                        busy_o <= 1'b1;
                       
                        if (gemv_done_i) begin
                            state_reg <= DONE;
                        end
                    end

                    DONE: begin

                        done_o <= 1'b1; 
                        busy_o <= 1'b0;
                        state_reg <= IDLE; 

                    end

                    default: begin
                        state_reg <= IDLE; // Default case to handle unexpected states
                        gemv_start_o <= 1'b0;
                        busy_o <= 1'b0;
                        done_o <= 1'b0;
                    end

                    
                endcase
            end
    end

endmodule