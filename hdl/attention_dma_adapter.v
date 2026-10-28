`timescale 1ns / 1ps

//==============================================================================
// Module: attention_dma_adapter
// Project: OpenEye - Single-Head Attention Accelerator
//==============================================================================
// Description:
//   Handles one GEMV transaction through the OpenEye_FPGA DMA interface.
//
//   This initial version:
//     - Reads preformatted 64-bit words from an external word source.
//     - Streams the words to OpenEye.
//     - Captures result words and their addresses.
//     - Reports completion after the final output word.
//
//   DMA packet generation and tensor storage are implemented separately.
//   Input streaming initially follows OpenEye's existing continuous-stream
//   testbench protocol. Per-word backpressure requires integration testing.
//==============================================================================

// DMA Adapter <--> OpenEye_FPGA Connections:
//
// Adapter Output                    OpenEye Input
// ---------------------------------------------------------------
// dma_data_o           -----------> data_dma_i
// dma_valid_o          -----------> enable_dma_i
// dma_result_ready_o   -----------> ready_dma_i
//
// Adapter Input                     OpenEye Output
// ---------------------------------------------------------------
// dma_ready_i          <----------- ready_dma_o
// dma_result_data_i    <----------- data_dma_o
// dma_result_valid_i   <----------- enable_dma_o
// dma_result_last_i    <----------- last_data_o
//
// Data Flow:
//   1. Adapter -> OpenEye:
//      Sends configuration, input activations, weights, and other
//      required data through the DMA input interface.
//
//   2. OpenEye -> Adapter:
//      Returns computed partial sums (PSUMs) through the DMA
//      output interface.

module attention_dma_adapter #(
    parameter integer DMA_WIDTH  = 64,
    parameter integer ADDR_WIDTH = 16
)(
    input  wire clk_i,
    input  wire rst_ni,

    // Main Attention controller
    input  wire gemv_start_i,
    output reg  gemv_done_o,
    output wire busy_o,

    // Input word source (future packet generator / buffer)
    input  wire [ADDR_WIDTH-1:0] stream_word_count_i,
    output wire [ADDR_WIDTH-1:0] stream_addr_o,
    input  wire [DMA_WIDTH-1:0]  stream_data_i,

    // DMA input interface of OpenEye_FPGA
    input  wire                 dma_ready_i,
    output wire [DMA_WIDTH-1:0] dma_data_o,
    output wire                 dma_valid_o,

    // DMA output interface of OpenEye_FPGA
    input  wire [DMA_WIDTH-1:0] dma_result_data_i,
    input  wire                 dma_result_valid_i,
    input  wire                 dma_result_last_i,
    output wire                 dma_result_ready_o,

    // Interface to future result buffer
    output wire                 result_write_o,
    output wire [DMA_WIDTH-1:0] result_data_o,
    output wire [ADDR_WIDTH-1:0] result_addr_o
);

    //==========================================================================
    // FSM States
    //==========================================================================

    localparam [1:0] IDLE        = 2'd0;
    localparam [1:0] WAIT_READY  = 2'd1;
    localparam [1:0] SEND_STREAM = 2'd2;
    localparam [1:0] WAIT_RESULT = 2'd3;

    reg [1:0] state_reg;

    reg [ADDR_WIDTH-1:0] stream_addr_reg;
    reg [ADDR_WIDTH-1:0] stream_count_reg;
    reg [ADDR_WIDTH-1:0] result_addr_reg;

    //==========================================================================
    // Combinational Output Logic
    //==========================================================================

    assign busy_o = (state_reg != IDLE);

    assign stream_addr_o = stream_addr_reg;

    assign dma_data_o  = stream_data_i;
    assign dma_valid_o = (state_reg == SEND_STREAM);

    assign dma_result_ready_o = (state_reg == WAIT_RESULT);

    assign result_write_o = dma_result_valid_i && dma_result_ready_o;
    assign result_data_o  = dma_result_data_i;
    assign result_addr_o  = result_addr_reg;

    //==========================================================================
    // Main FSM
    //==========================================================================

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_reg       <= IDLE;
            stream_addr_reg <= 0;
            stream_count_reg <= 0;
            result_addr_reg <= 0;
            gemv_done_o     <= 1'b0;
        end
        else begin
            gemv_done_o <= 1'b0;

            case (state_reg)

                IDLE: begin
                    if (gemv_start_i && stream_word_count_i != 0) begin
                        stream_count_reg <= stream_word_count_i;
                        stream_addr_reg  <= 0;
                        result_addr_reg  <= 0;
                        state_reg        <= WAIT_READY;
                    end
                end

                WAIT_READY: begin
                    if (dma_ready_i) begin
                        state_reg <= SEND_STREAM;
                    end
                end

                SEND_STREAM: begin
                    if (stream_addr_reg == stream_count_reg - 1'b1) begin
                        state_reg <= WAIT_RESULT;
                    end
                    else begin
                        stream_addr_reg <= stream_addr_reg + 1'b1;
                    end
                end

                WAIT_RESULT: begin
                    if (result_write_o) begin
                        result_addr_reg <= result_addr_reg + 1'b1;

                        if (dma_result_last_i) begin
                            gemv_done_o <= 1'b1;
                            state_reg   <= IDLE;
                        end
                    end
                end

                default: begin
                    state_reg <= IDLE;
                end

            endcase
        end
    end

endmodule