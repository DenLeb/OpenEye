`timescale 1ns / 1ps

//==============================================================================
// Module: attention_gemv_control_top
// Project: OpenEye - Single-Head Attention Accelerator
//==============================================================================
// Description:
//   Integrates the Attention GEMV controller and DMA adapter.
//
//   The controller manages execution requests, while the DMA adapter
//   handles the transfer of input and output data.
//
//   This module is intended for integration testing before connecting
//   the actual OpenEye_FPGA accelerator.
//==============================================================================

module attention_gemv_control_top #(
    parameter integer DMA_WIDTH  = 64,
    parameter integer ADDR_WIDTH = 16
)(
    input  wire clk_i,
    input  wire rst_ni,

    // External operation control
    input  wire start_i,
    output wire busy_o,
    output wire done_o,
    output wire adapter_busy_o,

    // Input stream source
    input  wire [ADDR_WIDTH-1:0] stream_word_count_i,
    output wire [ADDR_WIDTH-1:0] stream_addr_o,
    input  wire [DMA_WIDTH-1:0]  stream_data_i,

    // DMA input connection
    input  wire                 dma_ready_i,
    output wire [DMA_WIDTH-1:0] dma_data_o,
    output wire                 dma_valid_o,

    // DMA result connection
    input  wire [DMA_WIDTH-1:0] dma_result_data_i,
    input  wire                 dma_result_valid_i,
    input  wire                 dma_result_last_i,
    output wire                 dma_result_ready_o,

    // Result storage interface
    output wire                 result_write_o,
    output wire [DMA_WIDTH-1:0] result_data_o,
    output wire [ADDR_WIDTH-1:0] result_addr_o
);

    // Internal connections between the two modules
    wire gemv_start_wire;
    wire gemv_done_wire;

    //==========================================================================
    // Attention Controller Instance
    //==========================================================================

    attention_qkt_ctrl u_attention_ctrl (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .start_i      (start_i),
        .gemv_done_i  (gemv_done_wire),
        .gemv_start_o (gemv_start_wire),
        .busy_o       (busy_o),
        .done_o       (done_o)
    );

    //==========================================================================
    // DMA Adapter Instance
    //==========================================================================

    attention_dma_adapter #(
        .DMA_WIDTH  (DMA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_dma_adapter (
        .clk_i               (clk_i),
        .rst_ni              (rst_ni),

        .gemv_start_i        (gemv_start_wire),
        .gemv_done_o         (gemv_done_wire),
        .busy_o              (adapter_busy_o),

        .stream_word_count_i (stream_word_count_i),
        .stream_addr_o       (stream_addr_o),
        .stream_data_i       (stream_data_i),

        .dma_ready_i         (dma_ready_i),
        .dma_data_o          (dma_data_o),
        .dma_valid_o         (dma_valid_o),

        .dma_result_data_i   (dma_result_data_i),
        .dma_result_valid_i  (dma_result_valid_i),
        .dma_result_last_i   (dma_result_last_i),
        .dma_result_ready_o  (dma_result_ready_o),

        .result_write_o      (result_write_o),
        .result_data_o       (result_data_o),
        .result_addr_o       (result_addr_o)
    );

endmodule