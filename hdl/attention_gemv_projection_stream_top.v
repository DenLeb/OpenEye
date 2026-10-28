`timescale 1ns / 1ps

module attention_projection_stream_top #(
    parameter integer DMA_WIDTH    = 64,
    parameter integer ADDR_WIDTH   = 16,
    parameter integer STREAM_WORDS = 939
)(
        // Clock and control
    input wire clk_i,
    input wire rst_ni,
    input wire start_i,

    // Projection selection
    input wire [1:0] token_select_i,
    input wire [1:0] projection_select_i,

    // Execution status
    output wire busy_o,
    output wire adapter_busy_o,
    output wire done_o,

    // DMA transmit interface
    input  wire                 dma_ready_i,
    output wire [DMA_WIDTH-1:0] dma_data_o,
    output wire                 dma_valid_o,

    // DMA receive interface
    input  wire [DMA_WIDTH-1:0] dma_result_data_i,
    input  wire                 dma_result_valid_i,
    input  wire                 dma_result_last_i,
    output wire                 dma_result_ready_o,

    // Result storage interface
    output wire                  result_write_o,
    output wire [DMA_WIDTH-1:0]  result_data_o,
    output wire [ADDR_WIDTH-1:0] result_addr_o,

    // Activation RAM write interface
    input wire                  iact_we_i,
    input wire [ADDR_WIDTH-1:0] iact_wr_addr_i,
    input wire [DMA_WIDTH-1:0]  iact_wr_data_i,

    // Weight RAM write interface
    input wire                  wght_we_i,
    input wire [ADDR_WIDTH-1:0] wght_wr_addr_i,
    input wire [DMA_WIDTH-1:0]  wght_wr_data_i,

    // Debug
    output wire [ADDR_WIDTH-1:0] stream_addr_o
);

    wire [DMA_WIDTH-1:0] stream_data_wire;

    localparam [ADDR_WIDTH-1:0] STREAM_WORD_COUNT = STREAM_WORDS;

    attention_projection_memory #(
    .DMA_WIDTH  (DMA_WIDTH),
    .ADDR_WIDTH (ADDR_WIDTH)
    ) u_projection_memory (

        .clk_i                 (clk_i),
        .rst_ni                (rst_ni),
        .start_i               (start_i),
        .busy_i                (busy_o),

        .token_select_i        (token_select_i),
        .projection_select_i   (projection_select_i),

        .addr_i                (stream_addr_o),
        .data_o                (stream_data_wire),

        .iact_we_i             (iact_we_i),
        .iact_wr_addr_i        (iact_wr_addr_i),
        .iact_wr_data_i        (iact_wr_data_i),

        .wght_we_i             (wght_we_i),
        .wght_wr_addr_i        (wght_wr_addr_i),
        .wght_wr_data_i        (wght_wr_data_i)
    );

    attention_gemv_control_top #(
    .DMA_WIDTH  (DMA_WIDTH),
    .ADDR_WIDTH (ADDR_WIDTH)
    ) u_control_top (

        .clk_i               (clk_i),
        .rst_ni              (rst_ni),

        .start_i             (start_i),
        .busy_o              (busy_o),
        .done_o              (done_o),
        .adapter_busy_o      (adapter_busy_o),

        .stream_word_count_i (STREAM_WORD_COUNT),
        .stream_addr_o       (stream_addr_o),
        .stream_data_i       (stream_data_wire),

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