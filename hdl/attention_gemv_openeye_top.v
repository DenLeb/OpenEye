`timescale 1ns / 1ps

//==============================================================
// Module: attention_gemv_openeye_top
//
// Purpose:
//   Connect the verified GEMV ROM/control path to OpenEye_FPGA.
//
// This is an integration prototype for one fixed GEMV.
//==============================================================

module attention_gemv_openeye_top #(
    parameter integer DMA_WIDTH  = 64,
    parameter integer ADDR_WIDTH = 16,
    parameter integer ROM_DEPTH  = 939
)(
    input  wire clk_i,
    input  wire rst_ni,
    input  wire start_i,

    output wire busy_o,
    output wire adapter_busy_o,
    output wire done_o,

    // Result interface for verification
    output wire                 result_write_o,
    output wire [DMA_WIDTH-1:0] result_data_o,
    output wire [ADDR_WIDTH-1:0] result_addr_o,

    // Debug address
    output wire [ADDR_WIDTH-1:0] stream_addr_o
);

    //==========================================================
    // DMA connections between the adapter and OpenEye
    //==========================================================

    wire [DMA_WIDTH-1:0] dma_input_data;
    wire                 dma_input_valid;
    wire                 dma_input_ready;

    wire [DMA_WIDTH-1:0] dma_output_data;
    wire                 dma_output_valid;
    wire                 dma_output_last;
    wire                 dma_output_ready;

    //==========================================================
    // Controller + DMA Adapter + ROM
    //==========================================================

    attention_gemv_rom_top #(
        .DMA_WIDTH  (DMA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .ROM_DEPTH  (ROM_DEPTH)
    ) u_attention_stream (

        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .start_i        (start_i),

        .busy_o         (busy_o),
        .adapter_busy_o (adapter_busy_o),
        .done_o         (done_o),

        // Input DMA
        .dma_ready_i    (dma_input_ready),
        .dma_data_o     (dma_input_data),
        .dma_valid_o    (dma_input_valid),

        // Output DMA
        .dma_result_data_i  (dma_output_data),
        .dma_result_valid_i (dma_output_valid),
        .dma_result_last_i  (dma_output_last),
        .dma_result_ready_o (dma_output_ready),

        // Results
        .result_write_o (result_write_o),
        .result_data_o  (result_data_o),
        .result_addr_o  (result_addr_o),

        .stream_addr_o  (stream_addr_o)
    );

    //==========================================================
    // Real OpenEye accelerator
    //==========================================================

    OpenEye_FPGA u_openeye (

        .clk_i  (clk_i),
        .rst_ni (rst_ni),

        // DMA input
        .data_dma_i   (dma_input_data),
        .enable_dma_i (dma_input_valid),
        .ready_dma_o  (dma_input_ready),

        // DMA output
        .data_dma_o   (dma_output_data),
        .enable_dma_o (dma_output_valid),
        .last_data_o  (dma_output_last),
        .ready_dma_i  (dma_output_ready)
    );

endmodule