// OpenEye Q/K/V Projection: ROM Template + RAM Override (baseline v1)
// SPDX-License-Identifier: SHL-2.1
//
// Offline Python prepares the three memory images using GemmMapper.
// This module chooses the appropriate 64-bit DMA word at runtime.
//
// Projection IDs: 0=Q, 1=K, 2=V.
// Token IDs:      0..TOKEN_COUNT-1.
// Stream layout: status | iact | wght | psum | quantize.
//
// Compatibility with the existing combinational attention_gemv_rom.v:
// The read path is combinational. A future BRAM-optimized synchronous-read
// implementation needs a one-cycle look-ahead / valid-ready pipeline.
//
// The RAM write ports are provided for future memory updates. Keep both
// write enables low during the first projection ROM-override test.

`timescale 1ns / 1ps


module attention_projection_memory #(
    parameter integer DMA_WIDTH          = 64,
    parameter integer ADDR_WIDTH         = 16,
    parameter integer STATUS_WORDS       = 19,
    parameter integer IACT_WORDS         = 8,
    parameter integer WGHT_WORDS         = 240,
    parameter integer PSUM_WORDS         = 32,
    parameter integer QUANT_WORDS        = 640,
    parameter integer TOKEN_COUNT        = 4,
    parameter integer PROJECTION_COUNT   = 3,
    parameter COMMON_MEM_FILE            = "common_rom.mem",
    parameter ACTIVATION_MEM_FILE        = "activation_ram.mem",
    parameter WEIGHT_MEM_FILE            = "weight_ram.mem"
)(
    input  wire                     clk_i,
    input  wire                     rst_ni,
    input  wire                     start_i,
    input  wire                     busy_i,
    input  wire [1:0]               token_select_i,
    input  wire [1:0]               projection_select_i,
    input  wire [ADDR_WIDTH-1:0]    addr_i,
    output reg  [DMA_WIDTH-1:0]     data_o,

    // Optional write ports: unused for the first preinitialized-memory test.
    input  wire                     iact_we_i,
    input  wire [ADDR_WIDTH-1:0]    iact_wr_addr_i,
    input  wire [DMA_WIDTH-1:0]     iact_wr_data_i,
    input  wire                     wght_we_i,
    input  wire [ADDR_WIDTH-1:0]    wght_wr_addr_i,
    input  wire [DMA_WIDTH-1:0]     wght_wr_data_i
);

    localparam integer IACT_START       = STATUS_WORDS;
    localparam integer WGHT_START       = IACT_START + IACT_WORDS;
    localparam integer PSUM_START       = WGHT_START + WGHT_WORDS;
    localparam integer QUANT_START      = PSUM_START + PSUM_WORDS;
    localparam integer STREAM_WORDS     = QUANT_START + QUANT_WORDS;
    localparam integer IACT_RAM_DEPTH   = TOKEN_COUNT * IACT_WORDS;
    localparam integer COMMON_ROM_DEPTH = STATUS_WORDS + PSUM_WORDS + QUANT_WORDS;
    localparam integer WGHT_RAM_DEPTH   = PROJECTION_COUNT * WGHT_WORDS;

    reg [DMA_WIDTH-1:0] common_rom [0:COMMON_ROM_DEPTH-1];
    reg [DMA_WIDTH-1:0] activation_ram [0:IACT_RAM_DEPTH-1];
    reg [DMA_WIDTH-1:0] weight_ram     [0:WGHT_RAM_DEPTH-1];

    reg [1:0] active_token_reg;
    reg [1:0] active_projection_reg;

    // Read all offline-generated images. Simulation working directory must
    // contain these files (the provided test runner copies them there).
    initial begin
        $readmemh(COMMON_MEM_FILE, common_rom);
        $readmemh(ACTIVATION_MEM_FILE, activation_ram);
        $readmemh(WEIGHT_MEM_FILE, weight_ram);
    end

    // Latch the next operation's selectors when idle. This prevents a
    // mid-stream change of external selection inputs from corrupting a GEMV.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_token_reg      <= 2'd0;
            active_projection_reg <= 2'd0;
        end else if (start_i && !busy_i) begin
            active_token_reg      <= token_select_i;
            active_projection_reg <= projection_select_i;
        end
    end

    // Preinitialized, but writable RAMs. Do not write to them while a
    // projection is reading the same addresses in the baseline experiment.
    always @(posedge clk_i) begin
        if (iact_we_i && (iact_wr_addr_i < IACT_RAM_DEPTH))
            activation_ram[iact_wr_addr_i] <= iact_wr_data_i;
        if (wght_we_i && (wght_wr_addr_i < WGHT_RAM_DEPTH))
            weight_ram[wght_wr_addr_i] <= wght_wr_data_i;
    end

    wire [ADDR_WIDTH-1:0] iact_index =
        (active_token_reg * IACT_WORDS) + (addr_i - IACT_START);
    wire [ADDR_WIDTH-1:0] wght_index =
        (active_projection_reg * WGHT_WORDS) + (addr_i - WGHT_START);

    always @(*) begin
        data_o = {DMA_WIDTH{1'b0}};
        if (addr_i < STREAM_WORDS) begin
            if ((addr_i >= IACT_START) && (addr_i < WGHT_START)) begin
                if ((active_token_reg < TOKEN_COUNT) &&
                    (iact_index < IACT_RAM_DEPTH))
                    data_o = activation_ram[iact_index];
            end else if ((addr_i >= WGHT_START) && (addr_i < PSUM_START)) begin
                if ((active_projection_reg < PROJECTION_COUNT) &&
                    (wght_index < WGHT_RAM_DEPTH))
                    data_o = weight_ram[wght_index];
            end else if (addr_i < IACT_START) begin
                data_o = common_rom[addr_i];

            end else begin
                data_o = common_rom[addr_i - IACT_WORDS - WGHT_WORDS];
            end
        end
    end
endmodule
