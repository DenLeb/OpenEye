`timescale 1ns / 1ps

module attention_gemv_rom #(
    parameter integer DMA_WIDTH  = 64,
    parameter integer ADDR_WIDTH = 16,
    parameter integer ROM_DEPTH  = 939,
    parameter MEM_FILE = "gemv_reference.mem"
)(
    input  wire [ADDR_WIDTH-1:0] addr_i,
    output wire [DMA_WIDTH-1:0]  data_o
);

    // Memory containing the reference GEMV DMA stream
    reg [DMA_WIDTH-1:0] rom [0:ROM_DEPTH-1];

    // Initialize ROM from the hexadecimal memory file
    initial begin
        $readmemh(MEM_FILE, rom);
    end

    // Return the requested word
    assign data_o = (addr_i < ROM_DEPTH)
                  ? rom[addr_i]
                  : {DMA_WIDTH{1'b0}};

endmodule