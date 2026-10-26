
`timescale 1ns/1ps

module attention_qkt_dma_memory #(
    parameter integer DMA_WIDTH = 64,
    parameter integer ADDR_WIDTH = 16,

    parameter integer STATUS_WORDS = 19,
    parameter integer IACT_WORDS = 4,
    parameter integer WGHT_WORDS = 10,
    parameter integer PSUM_WORDS = 4,
    parameter integer QUANT_WORDS = 640,

    parameter TEMPLATE_FILE = "qkt_t0_dma.mem",

    parameter integer WGHT_INDEX_BITS =
        (WGHT_WORDS <= 1) ? 1 : $clog2(WGHT_WORDS),

    parameter integer COUNT_BITS =
        $clog2(WGHT_WORDS + 1)
)(
    input wire clk_i,
    input wire rst_ni,

    // Clear packed operands before a new GEMV.
    input wire clear_i,

    // From Q8 activation packer
    input wire [DMA_WIDTH-1:0] q8_word_i,
    input wire q8_word_valid_i,
    output wire q8_word_ready_o,

    // From K8 weight packer
    input wire [DMA_WIDTH-1:0] k8_word_i,
    input wire k8_word_valid_i,
    output wire k8_word_ready_o,
    input wire [WGHT_INDEX_BITS-1:0] k8_word_index_i,

    // All operands are available
    output wire operands_ready_o,

    // Existing GEMV controller read interface
    input wire [ADDR_WIDTH-1:0] stream_addr_i,
    output reg [DMA_WIDTH-1:0] stream_data_o
);

    localparam integer IACT_START = STATUS_WORDS;
    localparam integer IACT_LAST = IACT_START + IACT_WORDS - 1;
    localparam integer WGHT_START = IACT_START + IACT_WORDS;
    localparam integer PSUM_START = WGHT_START + WGHT_WORDS;

    localparam integer STREAM_WORDS =
        PSUM_START + PSUM_WORDS + QUANT_WORDS;

    // Static QKT configuration, PSUM and quantization data.
    reg [DMA_WIDTH-1:0] template_rom [0:STREAM_WORDS-1];

    initial begin
        $readmemh(TEMPLATE_FILE, template_rom);
    end

    // Packed operands, not additional Q8/K8 memories.
    reg [DMA_WIDTH-1:0] q8_packed;
    reg [DMA_WIDTH-1:0] k8_packed [0:WGHT_WORDS-1];

    reg q8_loaded;
    reg [COUNT_BITS-1:0] wght_count;

    assign q8_word_ready_o = !q8_loaded;

    assign k8_word_ready_o =
        (wght_count < WGHT_WORDS) &&
        (k8_word_index_i == wght_count);

    assign operands_ready_o =
        q8_loaded && (wght_count == WGHT_WORDS);

    integer i;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            q8_packed <= 0;
            q8_loaded <= 0;
            wght_count <= 0;

            for (i = 0; i < WGHT_WORDS; i = i + 1)
                k8_packed[i] <= 0;

        end else if (clear_i) begin
            q8_packed <= 0;
            q8_loaded <= 0;
            wght_count <= 0;

            for (i = 0; i < WGHT_WORDS; i = i + 1)
                k8_packed[i] <= 0;

        end else begin

            if (q8_word_valid_i && q8_word_ready_o) begin
                q8_packed <= q8_word_i;
                q8_loaded <= 1'b1;
            end

            if (k8_word_valid_i && k8_word_ready_o) begin
                k8_packed[wght_count] <= k8_word_i;
                wght_count <= wght_count + 1'b1;
            end

        end
    end

    // Combinational read, compatible with the existing
    // attention_gemv_control_top stream interface.
    always @(*) begin
        stream_data_o = {DMA_WIDTH{1'b0}};

        if (operands_ready_o &&
            (stream_addr_i < STREAM_WORDS)) begin

            if (stream_addr_i >= IACT_START &&
                stream_addr_i < WGHT_START) begin

                if (stream_addr_i == IACT_LAST)
                    stream_data_o = q8_packed;
                else
                    stream_data_o = 0;

            end else if (stream_addr_i >= WGHT_START &&
                         stream_addr_i < PSUM_START) begin

                stream_data_o =
                    k8_packed[stream_addr_i - WGHT_START];

            end else begin
                stream_data_o = template_rom[stream_addr_i];
            end
        end
    end

endmodule
