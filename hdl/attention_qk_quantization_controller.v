// Attention Q/K INT8 quantization controller.
// Reuses the UNMODIFIED OpenEye quantization_unit.v (two instances).
// Raw Q and K ports assume combinational/asynchronous reads only in PSUM_SINGLE_PORT = 0, as implemented
// by attention_projection_result_buffer.v.
// Power-of-two per-tensor scale is chosen in RTL from max(abs(raw)).
// For bit-exact round-half-away-from-zero: mantissa=1, sign-dependent offset.
// Designed for Q,K [NUM_TOKENS x NUM_FEATURES]; defaults to 4 x 8.
`timescale 1ns/1ps

module attention_qk_quantization_controller #(
    parameter integer NUM_TOKENS = 4,
    parameter integer NUM_FEATURES = 8,
    parameter integer DATA_WIDTH = 20,
    parameter integer SHIFT_WIDTH = 7,
    parameter integer PSUM_SINGLE_PORT = 0,
    parameter integer TOKEN_BITS = (NUM_TOKENS <= 1) ? 1 : $clog2(NUM_TOKENS),
    parameter integer FEATURE_BITS = (NUM_FEATURES <= 1) ? 1 : $clog2(NUM_FEATURES),
    parameter integer TOTAL = NUM_TOKENS * NUM_FEATURES,
    parameter integer INDEX_BITS = (TOTAL <= 1) ? 1 : $clog2(TOTAL)
) (
    input wire clk_i,
    input wire rst_ni,
    input wire start_i,
    output reg busy_o,
    output reg done_o, // one-clock completion pulse

    // Read addresses into the existing RAW Q/K result buffer.
    output wire [TOKEN_BITS-1:0] q_raw_rd_token_o,
    output wire [FEATURE_BITS-1:0] q_raw_rd_feature_o,
    output wire [TOKEN_BITS-1:0] k_raw_rd_token_o,
    output wire [FEATURE_BITS-1:0] k_raw_rd_feature_o,
    input wire signed [DATA_WIDTH-1:0] q_raw_data_i,
    input wire signed [DATA_WIDTH-1:0] k_raw_data_i,

    // Direct read interface to OpenEye internal PSUM RAM
    output wire psum_rd_req_o,
    output wire [3:0] psum_rd_page_o,
    output wire [2:0] psum_rd_feature_o,

    input wire psum_rd_ready_i,
    input wire psum_rd_valid_i,
    input wire signed [DATA_WIDTH-1:0] psum_rd_data_i,

    // Independent read ports for quantized Q and K.
    input wire [TOKEN_BITS-1:0] q8_rd_token_i,
    input wire [FEATURE_BITS-1:0] q8_rd_feature_i,
    input wire [TOKEN_BITS-1:0] k8_rd_token_i,
    input wire [FEATURE_BITS-1:0] k8_rd_feature_i,
    output wire signed [7:0] q8_data_o,
    output wire signed [7:0] k8_data_o,
    output reg [SHIFT_WIDTH-1:0] q_shift_o,
    output reg [SHIFT_WIDTH-1:0] k_shift_o
);
    localparam [3:0]
        IDLE          = 4'd0,
        SCAN          = 4'd1,
        PICK_SHIFT    = 4'd2,
        FEED          = 4'd3,
        DRAIN         = 4'd4,

        PSUM_REQ_Q    = 4'd5,
        PSUM_WAIT_Q   = 4'd6,
        PSUM_REQ_K    = 4'd7,
        PSUM_WAIT_K   = 4'd8,
        PSUM_USE_PAIR = 4'd9;

    reg [3:0] state;

    reg psum_feed_phase;

    reg signed [DATA_WIDTH-1:0] psum_q_reg;
    reg signed [DATA_WIDTH-1:0] psum_k_reg;
    reg [INDEX_BITS-1:0] scan_index;
    reg [INDEX_BITS-1:0] feed_index;
    reg [DATA_WIDTH:0] q_max_abs;
    reg [DATA_WIDTH:0] k_max_abs;

    // Delayed valid/address align with quantization_unit's 3-edge pipeline.
    reg [2:0] valid_pipe;
    reg [INDEX_BITS-1:0] addr_pipe0, addr_pipe1, addr_pipe2;

    reg signed [7:0] q8_mem [0:TOTAL-1];
    reg signed [7:0] k8_mem [0:TOTAL-1];

    wire [INDEX_BITS-1:0] raw_read_index =
        PSUM_SINGLE_PORT
            ? (psum_feed_phase ? feed_index : scan_index)
            : ((state == SCAN) ? scan_index : feed_index);

    wire [TOKEN_BITS-1:0] psum_token =
        raw_read_index / NUM_FEATURES;

    wire [FEATURE_BITS-1:0] psum_feature =
        raw_read_index % NUM_FEATURES;

    assign psum_rd_req_o =
        PSUM_SINGLE_PORT &&
        ((state == PSUM_REQ_Q) ||
        (state == PSUM_REQ_K));

    assign psum_rd_page_o = {
        (state == PSUM_REQ_K ? 2'd1 : 2'd0),
        psum_token
    };

    assign psum_rd_feature_o = psum_feature;

    wire signed [DATA_WIDTH-1:0] q_raw_sample =
    PSUM_SINGLE_PORT ? psum_q_reg : q_raw_data_i;

    wire signed [DATA_WIDTH-1:0] k_raw_sample =
        PSUM_SINGLE_PORT ? psum_k_reg : k_raw_data_i;

    assign q_raw_rd_token_o = raw_read_index / NUM_FEATURES;
    assign q_raw_rd_feature_o = raw_read_index % NUM_FEATURES;
    assign k_raw_rd_token_o = raw_read_index / NUM_FEATURES;
    assign k_raw_rd_feature_o = raw_read_index % NUM_FEATURES;

    wire [INDEX_BITS-1:0] q8_read_index =
        q8_rd_token_i * NUM_FEATURES + q8_rd_feature_i;
    wire [INDEX_BITS-1:0] k8_read_index =
        k8_rd_token_i * NUM_FEATURES + k8_rd_feature_i;
    assign q8_data_o = q8_mem[q8_read_index];
    assign k8_data_o = k8_mem[k8_read_index];

    // Widen BEFORE abs(), so abs(minimum signed 20-bit value) is valid.
    function [DATA_WIDTH:0] magnitude;
        input signed [DATA_WIDTH-1:0] value;
        reg signed [DATA_WIDTH:0] extended;
        begin
            extended = {value[DATA_WIDTH-1], value};
            magnitude = (extended < 0) ? -extended : extended;
        end
    endfunction

    // Smallest shift s with max(abs(value)) <= 127 * 2^s.
    // For signed 20-bit raw data, s<=13. A signed W-bit value
    // requires at most W-7 bits of downshift to fit signed INT8.
    // The loop deliberately stops there to avoid shift-threshold overflow.
    function [SHIFT_WIDTH-1:0] choose_shift;
        input [DATA_WIDTH:0] maximum;
        integer s;
        reg [DATA_WIDTH:0] threshold;
        begin
            choose_shift = 0;
            for (s=0; s<=DATA_WIDTH-7; s=s+1) begin
                threshold = {{(DATA_WIDTH-6){1'b0}},7'd127} << s;
                if (maximum > threshold)
                    choose_shift = s + 1;
            end
        end
    endfunction

    wire [DATA_WIDTH:0] q_abs_now = magnitude(q_raw_sample);
    wire [DATA_WIDTH:0] k_abs_now = magnitude(k_raw_sample);

    // A +half offset for nonnegative data, +(half-1) for negative.
    // This converts arithmetic >>> (floor) into round-half-away-from-zero.
    wire signed [DATA_WIDTH-1:0] q_offset =
        (q_shift_o == 0) ? {DATA_WIDTH{1'b0}} :
        (($signed({{(DATA_WIDTH-1){1'b0}},1'b1}) <<< (q_shift_o-1'b1))
            - (q_raw_sample < 0 ? 1'b1 : 1'b0));
    wire signed [DATA_WIDTH-1:0] k_offset =
        (k_shift_o == 0) ? {DATA_WIDTH{1'b0}} :
        (($signed({{(DATA_WIDTH-1){1'b0}},1'b1}) <<< (k_shift_o-1'b1))
            - (k_raw_sample < 0 ? 1'b1 : 1'b0));

    wire signed [7:0] quantized_q;
    wire signed [7:0] quantized_k;
    quantization_unit #(
        .PSUM_WIDTH(DATA_WIDTH), .MANT_WIDTH(25),
        .OFFSET_WIDTH(DATA_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH), .OUT_WIDTH(8)
    ) u_quant_q (
        .clk_i(clk_i), .rst_n(rst_ni), .quant_mant(25'sd1),
        .psum_data(q_raw_sample), .quant_offset(q_offset),
        .current_shift(q_shift_o), .quantized_value(quantized_q)
    );
    quantization_unit #(
        .PSUM_WIDTH(DATA_WIDTH), .MANT_WIDTH(25),
        .OFFSET_WIDTH(DATA_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH), .OUT_WIDTH(8)
    ) u_quant_k (
        .clk_i(clk_i), .rst_n(rst_ni), .quant_mant(25'sd1),
        .psum_data(k_raw_sample), .quant_offset(k_offset),
        .current_shift(k_shift_o), .quantized_value(quantized_k)
    );

    integer i;
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state <= IDLE;
            scan_index <= 0;
            feed_index <= 0;
            q_max_abs <= 0;
            k_max_abs <= 0;
            q_shift_o <= 0;
            k_shift_o <= 0;
            psum_feed_phase <= 1'b0;
            psum_q_reg <= 0;
            psum_k_reg <= 0;
            valid_pipe <= 0;
            addr_pipe0 <= 0;
            addr_pipe1 <= 0;
            addr_pipe2 <= 0;
            busy_o <= 0;
            done_o <= 0;
            for (i=0; i<TOTAL; i=i+1) begin
                q8_mem[i] <= 0;
                k8_mem[i] <= 0;
            end
        end else begin
            done_o <= 0;

            // The quantization unit produces sample n's value on its THIRD
            // rising edge. A downstream clocked RAM writes it on the NEXT edge,
            // when the third valid/address pipeline slot is high.
            valid_pipe <= {
                valid_pipe[1:0],
                (state == FEED) ||
                (PSUM_SINGLE_PORT &&
                state == PSUM_USE_PAIR &&
                psum_feed_phase)
            };
            addr_pipe2 <= addr_pipe1;
            addr_pipe1 <= addr_pipe0;
            addr_pipe0 <= feed_index;
            if (valid_pipe[2]) begin
                q8_mem[addr_pipe2] <= quantized_q;
                k8_mem[addr_pipe2] <= quantized_k;
            end

            case (state)
                IDLE: begin
                    if (start_i) begin
                        state <= PSUM_SINGLE_PORT ? PSUM_REQ_Q : SCAN;
                        psum_feed_phase <= 1'b0;
                        busy_o <= 1;
                        scan_index <= 0;
                        feed_index <= 0;
                        q_max_abs <= 0;
                        k_max_abs <= 0;
                        valid_pipe <= 0;
                    end
                end
                SCAN: begin
                    if (q_abs_now > q_max_abs) q_max_abs <= q_abs_now;
                    if (k_abs_now > k_max_abs) k_max_abs <= k_abs_now;
                    if (scan_index == TOTAL-1) begin
                        state <= PICK_SHIFT;
                    end else begin
                        scan_index <= scan_index + 1'b1;
                    end
                end
                PICK_SHIFT: begin
                    q_shift_o <= choose_shift(q_max_abs);
                    k_shift_o <= choose_shift(k_max_abs);
                    feed_index <= 0;
                    psum_feed_phase <= 1'b1;
                    state <= PSUM_SINGLE_PORT ? PSUM_REQ_Q : FEED;
                end
                FEED: begin
                    if (feed_index == TOTAL-1)
                        state <= DRAIN;
                    else
                        feed_index <= feed_index + 1'b1;
                end
                DRAIN: begin
                    if (valid_pipe == 3'b000) begin
                        state <= IDLE;
                        busy_o <= 0;
                        done_o <= 1;
                    end
                end
                PSUM_REQ_Q: begin
                    if (psum_rd_ready_i)
                        state <= PSUM_WAIT_Q;
                end

                PSUM_WAIT_Q: begin
                    if (psum_rd_valid_i) begin
                        psum_q_reg <= psum_rd_data_i;
                        state <= PSUM_REQ_K;
                    end
                end

                PSUM_REQ_K: begin
                    if (psum_rd_ready_i)
                        state <= PSUM_WAIT_K;
                end

                PSUM_WAIT_K: begin
                    if (psum_rd_valid_i) begin
                        psum_k_reg <= psum_rd_data_i;
                        state <= PSUM_USE_PAIR;
                    end
                end

                PSUM_USE_PAIR: begin
                    if (!psum_feed_phase) begin

                        // SCAN: update maximum absolute values
                        if (q_abs_now > q_max_abs)
                            q_max_abs <= q_abs_now;

                        if (k_abs_now > k_max_abs)
                            k_max_abs <= k_abs_now;

                        if (scan_index == TOTAL-1) begin
                            state <= PICK_SHIFT;
                        end else begin
                            scan_index <= scan_index + 1'b1;
                            state <= PSUM_REQ_Q;
                        end

                    end else begin

                        // FEED: Q and K are both available
                        if (feed_index == TOTAL-1) begin
                            state <= DRAIN;
                        end else begin
                            feed_index <= feed_index + 1'b1;
                            state <= PSUM_REQ_Q;
                        end

                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
