`timescale 1ns / 1ps

module attention_softmax_divider (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        start_i,

    input  wire [15:0] exp_i,
    input  wire [17:0] sum_i,

    output wire        busy_o,
    output reg         done_o,
    output reg  [15:0] probability_o
);

    // FSM State Definitions
    localparam [1:0] IDLE   = 2'd0;
    localparam [1:0] DIVIDE = 2'd1;
    localparam [1:0] DONE   = 2'd2;

    reg [1:0]  state;

    // Registers for Division Operation
    reg [18:0] remainder;
    reg [17:0] divisor;
    reg [15:0] quotient;
    reg [4:0]  bit_count;


    assign busy_o = (state == DIVIDE);

    wire [18:0] shifted_remainder;
    wire        quotient_bit;
    wire [18:0] next_remainder;
    wire [15:0] next_quotient;
    wire        round_up;

    assign shifted_remainder = remainder << 1;

    assign quotient_bit =
        (shifted_remainder >= {1'b0, divisor});

    assign next_remainder = quotient_bit
        ? shifted_remainder - {1'b0, divisor}
        : shifted_remainder;

    assign next_quotient = {quotient[14:0], quotient_bit};

    assign round_up =
        ({next_remainder, 1'b0} >= {2'b00, divisor});

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state         <= IDLE;
            remainder     <= 19'd0;
            divisor       <= 18'd0;
            quotient      <= 16'd0;
            bit_count     <= 5'd0;
            probability_o <= 16'd0;
            done_o        <= 1'b0;
        end else begin
            done_o <= 1'b0;

            case (state)

                IDLE: begin
                    if (start_i) begin
                        if (sum_i == 0) begin
                            probability_o <= 16'd0;
                            state <= DONE;
                        end else if (exp_i == 0) begin
                            probability_o <= 16'd0;
                            state <= DONE;
                        end else if ({2'b00, exp_i} >= sum_i) begin
                            probability_o <= 16'd32768;
                            state <= DONE;
                        end else begin
                            remainder <= {3'b000, exp_i};
                            divisor   <= sum_i;
                            quotient  <= 16'd0;
                            bit_count <= 5'd0;
                            state     <= DIVIDE;
                        end
                    end
                end

                DIVIDE: begin
                    remainder <= next_remainder;
                    quotient  <= next_quotient;

                    if (bit_count == 5'd14) begin
                        probability_o <= next_quotient
                                    + (round_up ? 16'd1 : 16'd0);
                        state <= DONE;
                    end else begin
                        bit_count <= bit_count + 1'b1;
                    end
                end

                DONE: begin
                    done_o <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;

            endcase
        end
    end
endmodule