
`timescale 1ns / 1ps

module attention_softmax_row (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        start_i,

    input  wire [79:0] scores_i,
    input  wire [6:0]  q_shift_i,
    input  wire [6:0]  k_shift_i,

    output wire [63:0] probabilities_o,
    output wire        busy_o,
    output wire        done_o
);

    // FSM states
    localparam [2:0]
        IDLE      = 3'd0,
        EXP_START = 3'd1,
        EXP_WAIT  = 3'd2,
        NORM_WAIT = 3'd3,
        DONE      = 3'd4;

    reg [2:0] state;

    // Registered inputs
    reg [79:0] scores_reg;
    reg [6:0]  q_shift_reg;
    reg [6:0]  k_shift_reg;

    // Exponential interface
    wire [63:0] exp_values;
    wire        exp_valid;
    wire        exp_start;

    // Normalization interface
    wire norm_start;
    wire norm_done;
    wire norm_busy;

    assign exp_start  = (state == EXP_START);
    assign norm_start = (state == EXP_WAIT) && exp_valid;

    assign busy_o = (state != IDLE) && (state != DONE);
    assign done_o = (state == DONE);


    // Stage 1: Scaling + Exponential
    attention_softmax_exp u_exp (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (exp_start),

        .scores_i     (scores_reg),
        .q_shift_i    (q_shift_reg),
        .k_shift_i    (k_shift_reg),

        .valid_o      (exp_valid),
        .exp_values_o (exp_values)
    );


    // Stage 2: Normalization
    attention_softmax_normalize u_normalize (
        .clk_i           (clk_i),
        .rst_ni          (rst_ni),
        .start_i         (norm_start),

        .exp_values_i    (exp_values),

        .probabilities_o (probabilities_o),
        .busy_o          (norm_busy),
        .done_o          (norm_done)
    );


    // Row controller
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state       <= IDLE;
            scores_reg  <= 80'd0;
            q_shift_reg <= 7'd0;
            k_shift_reg <= 7'd0;
        end else begin
            case (state)

                IDLE: begin
                    if (start_i) begin
                        scores_reg  <= scores_i;
                        q_shift_reg <= q_shift_i;
                        k_shift_reg <= k_shift_i;
                        state       <= EXP_START;
                    end
                end

                EXP_START: begin
                    state <= EXP_WAIT;
                end

                EXP_WAIT: begin
                    if (exp_valid)
                        state <= NORM_WAIT;
                end

                NORM_WAIT: begin
                    if (norm_done)
                        state <= DONE;
                end

                DONE: begin
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
