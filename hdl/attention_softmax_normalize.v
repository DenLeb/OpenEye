module attention_softmax_normalize (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        start_i,

    input  wire [63:0] exp_values_i,

    output wire [63:0] probabilities_o,
    output wire        busy_o,
    output wire        done_o
);

    wire [17:0] sum_next;

    assign sum_next =
        {2'b00, exp_values_i[15:0]}
        + {2'b00, exp_values_i[31:16]}
        + {2'b00, exp_values_i[47:32]}
        + {2'b00, exp_values_i[63:48]};

    reg [63:0] exp_row_reg;
    reg [17:0] sum_reg;
    
    localparam [1:0]
        IDLE   = 2'd0,
        LAUNCH = 2'd1,
        WAIT   = 2'd2,
        DONE   = 2'd3;

    reg [1:0] state;
    reg [3:0] done_seen;

    assign divider_start = (state == LAUNCH);

    assign busy_o = (state == LAUNCH) || (state == WAIT);

    assign done_o = (state == DONE);

    wire [3:0] divider_done;
    wire [3:0] divider_busy;
    wire       divider_start;

    genvar lane;

    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : GEN_DIVIDER

            attention_softmax_divider u_divider (

                .clk_i         (clk_i),
                .rst_ni        (rst_ni),
                .start_i       (divider_start),

                .exp_i         (exp_row_reg[lane*16 +: 16]),
                .sum_i         (sum_reg),

                .busy_o        (divider_busy[lane]),
                .done_o        (divider_done[lane]),

                .probability_o (probabilities_o[lane*16 +: 16])
            );

        end
    endgenerate

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state       <= IDLE;
            exp_row_reg <= 64'd0;
            sum_reg     <= 18'd0;
            done_seen   <= 4'b0000;
        end else begin

            case (state)

                IDLE: begin
                    if (start_i) begin
                        exp_row_reg <= exp_values_i;
                        sum_reg     <= sum_next;
                        done_seen   <= 4'b0000;
                        state       <= LAUNCH;
                    end
                end

                LAUNCH: begin
                    state <= WAIT;
                end

                WAIT: begin
                    done_seen <= done_seen | divider_done;

                    if (&(done_seen | divider_done)) begin
                        state <= DONE;
                    end
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