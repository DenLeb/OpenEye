
module attention_softmax (
    input  wire        clk_i,
    input  wire        rst_ni,
    input  wire        start_i,

    // Quantization shifts
    input  wire [6:0]  q_shift_i,
    input  wire [6:0]  k_shift_i,

    // QKT Score Buffer read interface
    output wire [1:0]  score_rd_token_o,
    output wire [1:0]  score_rd_index_o,
    input  wire signed [19:0] score_rd_data_i,
    input  wire        score_rd_valid_i,

    // Probability Buffer read interface
    input  wire [1:0]  prob_rd_token_i,
    input  wire [1:0]  prob_rd_index_i,
    output wire [15:0] prob_rd_data_o,
    output wire        prob_rd_valid_o,

    // Control
    output wire        busy_o,
    output wire        done_o,
    output wire        complete_o
);

    
    localparam [2:0]
        IDLE      = 3'd0,
        LOAD_ROW  = 3'd1,
        ROW_START = 3'd2,
        ROW_WAIT  = 3'd3,
        STORE_ROW = 3'd4,
        FINISH    = 3'd5;

    reg [2:0] state;

    // Current position in the 4x4 matrix
    reg [1:0] row_index;
    reg [1:0] col_index;

    // Four 20-bit scores for the current row
    reg [79:0] row_scores_reg;

    // Capture quantization shifts at start
    reg [6:0] q_shift_reg;
    reg [6:0] k_shift_reg;

    // Result storage: 16 probabilities
    reg [15:0] probability_mem [0:15];
    reg [15:0] probability_valid;

    // Row processor interface
    wire [63:0] row_probabilities;
    wire        row_start;
    wire        row_busy;
    wire        row_done;

    // Score Buffer read address
    assign score_rd_token_o = row_index;
    assign score_rd_index_o = col_index;

    // Row processor control
    assign row_start = (state == ROW_START);

    assign busy_o = (state != IDLE) && (state != FINISH);
    assign done_o = (state == FINISH);

    // All 16 probabilities are available
    assign complete_o = &probability_valid;


    // Row-Parallel Softmax processor
    attention_softmax_row u_softmax_row (
        .clk_i           (clk_i),
        .rst_ni          (rst_ni),
        .start_i         (row_start),

        .scores_i        (row_scores_reg),
        .q_shift_i       (q_shift_reg),
        .k_shift_i       (k_shift_reg),

        .probabilities_o (row_probabilities),
        .busy_o          (row_busy),
        .done_o          (row_done)
    );

    wire [3:0] prob_read_addr;

    assign prob_read_addr = {
        prob_rd_token_i,
        prob_rd_index_i
    };

    assign prob_rd_valid_o = probability_valid[prob_read_addr];

    assign prob_rd_data_o = prob_rd_valid_o
                        ? probability_mem[prob_read_addr]
                        : 16'd0;

                        
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state             <= IDLE;
            row_index         <= 2'd0;
            col_index         <= 2'd0;
            row_scores_reg    <= 80'd0;
            q_shift_reg       <= 7'd0;
            k_shift_reg       <= 7'd0;
            probability_valid <= 16'd0;
        end else begin
            case (state)

                IDLE: begin
                    if (start_i) begin
                        row_index         <= 2'd0;
                        col_index         <= 2'd0;
                        row_scores_reg    <= 80'd0;
                        q_shift_reg       <= q_shift_i;
                        k_shift_reg       <= k_shift_i;
                        probability_valid <= 16'd0;
                        state             <= LOAD_ROW;
                    end
                end

                LOAD_ROW: begin
                    if (score_rd_valid_i) begin
                        row_scores_reg[col_index*20 +: 20]
                            <= score_rd_data_i;

                        if (col_index == 2'd3) begin
                            col_index <= 2'd0;
                            state     <= ROW_START;
                        end else begin
                            col_index <= col_index + 1'b1;
                        end
                    end
                end

                ROW_START: begin
                    state <= ROW_WAIT;
                end

                ROW_WAIT: begin
                    if (row_done) begin
                        state <= STORE_ROW;
                    end
                end

                STORE_ROW: begin
                    probability_mem[{row_index, 2'b00}]
                        <= row_probabilities[15:0];

                    probability_mem[{row_index, 2'b01}]
                        <= row_probabilities[31:16];

                    probability_mem[{row_index, 2'b10}]
                        <= row_probabilities[47:32];

                    probability_mem[{row_index, 2'b11}]
                        <= row_probabilities[63:48];

                    probability_valid[{row_index, 2'b00} +: 4]
                        <= 4'b1111;

                    if (row_index == 2'd3) begin
                        state <= FINISH;
                    end else begin
                        row_index      <= row_index + 1'b1;
                        col_index      <= 2'd0;
                        row_scores_reg <= 80'd0;
                        state          <= LOAD_ROW;
                    end
                end

                FINISH: begin
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end
endmodule
