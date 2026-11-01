`timescale 1ns / 1ps
// Standalone stage 1 for 4-way row-parallel Softmax:
// max subtraction -> shift-aware Q4.12 scaling -> 65-entry exp LUT.
// Does NOT change OpenEye, score buffer, or scheduler.
// Packed inputs: scores_i[19:0] is score 0, scores_i[39:20] score 1, etc.
// Packed outputs: exp_values_o[15:0] is E0, [31:16] E1, etc.
// All ROM outputs are unsigned Q1.15: encoded 1.0 = 32768.
// Numerical test convention: Q20_real = Q20/4096, K20_real = K20/4096.
// Therefore: L_q12 = round((score - max)*2^(q_shift+k_shift-12)/sqrt(8)).
// Constant 23170/65536 approximates 1/sqrt(8).
// q_shift,k_shift are 0..13 for the existing 20-bit Q/K quantizer.
module attention_softmax_exp (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         valid_i,
    input  wire [79:0]  scores_i,
    input  wire [6:0]   q_shift_i,
    input  wire [6:0]   k_shift_i,
    output reg          valid_o,
    output reg [63:0]   exp_values_o
);
    function [15:0] lut_exp;
        input [6:0] addr;
        begin
            case (addr)
                7'd0: lut_exp = 16'd32768;
                7'd1: lut_exp = 16'd28918;
                7'd2: lut_exp = 16'd25520;
                7'd3: lut_exp = 16'd22521;
                7'd4: lut_exp = 16'd19875;
                7'd5: lut_exp = 16'd17539;
                7'd6: lut_exp = 16'd15479;
                7'd7: lut_exp = 16'd13660;
                7'd8: lut_exp = 16'd12055;
                7'd9: lut_exp = 16'd10638;
                7'd10: lut_exp = 16'd9388;
                7'd11: lut_exp = 16'd8285;
                7'd12: lut_exp = 16'd7312;
                7'd13: lut_exp = 16'd6452;
                7'd14: lut_exp = 16'd5694;
                7'd15: lut_exp = 16'd5025;
                7'd16: lut_exp = 16'd4435;
                7'd17: lut_exp = 16'd3914;
                7'd18: lut_exp = 16'd3454;
                7'd19: lut_exp = 16'd3048;
                7'd20: lut_exp = 16'd2690;
                7'd21: lut_exp = 16'd2374;
                7'd22: lut_exp = 16'd2095;
                7'd23: lut_exp = 16'd1849;
                7'd24: lut_exp = 16'd1631;
                7'd25: lut_exp = 16'd1440;
                7'd26: lut_exp = 16'd1271;
                7'd27: lut_exp = 16'd1121;
                7'd28: lut_exp = 16'd990;
                7'd29: lut_exp = 16'd873;
                7'd30: lut_exp = 16'd771;
                7'd31: lut_exp = 16'd680;
                7'd32: lut_exp = 16'd600;
                7'd33: lut_exp = 16'd530;
                7'd34: lut_exp = 16'd467;
                7'd35: lut_exp = 16'd412;
                7'd36: lut_exp = 16'd364;
                7'd37: lut_exp = 16'd321;
                7'd38: lut_exp = 16'd283;
                7'd39: lut_exp = 16'd250;
                7'd40: lut_exp = 16'd221;
                7'd41: lut_exp = 16'd195;
                7'd42: lut_exp = 16'd172;
                7'd43: lut_exp = 16'd152;
                7'd44: lut_exp = 16'd134;
                7'd45: lut_exp = 16'd118;
                7'd46: lut_exp = 16'd104;
                7'd47: lut_exp = 16'd92;
                7'd48: lut_exp = 16'd81;
                7'd49: lut_exp = 16'd72;
                7'd50: lut_exp = 16'd63;
                7'd51: lut_exp = 16'd56;
                7'd52: lut_exp = 16'd49;
                7'd53: lut_exp = 16'd43;
                7'd54: lut_exp = 16'd38;
                7'd55: lut_exp = 16'd34;
                7'd56: lut_exp = 16'd30;
                7'd57: lut_exp = 16'd26;
                7'd58: lut_exp = 16'd23;
                7'd59: lut_exp = 16'd21;
                7'd60: lut_exp = 16'd18;
                7'd61: lut_exp = 16'd16;
                7'd62: lut_exp = 16'd14;
                7'd63: lut_exp = 16'd12;
                7'd64: lut_exp = 16'd11;
                default: lut_exp = 16'd0;
            endcase
        end
    endfunction

    function [15:0] exp_from_score;
        input signed [19:0] score;
        input signed [19:0] row_max;
        input [7:0] total_shift;
        reg signed [20:0] diff;
        reg [20:0] distance;
        reg [36:0] product;
        reg [36:0] magnitude_q12;
        reg [7:0] right_shift;
        reg [6:0] address;
        begin
            diff = $signed({score[19], score}) - $signed({row_max[19], row_max});
            if (diff >= 0) begin
                // Exactly e^0 for the row maximum (also handles all-equal rows).
                exp_from_score = 16'd32768;
            end else if (total_shift > 8'd26) begin
                // Invalid shift combination for this 20-bit-quantizer baseline.
                // This guard avoids a negative shift count; legitimate inputs never reach it.
                exp_from_score = 16'd0;
            end else begin
                distance = -diff;
                product = distance * 37'd23170;
                // Q4.12 magnitude with round-to-nearest, ties up:
                // abs(delta) * (23170/65536) * 2^(total_shift-12) * 4096
                right_shift = 8'd28 - total_shift;
                magnitude_q12 = (product + (37'd1 << (right_shift - 1'b1))) >> right_shift;
                if (magnitude_q12 > 37'd32768) begin
                    // x < -8 -> exp(x) approximated as zero, NOT exp(-8).
                    exp_from_score = 16'd0;
                end else begin
                    // 0.125 spacing = 512 units in Q4.12; nearest-neighbor lookup.
                    address = (magnitude_q12 + 37'd256) >> 9;
                    // Due to input range, address stays in [0, 64].
                    exp_from_score = lut_exp(address);
                end
            end
        end
    endfunction

    reg signed [19:0] row_max;
    reg [7:0] total_shift;
    reg [63:0] exp_next;
    integer i;
    always @(*) begin
        row_max = $signed(scores_i[19:0]);
        for (i = 1; i < 4; i = i + 1) begin
            if ($signed(scores_i[i*20 +: 20]) > row_max)
                row_max = $signed(scores_i[i*20 +: 20]);
        end
        total_shift = {1'b0, q_shift_i} + {1'b0, k_shift_i};
        exp_next = 64'd0;
        for (i = 0; i < 4; i = i + 1)
            exp_next[i*16 +: 16] = exp_from_score(
                $signed(scores_i[i*20 +: 20]), row_max, total_shift
            );
    end

    // One registered output per valid input row; no busy state or row storage yet.
    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= 1'b0;
            exp_values_o <= 64'd0;
        end else begin
            valid_o <= valid_i;
            if (valid_i) exp_values_o <= exp_next;
        end
    end
endmodule
