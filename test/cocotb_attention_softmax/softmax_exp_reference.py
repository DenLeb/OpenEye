"""Bit-accurate software model for standalone OpenEye Softmax scaling+EXP stage."""
import math

LUT = [math.floor(math.exp(-i/8) * 32768 + 0.5) for i in range(65)]


def exp_row(scores, q_shift, k_shift):
    assert len(scores) == 4
    assert all(-(1 << 19) <= v < (1 << 19) for v in scores)
    assert 0 <= q_shift <= 13 and 0 <= k_shift <= 13
    max_score = max(scores)
    shift_right = 28 - (q_shift + k_shift)
    out = []
    for score in scores:
        if score == max_score:
            out.append(32768)
            continue
        magnitude = max_score - score
        q12_abs = (magnitude * 23170 + (1 << (shift_right - 1))) >> shift_right
        if q12_abs > 32768:
            out.append(0)
        else:
            addr = (q12_abs + 256) >> 9
            out.append(LUT[addr])
    return out


def softmax_ref(scores, q_shift, k_shift):
    """Floating-point reference using convention Q20_real=Q20/4096."""
    beta = 2 ** (q_shift + k_shift) / (4096**2 * math.sqrt(8))
    m = max(scores)
    terms = [math.exp((v - m) * beta) for v in scores]
    return [x / sum(terms) for x in terms]


if __name__ == '__main__':
    import numpy as np
    from pathlib import Path
    path = Path(__file__).resolve().parent / 'data' / 'qkt_t0_reference.npz'
    with np.load(path, allow_pickle=False) as f:
        score_matrix = f['Q8'].astype(np.int64) @ f['K8'].astype(np.int64).T
        qs, ks = int(f['q_shift']), int(f['k_shift'])
    for row in score_matrix:
        exp = exp_row([int(x) for x in row], qs, ks)
        print('scores:', row.tolist(), 'E_Q1.15:', exp, 'normalized:', [round(x/sum(exp), 6) for x in exp])
    assert exp_row([int(x) for x in score_matrix[0]], 6, 6) == [13660, 3914, 32768, 13660]
    assert exp_row([0,0,0,0], 6, 6) == [32768]*4
    assert exp_row([0,-100000,-200000,-300000], 6, 6) == [32768,0,0,0]
    print('PASS: Python reference and deterministic checks (not RTL simulation)')
