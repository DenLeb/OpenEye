"""Split two verified OpenEye GEMV streams into ROM/RAM images.

Phase-1 fixed-geometry baseline for the Attention ROM-override architecture.
The script changes neither OpenEye RTL nor the original reference files.

Run at the OpenEye repository root:
    python attention_rom_override_split.py

Defaults expect:
    test/cocotb_attention_qkt/data/gemv_reference.mem
    test/cocotb_attention_qkt/data/gemv_q1_reference.mem
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WORD_BITS = 64
EXPECTED_WORDS = 939
IACT_START, IACT_END = 19, 27     # [19, 26]: 8 DMA words
WGHT_START, WGHT_END = 27, 267   # [27, 266]: 240 DMA words


def read_mem(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference stream: {path}")
    words = []
    for line_num, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.split("//", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("0x"):
            line = line[2:]
        try:
            value = int(line.replace("_", ""), 16)
        except ValueError as exc:
            raise ValueError(f"Invalid hex at {path}:{line_num}: {line!r}") from exc
        if not 0 <= value < (1 << WORD_BITS):
            raise ValueError(f"Word outside {WORD_BITS}-bit range at {path}:{line_num}")
        words.append(value)
    if len(words) != EXPECTED_WORDS:
        raise ValueError(f"Expected {EXPECTED_WORDS} words in {path}; got {len(words)}")
    return words


def write_mem(path: Path, words: list[int]) -> None:
    path.write_text("".join(f"{value:016x}\n" for value in words), encoding="ascii")


def assemble(template: list[int], activations: list[int], weights: list[int]) -> list[int]:
    if len(activations) != IACT_END - IACT_START:
        raise ValueError("Unexpected activation RAM size")
    if len(weights) != WGHT_END - WGHT_START:
        raise ValueError("Unexpected weight RAM size")
    result = template.copy()
    result[IACT_START:IACT_END] = activations
    result[WGHT_START:WGHT_END] = weights
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test/cocotb_attention_qkt/data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    source_dir = args.data_dir
    output_dir = args.out_dir or source_dir / "rom_override_baseline"

    q0 = read_mem(source_dir / "gemv_reference.mem")
    q1 = read_mem(source_dir / "gemv_q1_reference.mem")

    differences = [i for i, (left, right) in enumerate(zip(q0, q1)) if left != right]
    outside_iact = [i for i in differences if not (IACT_START <= i < IACT_END)]
    if outside_iact:
        raise ValueError(
            f"Q0/Q1 differ outside the activation region at {outside_iact}. "
            "Cannot share a single common template/weight image yet."
        )

    common_rom = q0.copy()
    common_rom[IACT_START:WGHT_END] = [0] * (WGHT_END - IACT_START)
    activation_q0 = q0[IACT_START:IACT_END]
    activation_q1 = q1[IACT_START:IACT_END]
    weight_wq = q0[WGHT_START:WGHT_END]

    assert assemble(common_rom, activation_q0, weight_wq) == q0
    assert assemble(common_rom, activation_q1, weight_wq) == q1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_mem(output_dir / "common_rom.mem", common_rom)
    write_mem(output_dir / "activation_q0_ram.mem", activation_q0)
    write_mem(output_dir / "activation_q1_ram.mem", activation_q1)
    write_mem(output_dir / "weight_wq_ram.mem", weight_wq)

    manifest = {
        "dma_word_bits": WORD_BITS,
        "complete_stream_words": EXPECTED_WORDS,
        "common_rom_words": len(common_rom),
        "activation_override": {"start": IACT_START, "word_count": len(activation_q0)},
        "weight_override": {"start": WGHT_START, "word_count": len(weight_wq)},
        "q0_q1_differing_word_addresses": differences,
        "quantization": "Unchanged in the common ROM; supervisor reviewing optimization.",
        "skip_flags": "Baseline assumes all skip flags disabled.",
        "scope": "Verified Q0/Q1 only; dynamic Q/K/V mapping follows later.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("PASS: Both Q0 and Q1 streams reconstructed exactly (939 / 939 words each).")
    print(f"Differences between source streams: {differences}")
    print(f"Saved 4 .mem files and manifest.json to: {output_dir.resolve()}")
    print("Next: compare the generated images against the Attention Python mapper output.")


if __name__ == "__main__":
    main()
