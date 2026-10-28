"""OpenEye Attention Q/K/V projection stream and ROM-override image generator.

First-stage, INT8, fixed-geometry implementation.  Uses OpenEye's existing
DRAMContents, LayerParameters, and DenseMapper; no changes to OpenEye RTL.

By default, 4x8 input X and three 8x8 projection matrices are generated
with a reproducible random seed. Alternatively, use --input-npz containing
X, WQ, WK, WV (with W matrices in mathematical [input, output] layout).

The output is an *offline initialization image*, NOT a runtime Python scheduler.
All skip flags remain 0 until hardware reuse is independently verified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Keep these imports from the same OpenEye checkout used for RTL simulation.
from open_eye import DRAM
from open_eye import layer_parameters as lp
from open_eye import open_eye_parameters as oep
from open_eye import stream_dicts as strdic
from open_eye.dense_mapper import DenseMapper


SECTION_NAMES = ("status", "iact", "wght", "psum", "quantize")
WORD_MASK = (1 << 64) - 1
MIN_PAD_K = MIN_PAD_N = 32  # Existing attention cocotb test's proven envelope.

# Matches the public cocotb GEMM test platform; only used when unset in shell.
BASELINE_ENV = {
    "CLUSTER_COLUMNS": "2", "CLUSTER_ROWS": "2",
    "NUM_GLB_IACT": "3", "NUM_GLB_WGHT": "3", "NUM_GLB_PSUM": "4",
    "PARALLEL_MACS": "2", "BRANCHES": "1", "BUFFER_WIDTH": "12",
    "RAM_CELLS": "32", "QUANT_AMOUNT": "1024", "DATAFLOW": "row stationary",
}


def _layer_shim(k: int, n: int) -> SimpleNamespace:
    """Supply just the Dense layer interface expected by LayerParameters."""
    return SimpleNamespace(
        name="dense_attention_projection",
        input=SimpleNamespace(shape=(1, k)),
        output=SimpleNamespace(shape=(1, n)),
        kernel=SimpleNamespace(shape=(k, n)),
    )


def _as_int8_matrix(name: str, data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(data)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer operands, got {values.dtype}")
    if np.any(values < -128) or np.any(values > 127):
        raise ValueError(f"{name} contains values outside signed INT8 range")
    return values.astype(np.int64)


def input_tensors(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if args.input_npz:
        with np.load(args.input_npz, allow_pickle=False) as data:
            required = ("X", "WQ", "WK", "WV")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"Input NPZ is missing {missing}; expected {required}")
            x = _as_int8_matrix("X", data["X"], (args.seq_len, args.d_model))
            matrices = {
                kind: _as_int8_matrix(key, data[key], (args.d_model, args.d_model))
                for kind, key in (("q", "WQ"), ("k", "WK"), ("v", "WV"))
            }
    else:
        rng = np.random.default_rng(args.seed)
        x = rng.integers(-64, 64, size=(args.seq_len, args.d_model), dtype=np.int64)
        matrices = {
            kind: rng.integers(-64, 64, size=(args.d_model, args.d_model), dtype=np.int64)
            for kind in ("q", "k", "v")
        }
    return x, matrices


class AttentionProjectionMapper:
    """Adapt OpenEye's GEMM stream mapper to autonomous Q/K/V projection setup.

    One GEMV pass at a time is generated for the *existing* GEMM datapath.
    This is not an implementation of true batched/multi-token GEMM.
    """

    def __init__(self, params: object, k_pad: int = 32, n_pad: int = 32):
        if not params.SERIAL or int(params.DMA_BITWIDTH) != 64:
            raise ValueError("This ROM-override baseline requires SERIAL=1, DMA_BITWIDTH=64")
        self.params = params
        self.k_pad = k_pad
        self.n_pad = n_pad

    def make_pass(self, iact: np.ndarray, weights_out_in: np.ndarray) -> dict[str, list[int]]:
        """Build one complete, zero-skip GEMV DMA stream, preserving segment boundaries.

        iact has shape [K], and weights_out_in has shape [N, K].
        """
        k = int(len(iact))
        n = int(weights_out_in.shape[0])
        if weights_out_in.shape != (n, k):
            raise ValueError(f"Expected weights [N,K], got {weights_out_in.shape}")
        if k > self.k_pad or n > self.n_pad:
            raise ValueError("First baseline supports K,N <= 32; larger tiles need separate mapping verification")

        act_pad = np.zeros(self.k_pad, dtype=np.int64)
        act_pad[:k] = iact
        w_pad = np.zeros((self.n_pad, self.k_pad), dtype=np.int64)
        w_pad[:n, :k] = weights_out_in

        layer = _layer_shim(self.k_pad, self.n_pad)
        layer_params = lp.LayerParameters([0], layer, self.params, 0, 1)
        if layer_params.needed_total_transmissions != 1:
            raise ValueError("Only a one-repetition GEMV mapping is supported by this baseline")
        # No unverified OpenEye weight/activation reuse at this stage.
        layer_params.skipIact = 0
        layer_params.skipWght = 0
        layer_params.skipPsum = 0
        layer_params.send_values_out = 1  # Raw PSUM output via DMA.

        dram = DRAM.DRAMContents([layer], [layer_params])
        dram.fmap[0] = act_pad.tolist()
        dram.weights[0] = w_pad.tolist()
        dram.bias[0] = [0] * self.n_pad
        content = [dram.fmap[0], dram.weights[0], dram.bias[0]]

        mapper = DenseMapper(self.params, layer_params, 0, content, 0, 0)
        mapper.make_stream()
        raw = mapper.get_stream()
        if any(getattr(layer_params, key) != 0 for key in ("skipIact", "skipWght", "skipPsum")):
            raise ValueError("Baseline mapper unexpectedly enabled a skip flag")

        sections: dict[str, list[int]] = {}
        for name in SECTION_NAMES:
            section = raw[strdic.stream_parallel_dict[name]]
            words = [int(v) for v in section]
            if any(v < 0 or v > WORD_MASK for v in words):
                raise ValueError(f"Section {name} contains a value outside unsigned 64 bits")
            sections[name] = words
        if not sections["status"] or not sections["iact"] or not sections["wght"]:
            raise ValueError("Required DMA section is empty")
        return sections


def _write_mem(path: Path, words: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{w:016x}\n" for w in words), encoding="ascii")


def _flatten(sections: dict[str, list[int]]) -> list[int]:
    return [word for name in SECTION_NAMES for word in sections[name]]


def _assemble(template, iact, wght, offsets):
    status_end = offsets["iact"]["start"]

    return (
        template[:status_end]
        + iact
        + wght
        + template[status_end:]
    )


def export_projection_images(
    x: np.ndarray, matrices: dict[str, np.ndarray], mapper: AttentionProjectionMapper, out: Path
) -> dict:
    """Build one shared ROM plus banked activation and projection-weight RAM images."""
    seq_len, d_model = x.shape
    sections_by_pass: dict[str, dict[str, list[int]]] = {}
    weight_for_kind: dict[str, list[int]] = {}
    iact_for_token: dict[int, list[int]] = {}

    for kind in ("q", "k", "v"):
        for token in range(seq_len):
            name = f"{kind}_t{token}"
            # DRAM Dense weights are indexed [out][in]; mathematical W is [in][out].
            sections = mapper.make_pass(x[token], matrices[kind].T)
            sections_by_pass[name] = sections
            if kind not in weight_for_kind:
                weight_for_kind[kind] = sections["wght"]
            elif weight_for_kind[kind] != sections["wght"]:
                raise ValueError(f"Weight DMA image changed between tokens for {kind}")
            if token not in iact_for_token:
                iact_for_token[token] = sections["iact"]
            elif iact_for_token[token] != sections["iact"]:
                raise ValueError(f"Activation DMA image changed between Q/K/V for token {token}")

    first = sections_by_pass["q_t0"]
    counts = {name: len(first[name]) for name in SECTION_NAMES}
    offsets = {}
    at = 0
    for name in SECTION_NAMES:
        offsets[name] = {"start": at, "words": counts[name]}
        at += counts[name]

    # Full common template requires equal static sections across ALL 12 passes.
    # Fail rather than silently supply a wrong configuration for a later pass.
    for pass_name, sections in sections_by_pass.items():
        for name in SECTION_NAMES:
            if len(sections[name]) != counts[name]:
                raise ValueError(f"DMA section length for {pass_name}/{name} differs from q_t0")
            if name not in ("iact", "wght") and sections[name] != first[name]:
                raise ValueError(
                    f"Static section {name} differs for {pass_name}; "
                    "one common-ROM image is not valid for these inputs/configurations"
                )

    template = first["status"] + first["psum"] + first["quantize"]

    iact_words = counts["iact"]
    wght_words = counts["wght"]
    iact_ram = [word for token in range(seq_len) for word in iact_for_token[token]]
    weight_ram = [word for kind in ("q", "k", "v") for word in weight_for_kind[kind]]

    operations = []
    for bank, kind in enumerate(("q", "k", "v")):
        for token in range(seq_len):
            name = f"{kind}_t{token}"
            act_base = token * iact_words
            w_base = bank * wght_words
            expected = _flatten(sections_by_pass[name])
            reconstructed = _assemble(
                template,
                iact_ram[act_base:act_base + iact_words],
                weight_ram[w_base:w_base + wght_words],
                offsets,
            )
            if expected != reconstructed:
                raise AssertionError(f"ROM/RAM reconstruction mismatch for {name}")
            operations.append({
                "name": name, "activation_base_word": act_base,
                "weight_base_word": w_base, "dma_words": len(expected),
            })
            _write_mem(out / "verification_streams" / f"{name}.mem", expected)

    _write_mem(out / "common_rom.mem", template)
    _write_mem(out / "activation_ram.mem", iact_ram)
    _write_mem(out / "weight_ram.mem", weight_ram)
    np.savez(out / "inputs_used.npz", X=x, WQ=matrices["q"], WK=matrices["k"], WV=matrices["v"])

    manifest = {
        "scope": "Q/K/V projection only; 12 sequential GEMV passes via existing gemm_mode=0",
        "skip_flags": {"skipIact": 0, "skipWght": 0, "skipPsum": 0},
        "precision": "signed INT8 operands / raw PSUM output",
        "shape": {"sequence_length": seq_len, "embedding_width": d_model,
                  "padded_k": mapper.k_pad, "padded_n": mapper.n_pad},
        "dma_word_bits": 64, "complete_stream_words": sum(counts.values()),
        "sections": offsets,
        "memories": {"common_rom_words": len(template), "activation_ram_words": len(iact_ram),
                     "weight_ram_words": len(weight_ram)},
        "operations": operations,
        "caveat": "Bit-exact Python stream reconstruction checked, NOT yet RTL/DUT-verified. "
                  "RAM contents are initialization images; writable hardware is a separate RTL step.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("test/cocotb_attention_qkt/data/attention_mapper_v1"))
    args = parser.parse_args()
    if args.seq_len < 1 or not (1 <= args.d_model <= 32):
        parser.error("This baseline requires seq_len >= 1 and 1 <= d_model <= 32")
    for key, value in BASELINE_ENV.items():
        os.environ.setdefault(key, value)
    x, matrices = input_tensors(args)
    params = oep.get_oep(1)
    mapper = AttentionProjectionMapper(params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = export_projection_images(x, matrices, mapper, args.out_dir)

    print("PASS: Reconstructed all", len(manifest["operations"]), "Q/K/V DMA streams bit-exactly.")
    print("Section word counts:", {k: v["words"] for k, v in manifest["sections"].items()})
    print("Memory images:", manifest["memories"])
    print("Output:", args.out_dir.resolve())
    print("NOTE: This checks the Python Mapper/ROM overlay, not execution on OpenEye RTL.")


if __name__ == "__main__":
    main()
