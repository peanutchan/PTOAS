#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Golden reference for Phase 2 path bf16 -> f4E1M2x2 (packed 4-per-32bit).
#
# NOTE: The exact f4E1M2x2 byte encoding is hardware-defined and not standard
# IEEE.  We use a value set that:
#   (1) Only contains numbers exactly representable in f4E1M2x2, avoiding any
#       dependency on the rounding mode ({0, +/-1, +/-2, +/-4, ...}).  If the
#       encoding table is unknown, run the kernel on hardware once with these
#       inputs to capture the golden bytes, then paste them here.
#   (2) Two consecutive bf16 lanes pack into one output byte (low nibble = lane
#       2k, high nibble = lane 2k+1).  The golden must therefore be recorded
#       lane-pair-by-lane-pair.
#
# The placeholder byte 0xFF below marks slots the tester must fill in from a
# reference hardware run before this case is expected to pass compare.py.

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256   # bf16 logical lanes
BYTES = 128   # packed FP4 bytes (2 lanes / byte)

# Two bf16 lanes per byte.  Repeat 4 exact values across all lanes so the
# packed bytes are periodic.
VALUES = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)

# TODO: replace with actual f4E1M2x2 byte encoding captured from hardware.
# Layout convention assumed (verify against ISA):
#   byte[k] low  nibble = f4(VALUES[k*2 % 4])
#   byte[k] high nibble = f4(VALUES[(k*2+1) % 4])
F4E1M2X2_GOLDEN_BYTES_PERIOD = np.array([0xFF, 0xFF], dtype=np.uint8)  # placeholder


def bf16_bits(x: np.ndarray) -> np.ndarray:
    view = x.astype(np.float32).view(np.uint32)
    return (view >> 16).astype(np.uint16)


def generate(output_dir: Path) -> None:
    repeats = (ELEMS + len(VALUES) - 1) // len(VALUES)
    src_f32 = np.tile(VALUES, repeats)[:ELEMS].astype(np.float32)
    src = bf16_bits(src_f32)

    packed_period = np.tile(F4E1M2X2_GOLDEN_BYTES_PERIOD,
                            BYTES // len(F4E1M2X2_GOLDEN_BYTES_PERIOD))
    golden = packed_period[:BYTES].astype(np.uint8)
    dst = np.full(BYTES, 0xA5, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
