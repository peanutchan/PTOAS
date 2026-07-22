"""Pure PTODSL VMI control for loop-carried histogram accumulator vecscope.

This is intentionally independent of TileLang/DeepSeek.  It builds a two-chunk
VMI histogram accumulation pattern:

1. chunk 0 initializes a ui32 VMI accumulator from zero;
2. chunk 1 initializes the same accumulator from UB with vmi.vload;
3. both chunks update the accumulator through pto.for_(...).carry(...).

This is the pure PTODSL "ok" control for the TileLang inter-op issue: explicit
pto.vecscope regions are emitted around both chunks, so PTOAS accepts the same
cross-chunk accumulator pattern.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ptodsl import pto
from ptodsl._control_flow import vecscope as _vecscope


LANES = 256
CHUNK_TILES = 1
CHUNK_ELEMS = LANES * CHUNK_TILES
CHUNK_WORDS = CHUNK_ELEMS * 2
SRC_WORDS = CHUNK_WORDS * 2

SRC_UB = 0
OUT_UB = CHUNK_ELEMS * 8


@pto.jit(
    name="repro_vmi_loop_acc_vecscope_ptodsl",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ast_rewrite=False,
)
def repro_vmi_loop_acc_vecscope_ptodsl(
    src_gm: pto.ptr(pto.i32, "gm"),
    out_gm: pto.ptr(pto.i32, "gm"),
):
    src_ub = pto.castptr(pto.const(SRC_UB, dtype=pto.ui64), pto.ptr(pto.i32, "ub"))
    src32_ub = pto.castptr(pto.const(SRC_UB, dtype=pto.ui64), pto.ptr(pto.i32, "ub"))
    out_ub = pto.castptr(pto.const(OUT_UB, dtype=pto.ui64), pto.ptr(pto.ui32, "ub"))

    pto.mte_gm_ub(src_gm, src_ub, 0, CHUNK_WORDS * 4, nburst=(1, CHUNK_WORDS * 4, CHUNK_WORDS * 4))
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    with _vecscope():
        mask = pto.vmi.create_mask(LANES, size=LANES)
        hist16_zero = pto.vmi.vbrc(pto.ui16(0), size=LANES)
        acc0 = pto.vmi.vbrc(pto.ui32(0), size=LANES)

        loop0 = pto.for_(0, CHUNK_TILES, step=1).carry(acc=acc0)
        with loop0:
            values, _ = pto.vmi.vload(
                src32_ub,
                loop0.iv * LANES * 2,
                size=LANES,
                dist_mode="dintlv",
            )
            source = pto.vmi.vcvt(values, pto.i8, saturate=pto.VcvtSatMode.NOSAT)
            hist16 = pto.vmi.vdhist(hist16_zero, source, mask)
            hist32 = pto.vmi.vcvt(hist16, pto.ui32)
            loop0.update(acc=pto.vmi.vadd(loop0.acc, hist32, mask))

        pto.vmi.vstore(loop0.final("acc"), out_ub, 0, mask)

    pto.mte_gm_ub(
        pto.addptr(src_gm, CHUNK_WORDS),
        src_ub,
        0,
        CHUNK_WORDS * 4,
        nburst=(1, CHUNK_WORDS * 4, CHUNK_WORDS * 4),
    )
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=1)

    with _vecscope():
        mask = pto.vmi.create_mask(LANES, size=LANES)
        hist16_zero = pto.vmi.vbrc(pto.ui16(0), size=LANES)
        acc0 = pto.vmi.vload(out_ub, 0, size=LANES)

        loop1 = pto.for_(0, CHUNK_TILES, step=1).carry(acc=acc0)
        with loop1:
            values, _ = pto.vmi.vload(
                src32_ub,
                loop1.iv * LANES * 2,
                size=LANES,
                dist_mode="dintlv",
            )
            source = pto.vmi.vcvt(values, pto.i8, saturate=pto.VcvtSatMode.NOSAT)
            hist16 = pto.vmi.vdhist(hist16_zero, source, mask)
            hist32 = pto.vmi.vcvt(hist16, pto.ui32)
            loop1.update(acc=pto.vmi.vadd(loop1.acc, hist32, mask))

        pto.vmi.vstore(loop1.final("acc"), out_ub, 0, mask)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(out_ub, out_gm, 4, nburst=(1, 4, 4))


def main() -> None:
    ptoas_bin = os.environ.get("PTOAS_BIN", "ptoas")
    flags = [
        "--pto-arch=a5",
        "--pto-backend=vpto",
        "--pto-level=level3",
        "--enable-vmi",
        "--enable-tile-op-expand",
        "--cann-output-version=9.1.0-beta.3",
    ]

    tmpdir = Path(tempfile.mkdtemp(prefix="repro_vmi_loop_acc_vecscope_ptodsl_"))
    pto_path = tmpdir / "repro_vmi_loop_acc_vecscope_ptodsl.pto"
    obj_path = tmpdir / "repro_vmi_loop_acc_vecscope_ptodsl.fatobj.o"
    pto_path.write_text(repro_vmi_loop_acc_vecscope_ptodsl.compile().mlir_text(), encoding="utf-8")

    print(f"PTO: {pto_path}")
    result = subprocess.run(
        [ptoas_bin, *flags, str(pto_path), "-o", str(obj_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)
    if result.returncode == 0:
        print("PTOAS PASS: pure PTODSL does not reproduce the vecscope issue on this build.")
        return

    print("PTOAS FAIL: pure PTODSL reproduces the loop-carried VMI accumulator vecscope issue.")
    print("Relevant PTO excerpt:")
    pto_lines = pto_path.read_text(encoding="utf-8").splitlines()
    for line_no in range(1, min(90, len(pto_lines)) + 1):
        line = pto_lines[line_no - 1]
        if "scf.for" in line or "vmi.vload" in line or "vmi.vdhist" in line or "vmi.vadd" in line:
            start = max(1, line_no - 4)
            end = min(len(pto_lines), line_no + 6)
            for excerpt_line in range(start, end + 1):
                print(f"{excerpt_line:03d}: {pto_lines[excerpt_line - 1]}")
            break
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()