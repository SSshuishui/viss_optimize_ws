#!/usr/bin/env python3
"""Clone the frozen production kernel and bypass its tile classification."""
from pathlib import Path
import difflib

root = Path(__file__).resolve().parent
src = (root / "frozen" / "viss_recon_kernel_mirror_tile_vissfast.cuh").read_text()
begin = "template<int TILE_PIX, bool DO_BLOCKAGE>\n__global__ void viss_partial_all_halfsym_tilecone("
assert src.count(begin) == 1
start = src.index(begin)
end = src.index("\n}\n", start) + 2
old = src[start:end].strip() + "\n"
branch = "    if(active && DO_BLOCKAGE){\n      float cx = tile_cx[tid]"
assert old.count(branch) == 1
new = old.replace("void viss_partial_all_halfsym_tilecone(",
                  "void viss_partial_all_halfsym_point_only(")
new = new.replace(branch, branch.replace("DO_BLOCKAGE", "DO_BLOCKAGE && false"))
assert new.count("DO_BLOCKAGE && false") == 1
out = root / "build"
out.mkdir(exist_ok=True)
(out / "point_only.cuh").write_text(new)
(out / "kernel_change.diff").write_text("".join(difflib.unified_diff(
    old.splitlines(keepends=True), new.splitlines(keepends=True),
    fromfile="frozen production forward kernel", tofile="point-only clone")))
