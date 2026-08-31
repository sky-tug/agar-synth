#!/usr/bin/env python3
"""
Tiling for native-resolution inpainting.

WHY THIS FILE EXISTS (decision 3.51)
------------------------------------
A diffusion inpainting model works at a fixed canvas size (512 px for SD 1.5).
The obvious approach -- downscale the 2048 px plate to 512, inpaint, upscale --
destroys the thing this project is about:

    C.albicans median colony    27.5 px at 2048
    the same colony at 512       6.8 px

This is the imgsz=1280 argument (decision 2.8) in a harsher form. There the
detector merely struggled at 8.6 px; here the *generator* would be asked to
synthesise a 7 px object and the upscaler would then invent its texture. The
colony in the final image would not be a colony, and the small-class arms of the
substitution curve -- the paper's central claim -- would be measuring an
artefact.

So generation happens at NATIVE resolution, in tiles.

WHAT THIS COSTS
---------------
Measured on generated layouts (2048 px plate, plate radius 0.465*W):

    tile 384 -> 22.5 tiles/plate      tile 640 -> 12.1 tiles/plate
    tile 512 -> 16.3 tiles/plate      tile 768 ->  9.7 tiles/plate

Colonies are spread across the plate rather than clustered (decision 3.18:
Clark-Evans ~1.0), so mask-driven placement saves little over a full grid
(16.3 vs 20 at 512). There is no cheap way out of tiling.

Consequence: the production cost model is NOT "seconds per image". It is

    tiles_per_image  x  seconds_per_tile

and `scripts/budget.py` was assuming 8 s/image, i.e. roughly one tile. At 58,200
synthetic images this is the difference between ~129 and ~400 GPU-hours. The
number of denoising steps is therefore a BUDGET decision, not a quality detail.

THE INVARIANT THIS FILE GUARANTEES (decision 3.52)
--------------------------------------------------
**No colony is ever split across two tiles.**

If a colony straddles a tile boundary, its two halves are denoised in separate
passes from different noise. They will not agree: a seam runs through the middle
of the colony, and the detector learns a colony-shaped object with a line in it.
`place_tiles()` refuses to emit a layout that splits a colony, and
`test_generate.py` checks the invariant on generated plates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# SD 1.5 was trained at 512x512. Other sizes work but drift off-distribution;
# the value is a parameter so the tile-count/quality trade-off can be measured
# rather than assumed.
DEFAULT_TILE = 512

# Extra margin around a colony disk, in pixels. The model needs some context
# around the region it is filling, and a colony's shadow falls just outside its
# box. This padding affects only the TILE, never the mask, so decision 3.24
# ("the synthetic mask is exactly the labelled disk") is untouched.
CONTEXT_PAD = 24


@dataclass
class Tile:
    """An axis-aligned window into the full-resolution plate, in pixels."""
    x0: int
    y0: int
    size: int
    colony_idx: list[int]          # indices into the plan's colony list

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size

    def contains_box(self, bx0, by0, bx1, by1) -> bool:
        return (bx0 >= self.x0 and by0 >= self.y0
                and bx1 <= self.x1 and by1 <= self.y1)


def colony_boxes(colonies, W: int, H: int, pad: int = CONTEXT_PAD):
    """
    Pixel-space (x0, y0, x1, y1) of each colony disk, plus context padding,
    CLAMPED TO THE IMAGE.

    Decision 3.82. The clamp is not cosmetic. For a colony near the plate rim the
    context padding pushes the window past the image border:

        centre x = 72 px, a 367 px P.aeruginosa (r = 183), pad = 24
        -> x0 = 72 - 183 - 24 = -135

    Every tile place_tiles() can open is itself clamped to the image, so no tile
    can ever contain a box that starts at -135. The colony becomes uncoverable
    and place_tiles() aborts the whole run rather than drop it (decision 3.22).

    Pixels outside the image do not exist and do not need to be covered, so the
    box is intersected with the image here. The colony ITSELF is still covered
    whole -- only the padding is trimmed, and only on the side that ran off.

    The 10-plate demo package had no colony close enough to the rim for this to
    fire. It appeared on the first run over the full 2987-plate train split.
    """
    out = []
    for k in colonies:
        r = k["diameter"] / 2 * W + pad
        x0, y0 = k["xc"] * W - r, k["yc"] * H - r
        x1, y1 = k["xc"] * W + r, k["yc"] * H + r
        out.append((max(0.0, x0), max(0.0, y0),
                    min(float(W), x1), min(float(H), y1)))
    return out


def place_tiles(colonies, W: int, H: int, tile: int = DEFAULT_TILE,
                pad: int = CONTEXT_PAD) -> list[Tile]:
    """
    Cover every colony with as few `tile`-sized windows as possible, such that
    each colony lies ENTIRELY inside exactly one tile (decision 3.52).

    Greedy: take the topmost unplaced colony, open a window whose top-left
    corner is at that colony's top-left, absorb every other unplaced colony that
    fits wholly inside, then recentre the window on what it absorbed. Recentring
    is what lets the next window start clean instead of clipping a neighbour.

    A colony larger than the tile cannot be covered; that is an error, not
    something to silently drop -- see decision 3.22 for why this file never
    drops anything quietly.
    """
    boxes = colony_boxes(colonies, W, H, pad)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if x1 - x0 > tile or y1 - y0 > tile:
            raise ValueError(
                f"colony {i} needs {x1-x0:.0f}x{y1-y0:.0f} px but the tile is "
                f"{tile}. Raise --tile or lower CONTEXT_PAD. Silently dropping "
                f"it would leave an unlabelled object in the image.")

    remaining = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    tiles: list[Tile] = []

    while remaining:
        seed = remaining[0]
        sx0, sy0 = boxes[seed][0], boxes[seed][1]
        inside = [i for i in remaining
                  if boxes[i][0] >= sx0 and boxes[i][1] >= sy0
                  and boxes[i][2] <= sx0 + tile and boxes[i][3] <= sy0 + tile]
        if seed not in inside:
            inside = [seed]

        mx0 = min(boxes[i][0] for i in inside)
        my0 = min(boxes[i][1] for i in inside)
        mx1 = max(boxes[i][2] for i in inside)
        my1 = max(boxes[i][3] for i in inside)
        tx0 = int(round((mx0 + mx1) / 2 - tile / 2))
        ty0 = int(round((my0 + my1) / 2 - tile / 2))
        # Keep the window inside the image. This clamp can PUSH A COLONY OUT of
        # the window it was absorbed into -- near the image border the recentred
        # window moves, and a colony at the far edge of the group falls outside.
        # (The first version of this file assumed "the clamp can only help".
        # The invariant check below caught it on the first test run.)
        tx0 = max(0, min(tx0, W - tile))
        ty0 = max(0, min(ty0, H - tile))

        t = Tile(tx0, ty0, tile, [])
        # Re-select AFTER clamping: keep only what still fits wholly inside.
        # The seed always survives, because its own box is <= tile (checked
        # above) and the clamped window still covers the image region it is in.
        kept = [i for i in inside if t.contains_box(*boxes[i])]
        if seed not in kept:
            # Fall back to a window centred on the seed alone.
            b = boxes[seed]
            tx0 = max(0, min(int(round((b[0] + b[2]) / 2 - tile / 2)), W - tile))
            ty0 = max(0, min(int(round((b[1] + b[3]) / 2 - tile / 2)), H - tile))
            t = Tile(tx0, ty0, tile, [])
            kept = [i for i in inside if t.contains_box(*boxes[i])]
        inside = kept
        t = Tile(tx0, ty0, tile, sorted(inside))
        for i in inside:
            if not t.contains_box(*boxes[i]):
                raise AssertionError(
                    f"tile placement split colony {i}; this must never happen "
                    f"(decision 3.52)")
        if not inside:
            raise AssertionError(
                f"colony {seed} could not be covered by any {tile}px tile; "
                f"dropping it silently would leave an unlabelled object "
                f"(decision 3.22)")
        tiles.append(t)
        remaining = [i for i in remaining if i not in set(inside)]

    return tiles


def tile_stats(plans, W: int, H: int, tile: int = DEFAULT_TILE) -> dict:
    """
    Tiles-per-plate over a set of layout plans. Feeds the production cost model:
    generation cost is tiles x seconds-per-tile, not seconds-per-image.
    """
    counts = [len(place_tiles(p["colonies"], W, H, tile)) for p in plans]
    a = np.asarray(counts, dtype=float)
    return {
        "tile": tile,
        "n_plates": len(counts),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": int(a.max()),
        "total": int(a.sum()),
    }


def blend_weights(size: int, feather: int) -> np.ndarray:
    """
    Cosine ramp on all four edges, for compositing a generated tile back onto
    the plate. Tiles do not overlap on colonies (decision 3.52) but they may
    overlap on background; a hard paste would leave a visible square edge there.

    feather = 0 gives a hard paste, which is what the tests use when they need
    exact pixel arithmetic.
    """
    if feather <= 0:
        return np.ones((size, size), dtype=np.float32)
    r = np.ones(size, dtype=np.float32)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, math.pi, feather, dtype=np.float32)))
    r[:feather] = ramp
    r[-feather:] = ramp[::-1]
    return np.outer(r, r)


def composite(base: np.ndarray, generated: np.ndarray, mask: np.ndarray,
              t: Tile, feather: int = 32) -> np.ndarray:
    """
    Paste one generated tile back onto the plate, but ONLY where the mask says
    diffusion was allowed to paint.

    `base` is modified in place and returned. The mask gate is the point: even
    if the model changed pixels outside its mask (they do), those changes never
    reach the plate. Everything outside the mask is guaranteed byte-identical to
    the real background -- which is what lets decision 3.25's erase regions and
    the untouched plate rim be reasoned about at all.
    """
    m = (mask[t.y0:t.y1, t.x0:t.x1] > 0).astype(np.float32)
    if m.sum() == 0:
        return base
    w = m * blend_weights(t.size, feather)
    if base.ndim == 3:
        w = w[:, :, None]
    region = base[t.y0:t.y1, t.x0:t.x1].astype(np.float32)
    out = region * (1 - w) + generated.astype(np.float32) * w
    base[t.y0:t.y1, t.x0:t.x1] = np.clip(out, 0, 255).astype(base.dtype)
    return base
