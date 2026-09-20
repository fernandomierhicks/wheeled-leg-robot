# How to mark these sheets

## The legend

| colour | meaning |
|---|---|
| **RED** | material **MAY BE REMOVED** here |
| **GREEN** | material **MAY BE ADDED** here — write how far, e.g. `+6` |
| **unmarked** | leave it alone |

Two colours only. **Unmarked is the safe answer** — where you say nothing, the
pipeline's own derived keep-out still applies and nothing moves. So you only
need to mark where you actually want something to change; you do not have to
defend every bearing seat with a pen.

Free text is read by me, not by the script, so write whatever is useful:
*"3 mm to the coupler here"*, *"tibia sweeps through this in retract"*,
*"this face is the bearing seat, never touch"*.

## Two rules that matter

1. **Mark the PNG at 100%. Do not crop or resize it.** Every sheet has a
   `.json` and a `.npz` beside it holding the camera transform and a per-pixel
   part id + depth map. That is what turns a mark into an exact point in the
   part's own coordinate frame. A resized sheet silently relocates every mark,
   so `readmarks.py` refuses one whose width has changed.
2. **Keep the marked file next to the original.** Any name works —
   `asm_outboard_marked.png`, `asm_outboard (1).png`, `asm_outboard copy.png` —
   the sidecars are found by peeling the name back.

Bright red and bright green, reasonably thick. The sheets are deliberately drawn
in blues, violets, teals, ambers and greys, with no red or green anywhere, so
your pen is separated by hue alone.

## What is here

```
assembly/asm_outboard.png    the show face of every part, and how they crowd
assembly/asm_inboard.png     the BACK of every part - you have never seen this
assembly/asm_edge_±X.png     lateral stack-up; outboard is UP the page
assembly/asm_edge_±Y.png     lateral stack-up; outboard is UP the page
parts/<Part>_show.png        one part alone, unstyled, flat grey
parts/<Part>_back.png        its back face
parts/<Part>_end_X/Y.png     end-on, where a fin gives itself away
INDEX_assembly.png           all six assembly views on one sheet
INDEX_parts.png              all twenty part views on one sheet
```

Pose is **Middle**. Checkered areas mean *a motor, bearing, fastener or the
wheel sits in front of this surface* — the part underneath is still drawn in
full and is still markable; the checker is just telling you what is in the way.

## Then

Send the marked PNGs back. I run `tools/readmarks.py` on them, which writes an
**ECHO** sheet showing exactly what I understood — each zone boxed, named, and
reported in millimetres in that part's own frame. **Nothing is used until you
confirm the echo.** If I have read a mark onto the wrong rib, that is where you
catch it, not in plastic.

## Things I still need in words, not marks

1. Which faces are **bearing / mating / fastener seats** that must never move?
2. Which **pairs of parts run tight**, and what is the real clearance? The
   pipeline currently assumes 1.0 mm plus a 6 mm `PAD_Z` — a margin you chose
   without measuring and have flagged as yours to loosen. Measured from the
   assembly: `BearingWasher` sits **2.0 mm** above the Coupler, and
   `SmallBearingWahser` **2.0 mm** above both the Coupler and the Side panel.
3. What **minimum wall thickness** will you accept, and the minimum feature
   worth printing at 0.4 mm nozzle / 0.2 mm layer?
4. On the Coupler, Side panel and RobotMount the flange currently lands on the
   **inboard** face, because none of them has metal at its show-face perimeter.
   It is structural but never seen. Opening that up means loosening the
   keep-out — your call.
5. `A_Z = −23.5 mm` is inherited from baseline-1 and **has never been
   re-measured on the v4 box**. If anything here ends up depending on
   body-centre coordinates, that needs measuring first.
