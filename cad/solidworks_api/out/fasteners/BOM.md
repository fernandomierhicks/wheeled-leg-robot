# Fastener BOM -- counted in ROBOT.SLDASM (25_fasteners.py bom)

| fastener | length | count | where (top-level assembly: count) |
|---|---|---:|---|
| M2 button head | 4 mm | 4 | Box-1: 4 |
| M2 button head | 6 mm | 4 | LeftTibia-1: 2, Tibia-1: 2 |
| M2 button head | 8 mm | 4 | BODY-1: 2, MirrorBODY-2: 2 |
| M2 standoff | 15 mm | 4 | Box-1: 4 |
| M2.5 button head (ISO 7380) | 12 mm | 10 | BODY-1: 5, MirrorBODY-2: 5 |
| M2.5 button head (ISO 7380) | 16 mm | 6 | Femur-1: 3, LeftFemur-1: 3 |
| M3 button head (ISO 7380) | 6 mm | 21 | Box-1: 9, COUPLER-1: 6, MirrorCOUPLER-2: 6 |
| M3 button head (ISO 7380) | 8 mm | 12 | FEMUR_INSIDE-1: 6, MirrorFEMUR_INSIDE-2: 6 |
| M3 button head (ISO 7380) | 10 mm | 10 | LeftTibia-1: 5, Tibia-1: 5 |
| M3 button head (ISO 7380) | 14 mm | 16 | Box-1: 4, LeftTibia-1: 6, Tibia-1: 6 |
| M3 button head (ISO 7380) | 16 mm | 8 | BODY-1: 4, MirrorBODY-2: 4 |
| M3 button head (ISO 7380) | 18 mm | 4 | FEMUR_INSIDE-1: 2, MirrorFEMUR_INSIDE-2: 2 |
| M3 button head (ISO 7380) | 20 mm | 25 | BODY-1: 4, Box-1: 13, Femur-1: 2, LeftFemur-1: 2, MirrorBODY-2: 4 |
| M3 button head (ISO 7380) | 25 mm | 9 | BODY-1: 4, Box-1: 1, MirrorBODY-2: 4 |
| M3 button head (ISO 7380) | 30 mm | 9 | BODY-1: 3, Box-1: 3, MirrorBODY-2: 3 |
| M3 button head (ISO 7380) | 35 mm | 18 | BODY-1: 5, COUPLER-1: 4, MirrorBODY-2: 5, MirrorCOUPLER-2: 4 |
| M3 button head (ISO 7380) | 50 mm | 8 | FEMUR_INSIDE-1: 2, Femur-1: 2, LeftFemur-1: 2, MirrorFEMUR_INSIDE-2: 2 |
| M3 countersunk (ISO 10642) | 6 mm | 2 | Box-1: 2 |
| M3 countersunk (ISO 10642) | 10 mm | 16 | Box-1: 16 |
| M3 countersunk (ISO 10642) | 14 mm | 4 | Box-1: 4 |
| M3 countersunk (ISO 10642) -- file says x8, the model is 12 long | 12mm | 15 | Box-1: 15 |
| M3 hex nut |  | 48 | Box-1: 48 |
| M3 standoff, male-female (6 mm male) | 5 mm | 2 | Box-1: 2 |
| M3 standoff, male-female (6 mm male) | 8 mm | 2 | Box-1: 2 |
| M3 standoff, male-female (6 mm male) | 12 mm | 7 | Box-1: 7 |
| M3 washer 7 x 0.5 |  | 12 | FEMUR_INSIDE-1: 6, MirrorFEMUR_INSIDE-2: 6 |
| M4 button head (ISO 7380) | 25 mm | 8 | LeftTibia-1: 4, Tibia-1: 4 |
| **total** | | **288** | |

Not in the model (open):
* wheel hub -> motor can (4 per wheel, r17 on the hub face): the hub bosses pass through 7 mm holes in the can; nothing behind them is modelled to thread into (his call 2026-10-08: leave out, flag)
* EncoderCarrier r1.6 hole over the tibia (line 165): the carrier is SOLID on the head side (2.6 mm wall) -- no screw can go in; a 6th carrier->tibia screw would need that wall opened

In the model, to check:
* 4 short screws at the femur <-> Femur_inside joint, per leg (2 x M3x20 in Femur, 2 x M3x18 in FEMUR_INSIDE): each enters the far end of a self-tap bore the M3x50 from the other side is already in (lines 226-229): it clamps nothing.  Remove them and the BOM drops 4 x M3x20 + 4 x M3x18
