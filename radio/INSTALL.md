# Installing onto the TX15

## 1. Validate first

```
python radio/tools/gen_robotdef_lua.py --check
python radio/tools/make_sounds.py --check
python radio/tools/check_lua.py
python radio/tools/check_sdcard.py
```

All four must pass. They check the staged files against each other and against
the firmware's own schema, which is cheaper than finding out on the radio.

## 2. Put the radio in USB storage mode

Plug the TX15 in over USB. It asks what to do — choose **USB Storage**. If it
doesn't ask: *Radio settings → SD-HC card / USB → USB mode → Storage*.

## 3. Dry run

```
python radio/tools/install_to_radio.py
```

It finds the card, prints the board and firmware version off `RADIO/radio.yml`,
and shows exactly what it would write. Nothing is written.

Confirm it says `Board: tx15`. If it says anything else, stop — the model file
assumes this hardware.

## 4. Apply

```
python radio/tools/install_to_radio.py --apply
```

What it does and does not touch:

- **Never writes `RADIO/radio.yml`.** That file holds your hall-gimbal
  calibration. Re-doing that by hand is an afternoon you won't get back.
- **Never writes `MODELS/labels.yml`.** EdgeTX rebuilds it and picks up a new
  model file on its own.
- Picks a free `model<NN>.yml` slot instead of clobbering an existing model.
  Your `model1.yml` is untouched.
- Backs up anything it would overwrite into a timestamped folder under
  `/BACKUP/` on the card first.

Install just part of it with `--only THEMES WIDGETS SOUNDS`.

## 5. On the radio

1. Eject the USB volume, then exit USB mode on the radio.
2. **Radio settings → Themes → RoboBlue → Set as default.**
   The theme lives in `radio.yml`, which the installer deliberately does not
   write, so this step is yours.
3. **Model select → WLR ROBOT.** Boot goes straight to the HUD.

   The bring-up checklist is **not** shown as a boot modal (`displayChecklist:
   0`). It is still on the card and still reachable on demand from the model
   menu. It ships under three filenames — `WLR ROBOT.txt`, `WLR_ROBOT.txt` and
   `model90.txt` — because EdgeTX resolves it from the model name, the model
   name with underscores, or the model *file* name, and which rule the lookup
   uses could not be established for 3.0. Set `displayChecklist` back to 1 in
   `build_model.py` for a bring-up or jump-ladder session.
4. **Radio settings → Hardware**: set SE to momentary and SF to 2-position,
   after you've physically swapped the shoulder switches. Then power-cycle and
   confirm the switch warning fires when a switch is left down.
5. **Screen 2** is the Outputs monitor. That is the screen the CRSF bring-up
   gate needs.

## 6. Expected first-boot behaviour

- The HUD shows **NO ROBOT TELEMETRY**. Correct — the robot is still on iBUS
  and the CRSF telemetry emitter doesn't exist yet.
- Arm/disarm callouts work immediately: they're driven by logical switches on
  CH10, which needs no telemetry at all.
- The switch warning and throttle warning work immediately.
- Moving a stick or switch moves the right bar on screen 2 immediately.

That last one is worth doing before you go any further — it validates the whole
channel map with nothing powered on the robot.

## Updating later

Re-run the same command. Files that are already identical are skipped, so a
re-install after changing one sound copies one file.

If you edit the model **on the radio**, the radio rewrites its `model<NN>.yml`
and your edits will be overwritten by the next `--apply`. Copy the radio's
version back into `radio/sdcard/MODELS/` first if you want to keep them.

## If something goes wrong

- **Model doesn't appear in the list** — the filename must be
  `model<digits>.yml`. EdgeTX skips anything else without a word.
- **Widget missing from the widget picker** — `WIDGETS/WLRHUD/main.lua` must
  exist and return a `name`. Check for a Lua error on the radio.
- **HUD says it needs EdgeTX 2.11+** — the `lvgl` object is nil. That means the
  firmware predates the LVGL Lua API.
- **Theme looks wrong** — confirm all five required files copied
  (`theme.yml`, `logo.png`, `screenshot1..3.png`) plus
  `background_480x320.png`.
- **Anything overwritten by mistake** — it's in `/BACKUP/wlr-<timestamp>/` on
  the card, in its original layout.
