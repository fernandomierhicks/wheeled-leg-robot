# ODrive Firmware Update Procedure

Tested on: ODrive hw v3.6 variant 56, 2026-04-03

---

## Step 1 — Enter DFU Mode

With the ODrive connected via USB and detected by `odrive.find_any()`, run:

```bash
python -c "import odrive; odrv0 = odrive.find_any(); odrv0.enter_dfu_mode()"
```

The board reboots immediately into DFU mode. In Device Manager it now appears as **"STM32 BOOTLOADER"**.

---

## Step 2 — Flash Firmware with STM32CubeProgrammer

1. Download and install [STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html) from ST.
2. Open STM32CubeProgrammer.
3. Set interface to **USB**, click **Connect** — it detects the STM32 Bootloader.
4. Click **Open file**, select your `.elf` firmware file.
5. Click **Download**.
6. Click **Disconnect** when done — the board reboots into the new firmware.

CLI equivalent:
```bash
STM32_Programmer_CLI -c port=USB1 -w firmware.elf -v -rst
```

---

## Step 3 — Reinstall WinUSB Driver with Zadig

After flashing, the board re-enumerates but Windows shows it in Device Manager as **"ODrive 3.6 Native Interface"** with a yellow warning icon — the WinUSB driver is gone and needs to be reinstalled.

1. Download and run [Zadig](https://zadig.akeo.ie/).
2. Go to **Options → List All Devices**.
3. Select **"ODrive 3.6 Native Interface"** from the dropdown.
4. Set the target driver to **WinUSB**.
5. Click **Replace Driver** (or **Install Driver**).

> If two interfaces appear (Interface 0 and Interface 1), install WinUSB on **Interface 0** only.

---

## Step 4 — Verify

```bash
python -c "import odrive; odrv0 = odrive.find_any(); print(odrv0.hw_version_major, odrv0.hw_version_minor)"
```

Should print `3 6`. Then re-run the probe to document the new firmware state:

```bash
python probe_firmware.py
```
