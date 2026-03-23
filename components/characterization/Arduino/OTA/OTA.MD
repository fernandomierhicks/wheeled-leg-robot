# OTA Firmware Upload — Arduino UNO R4 WiFi

## Overview

The UNO R4 WiFi supports wireless firmware upload using the [jandrassy/ArduinoOTA](https://registry.platformio.org/libraries/jandrassy/ArduinoOTA) library. The board connects to WiFi and listens on TCP port 65280 for firmware binaries sent via HTTP POST.

## How It Works

1. Firmware includes `ArduinoOTA` + `WiFiS3` libraries
2. On boot, the board connects to WiFi and starts the OTA server on port 65280
3. `ArduinoOTA.poll()` must be called in `loop()` to accept incoming uploads
4. Upload tool sends a `POST /sketch` with Basic auth and the binary payload
5. Board writes firmware to internal storage, then reboots with the new sketch

**Important:** Every sketch uploaded via OTA must also include the OTA code, otherwise you lose wireless upload capability and must recover via USB.

## PlatformIO Setup

Two environments are configured in `components/characterization/Arduino/platformio.ini`:

| Environment | Protocol | Use |
|---|---|---|
| `uno_r4_wifi` | USB (sam-ba) | First flash or recovery |
| `ota` | WiFi (custom curl) | Wireless upload |

### Upload Commands

```bash
# USB upload (board connected via cable)
pio run -e uno_r4_wifi -t upload

# OTA upload (board on WiFi)
pio run -e ota -t upload
```

The OTA environment uses a custom upload command:
```
curl -u arduino:arduino --data-binary @$SOURCE http://$UPLOAD_PORT:65280/sketch
```

### Credentials

| Setting | Value |
|---|---|
| WiFi SSID | `Minnexico` |
| WiFi Password | `Fenway724` |
| OTA Username | `arduino` (hardcoded by library) |
| OTA Password | `arduino` (3rd param in `ArduinoOTA.begin()`) |
| Board IP | `192.168.1.102` (set in `upload_port` in platformio.ini) |

## Minimum Firmware for OTA

Any sketch must include at minimum:

```cpp
#include <WiFiS3.h>
#include <ArduinoOTA.h>

void setup() {
    WiFi.begin("Minnexico", "Fenway724");
    while (WiFi.status() != WL_CONNECTED) delay(1000);
    ArduinoOTA.begin(WiFi.localIP(), "Arduino_R4", "arduino", InternalStorage);
}

void loop() {
    ArduinoOTA.poll();
    // ... your code here ...
}
```

The `lib_deps` entry needed in `platformio.ini`:
```ini
lib_deps = jandrassy/ArduinoOTA@^1.1.1
```

## Changing the Board IP

If the board gets a different IP from DHCP, update `upload_port` in the `[env:ota]` section of `platformio.ini`. The board scrolls its current IP on the LED matrix at boot.

## Flash Size Constraint

With `InternalStorage`, the sketch binary is limited to **half the flash** (128 KB of 256 KB) because the library stores the new firmware in the upper half before copying it over on reboot.
