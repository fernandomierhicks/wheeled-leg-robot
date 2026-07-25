# LED

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

Non-blocking RGB LED driver with animations (solid, blink, pulse, fade).

## Wiring

Common-cathode RGB LED (default). For common-anode pass `active_low = true` to the constructor.

| Channel | Teensy 4.1 pin |
|---|---|
| R | 3 |
| G | 2 |
| B | 4 |

Pins are defined in `config.h` as `PIN_LED_R/G/B`.

## API

```cpp
RgbLed led(PIN_LED_R, PIN_LED_G, PIN_LED_B);
led.begin();         // call once in setup()
// control loop:
led.update();        // must be called every loop iteration
// animations:
led.solid(255, 0, 0);
led.blink(0, 255, 0, 200, 200);        // on_ms, off_ms
led.pulse(0, 0, 255, 1500);            // period_ms
led.fade_to(255, 255, 0, 500);         // duration_ms
```

## Gotchas

**`update()` is required** — all animations are millis()-based and driven entirely by `update()`. If you skip a call the animation stalls; it won't catch up when calls resume.

**`fade_to` is one-shot** — it transitions from the current output colour to the target over `duration_ms`, then holds as `SOLID`. Check `is_done()` to know when the transition finishes.
