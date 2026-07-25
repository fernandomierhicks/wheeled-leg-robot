# Buzzer

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

Non-blocking passive buzzer driver. Tone, MIDI note, and melody playback via PWM.

## Wiring

PWM pin → 1 kΩ → NPN base; collector → buzzer → V+; emitter → GND.

Teensy 4.1 pin: **5** (`PIN_BUZZER` in `config.h`).

## API

```cpp
Buzzer buz(PIN_BUZZER);
buz.begin();         // call once in setup()
// control loop:
buz.update();        // must be called every loop iteration
// play sounds:
buz.tone(440);                          // 440 Hz, hold until off()
buz.midi(69, 200, 500);                 // A4, volume 200, 500 ms
buz.play(notes_array, count, 255, false); // melody, no loop
buz.off();
```

`BuzzerNote` struct: `{midi_note, on_ms, gap_ms}`. `midi = 0` is a rest.

## Gotchas

**`update()` is required** — all timing is millis()-based. Must be called every loop iteration for notes and gaps to advance correctly.

**Volume via duty cycle** — volume (0–255) sets PWM duty, not amplitude; the perceived loudness curve is non-linear. Full volume (`255`) is a 50% duty square wave, which is the loudest for a passive buzzer.

**`play()` does not copy the array** — the `notes` pointer must remain valid for the duration of playback. Store melodies in `const` arrays in flash (`PROGMEM` or plain `const` global).
