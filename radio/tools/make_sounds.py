#!/usr/bin/env python3
"""Build the WLR sound pack into radio/sdcard/SOUNDS/en/WLR/.

Speech comes from the Windows SAPI voices via System.Speech (no network, no
service account); alert tones are synthesised with numpy. Everything is
normalised and written as 32 kHz / 16-bit / mono PCM, which is the format
EdgeTX's WAV player is happiest with.

The state and fault basenames are cross-checked against
WIDGETS/WLRHUD/robotdef.lua -- if the schema grows a fault and robotdef.lua is
regenerated, this refuses to run until sounds.json has words for it. A fault
the radio cannot name out loud is a fault you have to walk back to the laptop
for, which is the whole thing this pack exists to avoid.

Usage:
    python make_sounds.py                 # build everything that is missing
    python make_sounds.py --force         # rebuild every file
    python make_sounds.py --check         # verify coverage + format, write nothing
    python make_sounds.py --list-voices   # show installed SAPI voices
    python make_sounds.py --voice "Microsoft David Desktop"
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "sounds.json"
ROBOTDEF = REPO / "radio" / "sdcard" / "WIDGETS" / "WLRHUD" / "robotdef.lua"
OUT = REPO / "radio" / "sdcard" / "SOUNDS" / "en" / "WLR"
FLAT = REPO / "radio" / "sdcard" / "SOUNDS" / "en"

RATE = 32000
PEAK = 0.708  # -3 dBFS, leaves headroom for the radio's own mixing


# -- wav io -------------------------------------------------------------------

def write_wav(path, samples):
    """samples: float32 in [-1, 1]."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        ch, width, rate, n = (w.getnchannels(), w.getsampwidth(),
                              w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("%s: expected 16-bit, got %d-bit" % (path.name, width * 8))
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, rate


def resample(x, src_rate, dst_rate):
    if src_rate == dst_rate:
        return x
    n_out = int(round(len(x) * dst_rate / float(src_rate)))
    return np.interp(
        np.linspace(0.0, len(x) - 1.0, n_out),
        np.arange(len(x), dtype=np.float64),
        x.astype(np.float64),
    ).astype(np.float32)


def trim_silence(x, threshold=0.006, pad_ms=25):
    """Strip SAPI's leading/trailing dead air so callouts start on the word."""
    loud = np.abs(x) > threshold
    if not loud.any():
        return x
    pad = int(RATE * pad_ms / 1000)
    lo = max(0, int(np.argmax(loud)) - pad)
    hi = min(len(x), len(x) - int(np.argmax(loud[::-1])) + pad)
    return x[lo:hi]


def normalise(x, peak=PEAK):
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (peak / m) if m > 1e-6 else x


def fade(x, ms=6):
    n = min(int(RATE * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = x.copy()
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


# -- speech -------------------------------------------------------------------

PS_LIST_VOICES = (
    "Add-Type -AssemblyName System.Speech; "
    "(New-Object System.Speech.Synthesis.SpeechSynthesizer)."
    "GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
)


def powershell():
    """Locate powershell.exe.

    Run from Git Bash, PATH often does not carry System32, so a bare
    "powershell" fails with WinError 2. Resolve it explicitly.
    """
    import os
    import shutil

    found = shutil.which("powershell") or shutil.which("pwsh")
    if found:
        return found
    root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if candidate.exists():
        return str(candidate)
    raise SystemExit(
        "powershell.exe not found -- speech synthesis needs Windows SAPI. "
        "Tones can still be built with --check skipped on speech.")


def run_ps(script):
    return subprocess.run([powershell(), "-NoProfile", "-NonInteractive",
                           "-Command", script],
                          capture_output=True, text=True)


def list_voices():
    out = run_ps(PS_LIST_VOICES)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def synth_speech(text, wav_path, voice=None, rate=0):
    """Render `text` to `wav_path` with SAPI at 32 kHz/16-bit/mono."""
    select = ""
    if voice:
        select = "$s.SelectVoice('%s'); " % voice.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        + select +
        "$s.Rate = %d; " % rate +
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
        "%d, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); " % RATE +
        "$s.SetOutputToWaveFile('%s', $f); " % str(wav_path).replace("'", "''") +
        "$s.Speak('%s'); " % text.replace("'", "''") +
        "$s.Dispose()"
    )
    res = run_ps(script)
    if res.returncode != 0 or not wav_path.exists():
        raise RuntimeError("SAPI failed for %r:\n%s" % (text, res.stderr.strip()))


def build_speech(name, text, voice, rate):
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.wav"
        synth_speech(text, raw, voice=voice, rate=rate)
        data, src_rate = read_wav(raw)
    data = resample(data, src_rate, RATE)
    data = fade(normalise(trim_silence(data)))
    write_wav(OUT / (name + ".wav"), data)
    return len(data) / float(RATE)


# -- tones --------------------------------------------------------------------

def build_tone(name, spec):
    parts = []
    for seg in spec["segs"]:
        f0, ms = seg[0], seg[1]
        f1 = seg[2] if len(seg) > 2 else f0
        n = int(RATE * ms / 1000)
        if n <= 0:
            continue
        if f0 == 0:
            parts.append(np.zeros(n, dtype=np.float32))
            continue
        t = np.arange(n, dtype=np.float64) / RATE
        freq = np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        # A touch of second harmonic keeps it from sounding like a phone beep
        # through the TX15's small speaker.
        tone = np.sin(phase) + 0.22 * np.sin(2 * phase)
        parts.append(fade(normalise(tone.astype(np.float32), 1.0), ms=4))
        del t
    if not parts:
        raise ValueError("%s: no segments" % name)
    data = normalise(np.concatenate(parts), PEAK * spec.get("gain", 1.0))
    write_wav(OUT / (name + ".wav"), data)
    return len(data) / float(RATE)


# -- coverage check against robotdef.lua --------------------------------------

def robotdef_wavs():
    if not ROBOTDEF.exists():
        raise SystemExit(
            "%s missing -- run gen_robotdef_lua.py first" % ROBOTDEF)
    return set(re.findall(r'wav="([^"]+)"', ROBOTDEF.read_text(encoding="utf-8")))


def check_coverage(speech):
    required = robotdef_wavs()
    missing = sorted(required - set(speech))
    if missing:
        raise SystemExit(
            "sounds.json has no words for these robotdef.lua callouts: %s\n"
            "Add them under \"speech\" and re-run." % missing)
    return required


def check_flat(flat):
    problems = []
    for dest, src in sorted(flat.items()):
        d, s = FLAT / (dest + ".wav"), OUT / (src + ".wav")
        if not s.exists():
            problems.append("flat source %s.wav missing" % src)
        elif not d.exists():
            problems.append("flat copy %s.wav missing" % dest)
        elif d.stat().st_size != s.stat().st_size:
            problems.append("flat copy %s.wav is out of date with %s.wav"
                            % (dest, src))
    return problems


def check_built(names):
    problems = []
    for name in sorted(names):
        p = OUT / (name + ".wav")
        if not p.exists():
            problems.append("%s.wav missing" % name)
            continue
        with wave.open(str(p), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
                problems.append(
                    "%s.wav is %dch/%dbit/%dHz, want 1ch/16bit/%dHz"
                    % (name, w.getnchannels(), w.getsampwidth() * 8,
                       w.getframerate(), RATE))
    return problems


# -- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild existing files")
    ap.add_argument("--check", action="store_true",
                    help="verify coverage and format only, write nothing")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--voice", default=None, help="SAPI voice name")
    ap.add_argument("--rate", type=int, default=-1,
                    help="SAPI speaking rate, -10..10 (default -1, slightly "
                         "slow -- these are read in the field)")
    args = ap.parse_args()

    if args.list_voices:
        for v in list_voices():
            print(v)
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    speech = {k: v for k, v in manifest["speech"].items() if not k.startswith("_")}
    tones = {k: v for k, v in manifest["tones"].items() if not k.startswith("_")}
    flat = {k: v for k, v in manifest.get("flat", {}).items()
            if not k.startswith("_")}

    check_coverage(speech)
    names = set(speech) | set(tones)

    unknown = sorted(set(flat.values()) - names)
    if unknown:
        raise SystemExit("sounds.json 'flat' references unknown sounds: %s" % unknown)
    overlong = sorted(k for k in flat if len(k) > 6)
    if overlong:
        raise SystemExit(
            "flat names must be 6 characters or fewer for the Special Function "
            "track picker: %s" % overlong)

    if args.check:
        problems = check_built(names)
        problems += check_flat(flat)
        if problems:
            print("sound pack incomplete:", file=sys.stderr)
            for p in problems:
                print("  - %s" % p, file=sys.stderr)
            return 1
        print("sound pack OK: %d files, all 1ch/16bit/%d Hz" % (len(names), RATE))
        return 0

    if args.voice and args.voice not in list_voices():
        raise SystemExit("voice %r not installed; --list-voices to see options"
                         % args.voice)

    OUT.mkdir(parents=True, exist_ok=True)
    FLAT.mkdir(parents=True, exist_ok=True)
    built = skipped = 0
    total_s = 0.0

    for name, spec in sorted(tones.items()):
        if not args.force and (OUT / (name + ".wav")).exists():
            skipped += 1
            continue
        total_s += build_tone(name, spec)
        built += 1

    for name, text in sorted(speech.items()):
        if not args.force and (OUT / (name + ".wav")).exists():
            skipped += 1
            continue
        total_s += build_speech(name, text, args.voice, args.rate)
        built += 1

    for dest, src in sorted(flat.items()):
        shutil.copyfile(OUT / (src + ".wav"), FLAT / (dest + ".wav"))

    problems = check_built(names) + check_flat(flat)
    if problems:
        print("built with problems:", file=sys.stderr)
        for p in problems:
            print("  - %s" % p, file=sys.stderr)
        return 1

    size = sum((OUT / (n + ".wav")).stat().st_size for n in names)
    size += sum((FLAT / (n + ".wav")).stat().st_size for n in flat)
    print("sound pack ready in %s" % OUT.relative_to(REPO))
    print("  %d built, %d already present, %d total" % (built, skipped, len(names)))
    print("  %d flat copies in %s for Special Functions"
          % (len(flat), FLAT.relative_to(REPO)))
    print("  %.1f s of new audio, %.0f kB on card" % (total_s, size / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
