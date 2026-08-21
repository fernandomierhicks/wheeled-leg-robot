"""Fly the digital twin with the real TX15 transmitter.

The TX15 enumerates as a USB HID gamepad (the model ships
``usbJoystickIfMode: JOYSTICK``), so the same sticks, the same switch layout
and the same muscle memory that drive the robot can drive the simulation.

The point is not novelty. It is that a gain tuned in sim maps to the *same
parameter* on the robot -- ``twin/params_control.py`` and the firmware's
parameter table are both generated from ``protocol/schema.json`` -- so the
twin becomes a flight simulator for this specific machine rather than a
loosely related model of it.

To make that transfer real, this module decodes the gamepad the way the
FIRMWARE decodes the radio: HID -> channel microseconds -> the scaling in
``radio_update()``. Two consequences worth stating:

  * the thresholds are the firmware's own absolute microsecond literals
    (> 1990, < 1010), not re-derived ones
  * the command scaling reads ``radio_vel_max`` / ``radio_yaw_max`` /
    ``radio_roll_max`` from the shared schema, so sim and robot cannot drift

── EdgeTX's HID mapping ──────────────────────────────────────────────────────

From ``radio/src/usb_joystick.cpp``, simple mode (``usbJoystickExtMode: 0``):

    X  Y  Z  Rx Ry Rz S1 S2   ->  channels 1..8   (analog)
    buttons 1..8              ->  channels 9..16  ("on if channel > 0")

Which means a real limitation, stated up front rather than discovered later:
**channels 9-16 arrive as booleans.** CH9 (speed profile) is a three-position
switch, so over USB its middle position is indistinguishable from its up
position. CH13 (the encoded live-tune group) collapses entirely -- every
non-zero level reads as "on".

For flying the twin that is fine: everything that matters for stick work
(roll, velocity, hip height, yaw, jump, both tune knobs) is on an analog axis
and arrives at full resolution. If you later need faithful CH9/CH13, switch
the model to ``usbJoystickExtMode: 1`` and give them their own axes.

Requires ``pygame`` (SDL joystick backend). Imported lazily so the twin does
not gain a dependency for a tool most runs never use.
"""

from __future__ import annotations

from dataclasses import dataclass

# EdgeTX simple-mode HID layout. Index is the axis number SDL reports.
AXIS_TO_CHANNEL = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
BUTTON_TO_CHANNEL = {i: 9 + i for i in range(8)}

# Firmware thresholds, copied deliberately rather than re-derived. These are
# the same absolute microsecond literals main.cpp tests against; see
# crsf_protocol.h for why normalising happens at the boundary and these stay
# untouched.
CH_HIGH_US = 1990
CH_LOW_US = 1010
CH_MIN_US, CH_MID_US, CH_MAX_US = 1000, 1500, 2000


def axis_to_us(value: float) -> int:
    """SDL axis (-1.0 .. +1.0) -> channel microseconds (1000 .. 2000)."""
    v = max(-1.0, min(1.0, float(value)))
    return int(round(CH_MID_US + v * 500.0))


def button_to_us(pressed: bool) -> int:
    """A HID button is EdgeTX's 'channel > 0', so it can only be low or high."""
    return CH_MAX_US if pressed else CH_MIN_US


@dataclass
class RadioCommands:
    """One tick of transmitter input, decoded the way the firmware decodes it.

    Field names match the firmware parameters they drive, so wiring this into
    the twin is a lookup rather than a translation.
    """
    v_cmd_ms: float = 0.0
    omega_cmd_rds: float = 0.0
    radio_hip_cmd: float = 0.0
    roll_cmd_rad: float = 0.0
    armed: bool = False
    jump_edge: bool = False
    lean_enabled: bool = False
    tune_a: float = 0.0          # CH7 normalised 0..1
    tune_b: float = 0.0          # CH8 normalised 0..1
    link_alive: bool = False


class ChannelDecoder:
    """Channels (microseconds) -> firmware-equivalent commands.

    Pure: no pygame, no hardware. This is the half worth testing, because it
    is the half that has to agree with ``radio_update()`` in main.cpp.
    """

    def __init__(self, vel_max: float, yaw_max: float, roll_max: float):
        self.vel_max = float(vel_max)
        self.yaw_max = float(yaw_max)
        self.roll_max = float(roll_max)
        self._jump_prev = False

    @staticmethod
    def _norm_bipolar(us: int) -> float:
        """1000..2000 -> -1..+1, the firmware's (ch - 1500) / 500."""
        return max(-1.0, min(1.0, (us - CH_MID_US) / 500.0))

    @staticmethod
    def _norm_unipolar(us: int) -> float:
        """1000..2000 -> 0..1, the firmware's (ch - 1000) / 1000."""
        return max(0.0, min(1.0, (us - CH_MIN_US) / 1000.0))

    def decode(self, ch: dict[int, int], link_alive: bool = True) -> RadioCommands:
        """`ch` maps 1-indexed channel number to microseconds.

        A channel the transmitter did not send reads as 0, matching the
        driver's behaviour on a dead link -- see the FAILSAFE CONTRACT in
        crsf_protocol.h. Zero fails every `> 1990` test and passes every
        `< 1010` one, which is exactly why the firmware guards each combo with
        an explicit `alive &&` as well.
        """
        def get(n: int, neutral: int = CH_MIN_US) -> int:
            """Channel `n` in microseconds, or its safe neutral if absent.

            A channel the transmitter never sent is absent, not zero. Zero is
            what the firmware's channel() returns on a DEAD LINK, where it is
            correct precisely because every threshold is a "> 1990" test that
            zero fails. But it is NOT a valid input to the bipolar scaling:
            (0 - 1500) / 500 clamps to -1.0, which would command full reverse
            from a channel that simply is not there.
            
            So absent maps to the neutral for that channel's role -- centre
            for a stick, low for a switch -- and only a genuinely dead link
            zeroes everything.
            """
            if not link_alive:
                return 0
            v = ch.get(n, 0)
            return v if v >= CH_MIN_US else neutral

        cmd = RadioCommands(link_alive=link_alive)
        if not link_alive:
            self._jump_prev = False
            return cmd

        # radio_update(): CH1 roll, CH2 velocity, CH3 hip, CH4 yaw. The yaw
        # sign is inverted in firmware so stick-left yaws the robot left.
        cmd.roll_cmd_rad = self._norm_bipolar(get(1, CH_MID_US)) * self.roll_max
        cmd.v_cmd_ms = self._norm_bipolar(get(2, CH_MID_US)) * self.vel_max
        cmd.radio_hip_cmd = self._norm_unipolar(get(3))
        cmd.omega_cmd_rds = -self._norm_bipolar(get(4, CH_MID_US)) * self.yaw_max

        cmd.tune_a = self._norm_unipolar(get(7))
        cmd.tune_b = self._norm_unipolar(get(8))

        cmd.armed = get(10) > CH_HIGH_US
        cmd.lean_enabled = get(15) > CH_HIGH_US

        # CH6 is a rising edge in firmware: one jump per press, and holding
        # the button does not hop repeatedly.
        jump_now = get(6) > CH_HIGH_US
        cmd.jump_edge = jump_now and not self._jump_prev
        self._jump_prev = jump_now

        return cmd


class Tx15Joystick:
    """The TX15 as a USB HID gamepad, presented as radio channels.

    Exposes ``channel(n)`` in microseconds on purpose: it is the same
    interface the firmware's Crsf driver exposes, so the decode above can be
    read against main.cpp line for line.
    """

    # The TX15's USB descriptor is manufacturer "RM", product "TX15"
    # (radio/src/targets/tx15/usb_descriptor.h), so SDL reports a name
    # containing "TX15". Matching on it matters: a development machine often
    # has other HID devices enumerated, and silently binding to a SpaceMouse
    # driver would be a confusing way to discover that.
    DEFAULT_NAME_HINT = "tx15"

    def __init__(self, index: int = 0, name_hint: str = DEFAULT_NAME_HINT):
        import pygame  # lazy: the twin should not need pygame to run

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "No USB joystick found.\n"
                "On the radio: plug in USB and choose 'USB Joystick' (not USB\n"
                "Storage) when it asks. Model settings must have\n"
                "usbJoystickIfMode: JOYSTICK, which the WLR ROBOT model does.")

        names = []
        chosen = None
        for i in range(count):
            js = pygame.joystick.Joystick(i)
            names.append(js.get_name())
            if name_hint and name_hint.lower() in js.get_name().lower():
                chosen = i
                break

        if chosen is None:
            if name_hint:
                # Do not quietly grab device 0. On a machine with other HID
                # devices that is how you end up flying the twin with a
                # 3D mouse and wondering why the sticks do nothing.
                seen = "\n  ".join("[%d] %s" % (i, n)
                                   for i, n in enumerate(names))
                raise RuntimeError(
                    "No HID device matching %r found. Devices seen:\n  %s\n\n"
                    "If the radio is plugged in, it is probably in USB Storage\n"
                    "mode -- unplug, replug, and choose 'USB Joystick'.\n"
                    "To use one of the above anyway, pass --index N --any."
                    % (name_hint, seen))
            chosen = index

        self._js = pygame.joystick.Joystick(chosen)
        self._js.init()
        self.name = self._js.get_name()
        self.num_axes = self._js.get_numaxes()
        self.num_buttons = self._js.get_numbuttons()
        self._ch: dict[int, int] = {}
        self.poll()

    def poll(self) -> dict[int, int]:
        """Pump SDL events and refresh the channel map. Call every tick."""
        self._pygame.event.pump()
        ch = {}
        for axis, chan in AXIS_TO_CHANNEL.items():
            if axis < self.num_axes:
                ch[chan] = axis_to_us(self._js.get_axis(axis))
        for btn, chan in BUTTON_TO_CHANNEL.items():
            if btn < self.num_buttons:
                ch[chan] = button_to_us(bool(self._js.get_button(btn)))
        self._ch = ch
        return ch

    def channel(self, n: int) -> int:
        """Channel `n` (1-indexed) in microseconds, 0 if the radio never sent it."""
        return self._ch.get(n, 0)

    def channels(self) -> dict[int, int]:
        return dict(self._ch)

    def describe(self) -> str:
        return ("%s -- %d axes, %d buttons -> channels 1-%d analog, %d-%d binary"
                % (self.name, self.num_axes, self.num_buttons,
                   min(self.num_axes, 8), 9, 8 + min(self.num_buttons, 8)))

    def close(self):
        try:
            self._js.quit()
            self._pygame.joystick.quit()
            self._pygame.quit()
        except Exception:
            pass


def make_decoder_from_params(params) -> ChannelDecoder:
    """Build a decoder using the SHARED schema limits.

    `params` is twin/params_control's value mapping. Reading the limits from
    there rather than hardcoding them is what keeps a sim session and a robot
    session talking about the same numbers.
    """
    def val(name: str, fallback: float) -> float:
        try:
            v = params[name]
            return float(v) if v is not None else fallback
        except Exception:
            return fallback

    return ChannelDecoder(
        vel_max=val("radio_vel_max", 0.5),
        yaw_max=val("radio_yaw_max", 1.0),
        roll_max=val("radio_roll_max", 0.17),
    )
