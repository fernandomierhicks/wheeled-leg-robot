"""joy_debug.py — Minimal joystick debug. Prints axes + buttons to terminal."""
import pygame, time, sys

pygame.display.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME)
pygame.joystick.init()

n = pygame.joystick.get_count()
if n == 0:
    print("No joystick found. Plug in joystick and retry.")
    sys.exit(1)

joy = pygame.joystick.Joystick(0)
joy.init()
print(f"Joystick: {joy.get_name()}")
print(f"  Axes: {joy.get_numaxes()}  Buttons: {joy.get_numbuttons()}  Hats: {joy.get_numhats()}")
print("Move axes / press buttons. Ctrl-C to quit.\n")

axes    = [0.0] * joy.get_numaxes()
buttons = [0]   * joy.get_numbuttons()

try:
    while True:
        changed = False
        for ev in pygame.event.get():
            if ev.type == pygame.JOYAXISMOTION:
                if abs(ev.value - axes[ev.axis]) > 0.01:
                    axes[ev.axis] = ev.value
                    changed = True
            elif ev.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
                buttons[ev.button] = 1 if ev.type == pygame.JOYBUTTONDOWN else 0
                changed = True

        if changed:
            axis_str   = "  ".join(f"A{i}:{v:+.2f}" for i, v in enumerate(axes))
            btn_str    = "  ".join(f"B{i}:{v}" for i, v in enumerate(buttons) if v)
            print(f"\rAxes: {axis_str}   Btns: [{btn_str}]        ", end="", flush=True)

        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nDone.")
