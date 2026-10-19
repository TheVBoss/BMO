#!/usr/bin/env python3
# Interactive wizard for wiring up (or re-wiring) the 7 buttons described in
# config.BUTTON_PINS. Prompts for one button at a time - UpArrow, DownArrow,
# LeftArrow, RightArrow, Triangle, LargeCircle, SmallCircle - listens for its
# "B<pin>:0" press event from the Arduino Micro running
# extra/ultrasonic_sensor/ultrasonic_sensor.ino, and once all 7 are detected,
# writes the discovered pin numbers straight into config.py's BUTTON_PINS.
#
# Run this after wiring up new buttons, or whenever you're not sure which
# physical button is on which pin - no need to know the pin numbers
# yourself, just press what's asked for.
import re
import sys
from pathlib import Path
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import serial

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.py"

# Order to prompt in. Written back into config.py using config.BUTTON_PINS's
# existing key order, not this prompt order, so the diff stays minimal.
PROMPT_ORDER = ["UpArrow", "DownArrow", "LeftArrow", "RightArrow", "Triangle", "LargeCircle", "SmallCircle"]


def _read_press(ser, taken_pins):
    """Block until a press event ('B<pin>:0') on a not-yet-assigned pin arrives; return its pin."""
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line.startswith("B") or ":" not in line:
            continue  # not a button event (e.g. a stray distance reading)

        try:
            pin_str, state_str = line[1:].split(":")
            pin = int(pin_str)
            state = int(state_str)
        except ValueError:
            continue

        if state != 0:
            continue  # release event, ignore - only presses count

        if pin in taken_pins:
            print(f"  (pin {pin} is already assigned to {taken_pins[pin]} - ignoring, "
                  f"probably bounce or a stray signal)")
            continue

        return pin


def _write_config(assigned):
    """Rewrite config.py's BUTTON_PINS block in place with the newly detected pins."""
    text = CONFIG_PATH.read_text()

    # Keep config.BUTTON_PINS's existing key order, just swap in the new pin
    # numbers, and reproduce the original's aligned-colon formatting.
    order = list(config.BUTTON_PINS.keys())
    key_strs = [f'"{name}":' for name in order]
    width = max(len(k) for k in key_strs)
    lines = [f'    {k.ljust(width)} {assigned[name]},' for k, name in zip(key_strs, order)]
    new_block = "BUTTON_PINS = {\n" + "\n".join(lines) + "\n}"

    new_text, n = re.subn(r"BUTTON_PINS = \{[^}]*\}", new_block, text)
    if n != 1:
        print("Could not find a single BUTTON_PINS block in config.py - not modified.", file=sys.stderr)
        sys.exit(1)

    backup_path = CONFIG_PATH.with_suffix(".py.bak")
    backup_path.write_text(text)
    CONFIG_PATH.write_text(new_text)
    print(f"(previous config.py saved to {backup_path.name})")


def main():
    ser = serial.Serial(config.BUTTON_PORT, config.BUTTON_BAUD, timeout=1)
    sleep(2)  # let the Arduino finish its auto-reset after the port opens
    ser.reset_input_buffer()

    print("Button setup wizard. Press each button when prompted below. Ctrl+C to abort.\n")

    assigned    = {}  # name -> pin
    taken_pins  = {}  # pin -> name, for duplicate detection

    try:
        for name in PROMPT_ORDER:
            print(f"Press {name}...")
            pin = _read_press(ser, taken_pins)
            assigned[name] = pin
            taken_pins[pin] = name
            print(f"  -> {name} = pin {pin}\n")
    except KeyboardInterrupt:
        print("\nAborted - config.py not modified.")
        return

    print("=== Detected mapping ===")
    for name, pin in assigned.items():
        print(f"{name}: pin {pin}")

    answer = input("\nWrite this into config.py's BUTTON_PINS? [y/N] ").strip().lower()
    if answer != "y":
        print("Not written.")
        return

    _write_config(assigned)
    print("config.py updated.")


if __name__ == "__main__":
    main()
