#!/usr/bin/env python3
# Listens for button press/release events from the Arduino Micro running
# extra/ultrasonic_sensor/ultrasonic_sensor.ino, and checks that each
# detected pin matches config.BUTTON_PINS.
#
# Run this after wiring up new buttons (or after re-wiring) to fill in /
# verify config.BUTTON_PINS. Press each button one at a time, then Ctrl+C
# to stop and see the summary.
import sys
from pathlib import Path
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import serial

PIN_TO_NAME = {pin: name for name, pin in config.BUTTON_PINS.items()}

ser = serial.Serial(config.BUTTON_PORT, config.BUTTON_BAUD, timeout=1)
sleep(2)  # let the Arduino finish its auto-reset after the port opens
ser.reset_input_buffer()

print("Press each button one at a time. Ctrl+C to stop.\n")

pressed_pins = set()

try:
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

        name = PIN_TO_NAME.get(pin, "UNKNOWN")
        event = "pressed" if state == 0 else "released"
        print(f"pin {pin:>2} ({name}) {event}")

        if state == 0:
            pressed_pins.add(pin)
            if pin not in PIN_TO_NAME:
                print(f"  -> pin {pin} is not in config.BUTTON_PINS, add it there")
except KeyboardInterrupt:
    pass

print("\n=== Summary ===")
for name, pin in config.BUTTON_PINS.items():
    status = "OK (press detected)" if pin in pressed_pins else "NOT PRESSED / NO SIGNAL"
    print(f"{name} (pin {pin}): {status}")
