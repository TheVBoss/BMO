#!/usr/bin/env python3
# Combines what used to be motor_independent_test.py and
# encoder_identify_test.py into one script:
#
#   1. Spins each wheel position (LeftFront/LeftBack/RightFront/RightBack)
#      one at a time, forward then reverse, so you can visually confirm it
#      spins the right wheel in the right direction at a sane speed -
#      isolated from the others and from any PID/trim/steering logic.
#   2. While each motor spins, watches every encoder pin to see which one
#      pulses, and reports whether that matches config.ENCODER_PINS - telling
#      you which physical encoder wire actually belongs to which wheel.
#
# Run this after wiring up a new robot (or after re-wiring anything) to fill
# in / verify config.MOTORS and config.ENCODER_PINS.
import sys
from pathlib import Path
from time import time, sleep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from fusion_hat.motor import Motor
import gpiod
from gpiod.line import Edge, Bias

TEST_POWER = 40
RUN_TIME = 1.5    # seconds per direction
PAUSE_TIME = 1.0  # seconds between motors/directions

ALL_PINS = list(config.ENCODER_PINS.values())

motors = {
    position: Motor(spec["id"], is_reversed=spec["reversed"])
    for position, spec in config.MOTORS.items()
}


def stop_all():
    for motor in motors.values():
        motor.stop()


results = {}  # position -> (detected_pin, expected_pin, match)

try:
    with gpiod.request_lines(
        config.ENCODER_GPIO_CHIP,
        consumer="motor_encoder_identify",
        config={
            pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
            for pin in ALL_PINS
        }
    ) as request:
        for position, motor in motors.items():
            expected_pin = config.ENCODER_PINS[position]
            print(f"--- {position} (motor {config.MOTORS[position]['id']}, expected encoder pin {expected_pin}) ---")

            for label, power in (("Forward", TEST_POWER), ("Reverse", -TEST_POWER)):
                pulses = {pin: 0 for pin in ALL_PINS}

                print(f"  {label} at power {power}")
                motor.power(power)

                start = time()
                while time() - start < RUN_TIME:
                    if request.wait_edge_events(timeout=0.001):
                        for event in request.read_edge_events():
                            if event.line_offset in pulses:
                                pulses[event.line_offset] += 1

                motor.stop()
                sleep(PAUSE_TIME)

                for pin in ALL_PINS:
                    print(f"    pin {pin:>2} pulses: {pulses[pin]}")

                detected_pin = max(pulses, key=pulses.get)
                if pulses[detected_pin] == 0:
                    print("    -> no pulses detected on any pin, check wiring")
                    detected_pin = None
                else:
                    print(f"    -> pin {detected_pin} is {position}'s encoder")

                if detected_pin is not None:
                    results[position] = (detected_pin, expected_pin, detected_pin == expected_pin)

            print()

finally:
    stop_all()

print("=== Summary ===")
any_mismatch = False
for position in config.MOTORS:
    if position not in results:
        print(f"{position}: NO PULSES DETECTED - check wiring")
        any_mismatch = True
        continue
    detected_pin, expected_pin, match = results[position]
    status = "OK" if match else "MISMATCH"
    if not match:
        any_mismatch = True
    print(f"{position}: encoder pin {detected_pin} (config says {expected_pin}) -> {status}")

if any_mismatch:
    print("\nUpdate config.ENCODER_PINS to match the detected pins above.")
else:
    print("\nAll encoder pins match config.ENCODER_PINS.")
