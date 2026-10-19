#!/usr/bin/env python3
"""
Interactive drive-function test menu - lets you plug in specific values
(distance, angle, direction, duration) and run DrivePID/TurnPID/DriveTime/
TurnTime/StopPID directly, one at a time, without going through the full
AI command pipeline (wake word, ai_brain parsing, etc).

Always drives at config.motor_max_power - the same power level
actions.get_movement_type() uses for every real command (see actions.py) -
so a test run here behaves the same as the real thing, not some separate
tuned-down test speed.

Run with: python3 extra/drive_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from actions import DrivePID, TurnPID, DriveTime, TurnTime, StopPID

MAX_POWER = config.motor_max_power


def _read_float(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("  Not a number, try again.")


def _read_choice(prompt, choices):
    """choices: dict of accepted input -> value, e.g. {'f': 1, 'b': -1}."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in choices:
            return choices[raw]
        print(f"  Enter one of: {', '.join(choices)}")


def test_drive_pid():
    distance = _read_float("Distance in mm (negative = reverse): ")
    print(f"\nDrivePID(distance={distance}, max_power={MAX_POWER})")
    DrivePID(distance, MAX_POWER)


def test_turn_pid():
    angle = _read_float("Angle in degrees (positive = CW/right, negative = CCW/left): ")
    print(f"\nTurnPID(angle={angle}, max_power={MAX_POWER})")
    TurnPID(angle, MAX_POWER)


def test_drive_time():
    direction = _read_choice("Direction - f)orward or b)ackward: ", {"f": 1, "b": -1})
    duration = _read_float("Duration in seconds: ")
    print(f"\nDriveTime(direction={direction}, duration={duration}, max_power={MAX_POWER})")
    DriveTime(direction, duration, MAX_POWER)


def test_turn_time():
    direction = _read_choice("Direction - l)eft/CCW or r)ight/CW: ", {"l": 1, "r": -1})
    duration = _read_float("Duration in seconds: ")
    print(f"\nTurnTime(direction={direction}, duration={duration}, max_power={MAX_POWER})")
    TurnTime(direction, duration, MAX_POWER)


def test_stop_pid():
    distance = _read_float("Target stop distance from obstacle in mm: ")
    print(f"\nStopPID(distance={distance}, max_power={MAX_POWER})")
    StopPID(distance, MAX_POWER)


def _print_menu():
    print(f"\n=== Drive Test (max_power = {MAX_POWER}, from config.motor_max_power) ===")
    print("[1] DrivePID   - drive a straight-line distance (mm)")
    print("[2] TurnPID    - pivot-turn to an angle (degrees)")
    print("[3] DriveTime  - open-loop drive for a duration (seconds)")
    print("[4] TurnTime   - open-loop pivot-turn for a duration (seconds)")
    print("[5] StopPID    - drive forward until an obstacle is N mm away")
    print("[q] Quit")


def main():
    while True:
        _print_menu()
        choice = input("Choose an option: ").strip().lower()

        if choice == "1":
            test_drive_pid()
        elif choice == "2":
            test_turn_pid()
        elif choice == "3":
            test_drive_time()
        elif choice == "4":
            test_turn_time()
        elif choice == "5":
            test_stop_pid()
        elif choice == "q":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
