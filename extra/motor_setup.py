#!/usr/bin/env python3
"""
Interactive motor/encoder setup menu (like audio_setup.py, but for the
drivetrain). Combines what used to be motor_encoder_identify_test.py and
encoder_test.py into one place, plus an easy way to save the results:

  [1] Identify & Assign Motor IDs - spins each Fusion Hat motor slot
      (M0-M3) one at a time and asks which physical wheel just moved and
      whether that was "forward" - tests BEFORE assigning, so you never
      have to guess. Once all 4 are identified, overwrites config.MOTORS.
  [2] Spin & Identify Encoders - spins each wheel position forward then
      reverse, one at a time, so you can visually confirm it's the right
      wheel spinning the right way, while watching every encoder pin to see
      which one pulses for it. Offers to write any mismatches straight into
      config.ENCODER_PINS.
  [3] Toggle a Motor's 'reversed' Flag - lists config.MOTORS, lets you flip
      one position's reversed flag (with an optional preview spin first),
      and saves it.
  [4] Distance Calibration - commands the real DrivePID (from actions.py) to
      drive a distance you choose, at config.motor_max_power - the exact
      function and power real driving uses - then compares that commanded
      distance to what you actually measure with a tape measure, and offers
      to save a corrected config.WHEEL_DIAMETER_MM so config.MM_PER_PULSE
      (see actions.py) matches reality.
  [5] Tune Steering Correction - tests whether KP_STEER (DrivePID) or
      KP_SYNC (TurnPID) is actually correcting left/right drift the right
      way: drives/turns once with the gain forced to 0 as a baseline, once
      with a value you choose, and compares the resulting left/right pulse
      imbalance so you can tell a wrong sign or too-weak/strong gain from
      real behavior instead of guessing. Saves straight into config.py.

Run this after wiring up a new robot, after re-wiring anything, or whenever
the robot's drive distance/turn angle seems consistently off.

Run with: python3 extra/motor_setup.py
"""
import re
import sys
from pathlib import Path
from time import time, sleep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from fusion_hat.motor import Motor
import gpiod
from gpiod.line import Edge, Bias
from actions import DrivePID, TurnPID

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.py"

SPIN_POWER      = 23
SPIN_RUN_TIME   = 1.5  # seconds per direction
SPIN_PAUSE_TIME = 1.0  # seconds between motors/directions


def _fresh_motors():
    return {
        position: Motor(spec["id"], is_reversed=spec["reversed"])
        for position, spec in config.MOTORS.items()
    }


def _stop_all(motors):
    for motor in motors.values():
        motor.stop()


def _countdown(seconds=2, action="spinning"):
    """Brief heads-up before a motor moves, so it's not a surprise."""
    for remaining in range(seconds, 0, -1):
        print(f"  {action} in {remaining}...")
        sleep(1)


# ---------------------------------------------------------------------------
# Config writers
# ---------------------------------------------------------------------------

def _write_scalar(key, value):
    text = CONFIG_PATH.read_text()
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"couldn't find a '{key} =' line in config.py to update")
    CONFIG_PATH.write_text(pattern.sub(f"{key} = {value}", text, count=1))


def _write_dict_block(key, order, value_strs):
    """
    Rewrite a `key = { ... }` block in config.py, preserving key order and
    aligned-colon formatting. Matches up to the first line that starts with
    '}' (not just the first '}' anywhere) so this also works for MOTORS,
    whose values are themselves inline dicts with their own braces.
    """
    text = CONFIG_PATH.read_text()
    key_strs = [f'"{name}":' for name in order]
    width = max(len(k) for k in key_strs)
    lines = [f'    {k.ljust(width)} {value_strs[name]},' for k, name in zip(key_strs, order)]
    new_block = f"{key} = {{\n" + "\n".join(lines) + "\n}"

    pattern = re.compile(rf"^{key} = \{{.*?^\}}", re.DOTALL | re.MULTILINE)
    new_text, n = pattern.subn(new_block, text, count=1)
    if n != 1:
        raise ValueError(f"couldn't find a single '{key} = {{...}}' block in config.py")

    backup_path = CONFIG_PATH.with_suffix(".py.bak")
    backup_path.write_text(text)
    CONFIG_PATH.write_text(new_text)
    print(f"(previous config.py saved to {backup_path.name})")


def _write_dict_key(key, value):
    """
    Rewrite a single `"KEY": value,` line anywhere in config.py, preserving
    everything else on that line (trailing comment included) instead of
    rewriting the whole surrounding dict block. Assumes `key` appears
    exactly once in the file - true for KP_STEER/KP_SYNC, each only
    declared once under DRIVE_PID/TURN_PID respectively.
    """
    text = CONFIG_PATH.read_text()
    pattern = re.compile(rf'("{key}":\s*)[^,\n]+(,.*)$', re.MULTILINE)
    new_text, n = pattern.subn(rf'\g<1>{value}\g<2>', text, count=1)
    if n != 1:
        raise ValueError(f"couldn't find a single '\"{key}\":' line in config.py")

    backup_path = CONFIG_PATH.with_suffix(".py.bak")
    backup_path.write_text(text)
    CONFIG_PATH.write_text(new_text)
    print(f"(previous config.py saved to {backup_path.name})")


# ---------------------------------------------------------------------------
# 1. Identify & assign motor IDs
# ---------------------------------------------------------------------------

def identify_motor_ids():
    """
    Spins each Fusion Hat motor slot (M0-M3) one at a time - tests first,
    assigns after - so you can watch which physical wheel moves and which
    way, instead of guessing. Once a wheel position is picked for a slot, it
    drops out of the picklist for the remaining slots so the same position
    can't be assigned twice.
    """
    positions = list(config.MOTORS.keys())  # e.g. LeftFront/LeftBack/RightFront/RightBack
    motor_ids = ["M0", "M1", "M2", "M3"]
    assigned = {}  # position -> {"id": motor_id, "reversed": bool}

    print("\nLeft/Right: stand BEHIND the robot, looking the same way it faces (toward its front) - "
          "not in front of it looking back.")
    print("Each motor slot will spin briefly - watch which wheel moves.\n")

    for motor_id in motor_ids:
        motor = Motor(motor_id, is_reversed=False)

        while True:
            print(f"--- Spinning {motor_id} ---")
            _countdown()
            motor.power(SPIN_POWER)
            sleep(SPIN_RUN_TIME)
            motor.stop()

            remaining = [p for p in positions if p not in assigned]
            print("Which wheel just moved?")
            for i, p in enumerate(remaining, start=1):
                print(f"  {i}. {p}")
            print("  r. Redo (missed it - spin again)")
            print("  s. Skip (didn't move / check wiring)")
            choice = input("Choice: ").strip().lower()

            if choice == "r":
                print()
                continue
            if choice == "s":
                print(f"  Skipping {motor_id}.\n")
                break
            if choice.isdigit() and 1 <= int(choice) <= len(remaining):
                position = remaining[int(choice) - 1]
                forward_ok = input(f"  Was that {position} spinning forward? [y/N] ").strip().lower() == "y"
                reversed_flag = not forward_ok
                assigned[position] = {"id": motor_id, "reversed": reversed_flag}
                print(f"  -> {position} = {{'id': '{motor_id}', 'reversed': {reversed_flag}}}\n")
                break

            print("  Invalid choice, try again.\n")

    print("=== New MOTORS mapping ===")
    for position in positions:
        if position in assigned:
            print(f"{position}: {assigned[position]}")
        else:
            print(f"{position}: not assigned - keeping current value ({config.MOTORS[position]})")

    if not assigned:
        print("\nNothing assigned - config.py not touched.")
        return

    prompt = ("\nSave this as the new config.MOTORS? [y/N] " if len(assigned) == len(positions) else
              "\nSome positions weren't assigned - save the ones that WERE, keeping existing values "
              "for the rest? [y/N] ")
    if input(prompt).strip().lower() != "y":
        print("Not written.")
        return

    for position in positions:
        if position not in assigned:
            assigned[position] = dict(config.MOTORS[position])

    value_strs = {p: f'{{"id": "{assigned[p]["id"]}", "reversed": {assigned[p]["reversed"]}}}' for p in positions}
    _write_dict_block("MOTORS", positions, value_strs)
    print("config.py updated.")


# ---------------------------------------------------------------------------
# 2. Spin & identify encoders
# ---------------------------------------------------------------------------

SPIN_ORDER = ["LeftFront", "LeftBack", "RightFront", "RightBack"]


def spin_and_identify():
    all_pins = list(config.ENCODER_PINS.values())
    motors = _fresh_motors()
    order = [p for p in SPIN_ORDER if p in motors]
    results = {}  # position -> detected_pin

    try:
        with gpiod.request_lines(
            config.ENCODER_GPIO_CHIP,
            consumer="motor_setup",
            config={
                pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
                for pin in all_pins
            }
        ) as request:
            for position in order:
                motor = motors[position]
                expected_pin = config.ENCODER_PINS[position]
                print(f"\n--- {position} (motor {config.MOTORS[position]['id']}, "
                      f"expected encoder pin {expected_pin}) ---")

                pulses = {pin: 0 for pin in all_pins}
                print(f"  Forward at power {SPIN_POWER}")
                _countdown()
                motor.power(SPIN_POWER)

                start = time()
                while time() - start < SPIN_RUN_TIME:
                    if request.wait_edge_events(timeout=0.001):
                        for event in request.read_edge_events():
                            if event.line_offset in pulses:
                                pulses[event.line_offset] += 1

                motor.stop()
                sleep(SPIN_PAUSE_TIME)

                for pin in all_pins:
                    print(f"    pin {pin:>2} pulses: {pulses[pin]}")

                detected_pin = max(pulses, key=pulses.get)
                if pulses[detected_pin] == 0:
                    print("    -> no pulses detected on any pin, check wiring")
                else:
                    print(f"    -> pin {detected_pin} is {position}'s encoder")
                    results[position] = detected_pin
    finally:
        _stop_all(motors)

    print("\n=== Summary ===")
    any_mismatch = False
    for position in order:
        expected_pin = config.ENCODER_PINS[position]
        if position not in results:
            print(f"{position}: NO PULSES DETECTED - check wiring")
            any_mismatch = True
            continue
        detected_pin = results[position]
        match = detected_pin == expected_pin
        status = "OK" if match else "MISMATCH"
        any_mismatch = any_mismatch or not match
        print(f"{position}: encoder pin {detected_pin} (config says {expected_pin}) -> {status}")

    if not any_mismatch:
        print("\nAll encoder pins match config.ENCODER_PINS. Nothing to reassign.")
        return

    if len(results) < len(config.MOTORS):
        print("\nSome positions had no pulses detected - fix wiring before reassigning pins.")
        return

    if input("\nWrite the detected pins into config.ENCODER_PINS? [y/N] ").strip().lower() == "y":
        order = list(config.ENCODER_PINS.keys())
        _write_dict_block("ENCODER_PINS", order, {name: str(results[name]) for name in order})
        print("config.py updated.")
    else:
        print("Not written.")


# ---------------------------------------------------------------------------
# 3. Toggle a motor's reversed flag
# ---------------------------------------------------------------------------

def toggle_reversed():
    positions = list(config.MOTORS.keys())
    print("\nCurrent MOTORS:")
    for i, name in enumerate(positions, start=1):
        spec = config.MOTORS[name]
        print(f"  {i}. {name}: id={spec['id']}, reversed={spec['reversed']}")

    choice = input("\nToggle which position's 'reversed' flag? (number, or 'c' to cancel): ").strip().lower()
    if choice == "c" or not choice.isdigit() or not (1 <= int(choice) <= len(positions)):
        print("Cancelled.")
        return

    name = positions[int(choice) - 1]
    new_reversed = not config.MOTORS[name]["reversed"]

    if input(f"Spin {name} forward briefly to preview the new direction first? [y/N] ").strip().lower() == "y":
        motor = Motor(config.MOTORS[name]["id"], is_reversed=new_reversed)
        print(f"Spinning {name} forward at power {SPIN_POWER} for {SPIN_RUN_TIME}s...")
        _countdown()
        motor.power(SPIN_POWER)
        sleep(SPIN_RUN_TIME)
        motor.stop()

    if input(f"Set {name} reversed = {new_reversed}? [y/N] ").strip().lower() != "y":
        print("Not changed.")
        return

    order = list(config.MOTORS.keys())
    value_strs = {}
    for pos in order:
        spec = config.MOTORS[pos]
        reversed_val = new_reversed if pos == name else spec["reversed"]
        value_strs[pos] = f'{{"id": "{spec["id"]}", "reversed": {reversed_val}}}'
    _write_dict_block("MOTORS", order, value_strs)
    print("config.py updated.")


# ---------------------------------------------------------------------------
# 4. Distance calibration
# ---------------------------------------------------------------------------

def calibrate_distance():
    """
    Commands the real DrivePID (from actions.py) to drive a distance you
    pick, at config.motor_max_power - the exact function and power level
    real driving uses, steering correction and all - then compares that
    commanded distance to what you actually measure with a tape measure.
    """
    try:
        commanded_mm = float(input("Distance for DrivePID to drive, in mm (e.g. 1000): "))
    except ValueError:
        print("Not a number - cancelled.")
        return

    while True:
        print(f"\nDrivePID(distance={commanded_mm}, max_power={config.motor_max_power}). "
              f"Mark the start position now.")
        _countdown(action="driving")
        DrivePID(commanded_mm, config.motor_max_power)

        prompt = "\n[c] Continue - enter the measured distance   [r] Redo (drive again)   [q] Cancel: "
        valid = ("c", "r", "q")
        while True:
            choice = input(prompt).strip().lower()
            if choice in valid:
                break
            print(f"  Enter one of: {', '.join(valid)}")

        if choice == "q":
            print("Cancelled.")
            return
        if choice == "r":
            continue
        break  # choice == "c"

    try:
        measured_mm = float(input("Measure the ACTUAL distance traveled (mm) with a tape measure, then enter it: "))
    except ValueError:
        print("Not a number - cancelled.")
        return

    calibration_factor = measured_mm / commanded_mm
    new_wheel_diameter = round(config.WHEEL_DIAMETER_MM * calibration_factor, 2)

    print("---")
    print(f"Commanded: {commanded_mm}mm   Measured: {measured_mm}mm")
    print(f"Calibration factor: {calibration_factor:.4f}")
    print(f"-> WHEEL_DIAMETER_MM {config.WHEEL_DIAMETER_MM} -> {new_wheel_diameter} would match reality")

    if input(f"\nSave WHEEL_DIAMETER_MM = {new_wheel_diameter} to config.py? [y/N] ").strip().lower() == "y":
        _write_scalar("WHEEL_DIAMETER_MM", new_wheel_diameter)
        print("config.py updated.")
    else:
        print("Not saved.")


# ---------------------------------------------------------------------------
# 5. Tune steering correction (KP_STEER / KP_SYNC)
# ---------------------------------------------------------------------------

def tune_steering():
    """
    Tests whether KP_STEER (DrivePID) or KP_SYNC (TurnPID) is actually
    correcting left/right drift in the right direction, rather than just
    assuming the sign is right. Runs the real DrivePID/TurnPID (from
    actions.py) once with the gain forced to 0 as a baseline (no
    correction), then again with a value you choose, and compares the
    resulting left/right pulse imbalance between the two - if the gain is
    helping, the imbalance should shrink; if it's backwards, it'll grow.
    """
    print("\nWhich gain do you want to test?")
    print("  1. KP_STEER (DrivePID - driving straight)")
    print("  2. KP_SYNC  (TurnPID - pivot turning)")
    choice = input("Choice: ").strip()

    if choice == "1":
        gains, key = config.DRIVE_PID, "KP_STEER"
        run_test = lambda p: DrivePID(p, config.motor_max_power)
        param_prompt = "Test distance in mm (e.g. 1000): "
    elif choice == "2":
        gains, key = config.TURN_PID, "KP_SYNC"
        run_test = lambda p: TurnPID(p, config.motor_max_power)
        param_prompt = "Test angle in degrees (e.g. 180): "
    else:
        print("Cancelled.")
        return

    try:
        param = float(input(param_prompt))
    except ValueError:
        print("Not a number - cancelled.")
        return

    action = "turning" if choice == "2" else "driving"
    original_value = gains[key]
    print(f"\nCurrent {key} = {original_value}")

    def run(value, tag):
        gains[key] = value
        input(f"\n--- {tag}: {key} = {value} --- Reposition the robot, then press Enter when ready...")
        _countdown(action=action)
        result = run_test(param)
        gains[key] = original_value  # restore immediately - only ever mutated for the instant run_test() reads it
        if result is None:
            return None
        diff = result["pulses_right"] - result["pulses_left"]
        print(f"pulses_left={result['pulses_left']:.1f}  pulses_right={result['pulses_right']:.1f}  diff={diff:+.1f}")
        return diff

    baseline_diff = run(0.0, "Baseline (correction OFF)")
    if baseline_diff is None:
        print("No measurement returned - check the output above (not on Pi, or blocked by an obstacle).")
        return

    current_value = original_value
    while True:
        with_gain_diff = run(current_value, "Testing")
        if with_gain_diff is None:
            print("No measurement returned - check the output above.")
            return

        print(f"\nBaseline (no correction) imbalance: {baseline_diff:+.1f} pulses")
        print(f"With {key}={current_value} imbalance:  {with_gain_diff:+.1f} pulses")
        if abs(with_gain_diff) < abs(baseline_diff) * 0.9:
            print("-> Correcting the right way (imbalance shrank vs baseline).")
        elif abs(with_gain_diff) > abs(baseline_diff) * 1.1:
            print(f"-> Making it WORSE, not better - try flipping the sign (try {-current_value}) "
                  f"or double check the wiring/direction.")
        else:
            print("-> Not much different from baseline - try a bigger magnitude.")

        answer = input(f"\nEnter a new {key} value to try, 's' to save {current_value} to config.py, "
                        f"or 'q' to quit without saving: ").strip().lower()
        if answer == "q":
            print("Not saved.")
            return
        if answer == "s":
            _write_dict_key(key, current_value)
            gains[key] = current_value  # keep this process's in-memory config in sync with the file
            print("config.py updated.")
            return
        try:
            current_value = float(answer)
        except ValueError:
            print("Not a number, try again.")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def _print_menu():
    print("\n=== Motor Setup ===")
    print("[1] Identify & Assign Motor IDs (spin each M0-M3 slot, tell it which wheel + direction)")
    print("[2] Spin & Identify Encoders    (spin each wheel fwd/reverse, detect its encoder pin)")
    print("[3] Toggle a Motor's 'reversed' Flag")
    print("[4] Distance Calibration        (drive a short distance, measure it, calibrate wheel diameter)")
    print("[5] Tune Steering Correction    (test KP_STEER/KP_SYNC direction, adjust the gain)")
    print("[q] Quit")


def main():
    while True:
        _print_menu()
        choice = input("Choose an option: ").strip().lower()

        if choice == "1":
            identify_motor_ids()
        elif choice == "2":
            spin_and_identify()
        elif choice == "3":
            toggle_reversed()
        elif choice == "4":
            calibrate_distance()
        elif choice == "5":
            tune_steering()
        elif choice == "q":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
