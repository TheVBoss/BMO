#!/usr/bin/env python3
"""
actions.py — Motor control, camera/photo, AI action dispatch, and the
button-box daemon, all in one file (merged from the former
motor_functions.py, photo_analysing.py, actions.py, and button_control.py).

Running this file directly (`python actions.py`) launches the button-box
daemon (formerly button_control.py's job) - see extra/bmo-buttons.service.
Everything else here is imported as a library by main.py/ai_brain.py.
"""
import base64
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import config

# gpiod (encoder GPIO access), fusion_hat (motor driver), and evdev (virtual
# mouse for the button box) are Linux/Pi-only and won't import on Windows -
# only load them when actually on the Pi. Motor functions below short-circuit
# into harmless stubs when not on the Pi instead of touching gpiod/fusion_hat;
# the button daemon (main() below) is simply never invoked off the Pi, so
# evdev only needs to actually exist when it is.
if config.is_raspberry_pi:
    from fusion_hat.motor import Motor  # type: ignore[import-not-found]
    import gpiod  # type: ignore[import-not-found]
    from gpiod.line import Edge, Bias  # type: ignore[import-not-found]
    import serial  # type: ignore[import-not-found]
    from evdev import UInput, ecodes as e  # type: ignore[import-not-found]

from config import enable_text_to_speech, name, is_raspberry_pi
from config import api_key, use_camera, test_image, vision_model, vision_max_tokens, camera_sound
from display_change import show, speak_with_display, stop as stop_display
from groq import Groq


def _log_run_header(label):
    print(f"\n=== {label} start ===")


# ---------------------------------------------------------------------------
# Motor control (formerly motor_functions.py)
# ---------------------------------------------------------------------------

# Wheel and encoder constants - shared by any PID drive function (DrivePID,
# TurnPID, etc.) since they describe the hardware, not a specific maneuver.
# Raw values live in config.py; derive the rest here.
WHEEL_CIRCUMFERENCE_MM = math.pi * config.WHEEL_DIAMETER_MM       # Wheel circumference in mm (diameter x pi)
PULSES_PER_REV = config.ENCODER_PPR * config.GEAR_RATIO            # Total pulses per wheel revolution
MM_PER_PULSE = WHEEL_CIRCUMFERENCE_MM / PULSES_PER_REV              # mm of travel per encoder pulse

# Turning geometry, used by TurnPID. ROBOT_WIDTH_MM is the left/right wheel
# center-to-center track width; WHEEL_WIDTH_MM is the tire contact-patch
# width. Subtracting it approximates the effective pivot circle the wheels
# actually trace on the ground (rather than the wheels' centerline circle).
TURNING_CIRCUMFERENCE_MM = (config.ROBOT_WIDTH_MM - config.WHEEL_WIDTH_MM) * math.pi

CONTROL_DT = config.CONTROL_DT


class _StubMotor:
    # Stands in for fusion_hat.motor.Motor when not on the Pi - power()/stop()
    # are no-ops so the PID functions' motor calls don't raise, even though
    # those functions short-circuit before doing anything meaningful anyway.
    def __init__(self, motor_id, is_reversed=False):
        self.motor_id = motor_id
        self.is_reversed = is_reversed

    def power(self, _value):
        pass

    def stop(self):
        pass


# Build the four named motors and the wheel-position -> encoder-pin mapping
# from config.MOTORS / config.ENCODER_PINS (see extra/motor_setup.py
# for how those were determined).
_MotorClass = Motor if config.is_raspberry_pi else _StubMotor
motors = {
    position: _MotorClass(spec["id"], is_reversed=spec["reversed"])
    for position, spec in config.MOTORS.items()
}
ENCODER_PIN = {
    config.MOTORS[position]["id"]: pin
    for position, pin in config.ENCODER_PINS.items()
}
PIN_TO_MOTOR = {pin: motor for motor, pin in ENCODER_PIN.items()}

# Named by wheel position, not by M-number - which M-id belongs to which
# wheel is whatever config.MOTORS says, and can change independently of the
# left/right, front/back groupings the PID loops below rely on.
motor_left_front  = motors["LeftFront"]
motor_left_back   = motors["LeftBack"]
motor_right_front = motors["RightFront"]
motor_right_back  = motors["RightBack"]

id_left_front  = config.MOTORS["LeftFront"]["id"]
id_left_back   = config.MOTORS["LeftBack"]["id"]
id_right_front = config.MOTORS["RightFront"]["id"]
id_right_back  = config.MOTORS["RightBack"]["id"]


def stop():
    motor_left_front.stop()
    motor_left_back.stop()
    motor_right_front.stop()
    motor_right_back.stop()


def DrivePID(distance, max_power):
    # Drives the robot for `distance` mm (negative = reverse) using a PID
    # control loop, with proportional left/right steering correction and
    # front/back motor-balance correction on each side.

    if not config.is_raspberry_pi:
        print(f"[DrivePID] Stub (not on Pi): would drive {distance}mm at power {max_power}.")
        return

    if distance > 0:
        pre_dist = _scan_distance_mm()
        if pre_dist is not None and pre_dist <= config.safety_stop_distance_mm:
            output_text(f"I can't drive forward — there's an obstacle only {int(pre_dist)}mm away! I need at least {config.safety_stop_distance_mm}mm of clearance.")
            return

    gains = config.DRIVE_PID
    ACCEL_PER_SEC_FACTOR = gains["ACCEL_PER_SEC_FACTOR"]  # accel_rate = max_power / this

    # PID gains for the distance error (error = target - current_distance) -> drives motor power.
    # KP_DIST picks between two fixed values based on how long this drive
    # is - a long drive needs a different KP than a short one to avoid
    # being too sluggish (long) or overshooting/oscillating (short).
    KP_DIST = gains["KP_DIST_LONG"] if abs(distance) > gains["KP_DIST_THRESHOLD_MM"] else gains["KP_DIST_SHORT"]
    KI_DIST = gains["KI_DIST"]
    KD_DIST = gains["KD_DIST"]

    # Derivative low-pass filter coefficient (0..1). Lower = more smoothing.
    # Derivative is computed on the measurement (distance), not the error, to
    # avoid a derivative spike whenever the target/error changes ("derivative kick").
    D_FILTER_ALPHA = gains["D_FILTER_ALPHA"]

    # Proportional-only steering correction for left/right pulse difference (Pcont)
    KP_STEER = gains["KP_STEER"]

    # Proportional correction between the two motors on the SAME side (front vs
    # back). They get commanded the same power, but motor-to-motor variance means
    # their actual speeds can drift apart and fight each other / scrub the wheels.
    # This nudges power away from whichever motor has more cumulative pulses
    # (faster) and toward whichever has fewer (slower).
    KP_BALANCE = gains["KP_BALANCE"]

    def _split_power(power, pulses_a, pulses_b):
        # Returns (power_a, power_b) such that the faster motor (more pulses)
        # gets less power and the slower motor gets more, magnitude-only so it
        # works the same regardless of forward/reverse sign.
        sign = -1 if power < 0 else 1
        magnitude = abs(power)
        balance = KP_BALANCE * (pulses_a - pulses_b)
        mag_a = max(0, min(max_power, magnitude - balance))
        mag_b = max(0, min(max_power, magnitude + balance))
        return int(sign * mag_a), int(sign * mag_b)

    direction = -1 if distance < 0 else 1
    target_pulses = int(abs(distance) / MM_PER_PULSE)
    accel_rate = max_power / ACCEL_PER_SEC_FACTOR  # max power change per second (smooth acceleration / slew limit)
    integral_limit = max_power / KI_DIST if KI_DIST else 0  # clamp so the integral term alone can't exceed max_power (anti-windup)

    pulses = {'M0': 0, 'M1': 0, 'M2': 0, 'M3': 0}

    def set_left(power):
        power = max(-max_power, min(max_power, power))
        power_lf, power_lb = _split_power(power, pulses[id_left_front], pulses[id_left_back])
        motor_left_front.power(power_lf)
        motor_left_back.power(power_lb)

    def set_right(power):
        power = max(-max_power, min(max_power, power))
        power_rf, power_rb = _split_power(power, pulses[id_right_front], pulses[id_right_back])
        motor_right_front.power(power_rf)
        motor_right_back.power(power_rb)

    def all_stop():
        motor_left_front.stop()
        motor_left_back.stop()
        motor_right_front.stop()
        motor_right_back.stop()

    _log_run_header("DrivePID")
    print(f"Target distance: {distance}mm")
    print(f"Expected pulses: {target_pulses}")
    print("---")

    integral_error = 0.0    # accumulated distance error, for KI_DIST
    prev_distance = 0.0     # previous measured distance, for filtered derivative
    derivative_filtered = 0.0
    smoothed_power = 0.0    # slew-limited power actually applied (smooth acceleration)

    pulses_left = pulses_right = 0
    elapsed = 0.0

    safety_stop_event = threading.Event()
    safety_triggered  = threading.Event()
    if distance > 0:
        threading.Thread(
            target=_ultrasonic_safety_worker,
            args=(safety_stop_event, safety_triggered),
            daemon=True,
            name="drive_safety",
        ).start()

    try:
        with gpiod.request_lines(
            config.ENCODER_GPIO_CHIP,
            consumer="encoder",
            config={
                # bias=PULL_DOWN holds the line low when idle so it can't float and
                # cause spurious/missed RISING edges - adjust to PULL_UP if your
                # encoder's idle state is high instead of low.
                pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
                for pin in ENCODER_PIN.values()
            }
        ) as request:
            start = time.time()
            next_control = start + CONTROL_DT

            # Stop on the AVERAGE of both sides reaching target, so a steering
            # imbalance doesn't cut the move short the instant the faster side
            # crosses the line.
            avg_pulses = 0
            while avg_pulses < target_pulses:
                if safety_triggered.is_set():
                    break
                # Drain encoder edges as fast as they arrive (non-blocking-ish).
                if request.wait_edge_events(timeout=0.001):
                    for event in request.read_edge_events():
                        motor = PIN_TO_MOTOR.get(event.line_offset)
                        if motor:
                            pulses[motor] += 1

                pulses_left = (pulses[id_left_front] + pulses[id_left_back]) / 2
                pulses_right = (pulses[id_right_front] + pulses[id_right_back]) / 2
                avg_pulses = (pulses_left + pulses_right) / 2

                now = time.time()
                if now < next_control:
                    continue  # not time for a control update yet; keep counting pulses
                dt = now - (next_control - CONTROL_DT)
                next_control = now + CONTROL_DT

                # PID on distance error (magnitude) -> commanded power (magnitude).
                current_distance = avg_pulses * MM_PER_PULSE
                error = abs(distance) - current_distance

                integral_error += error * dt
                integral_error = max(-integral_limit, min(integral_limit, integral_error))  # anti-windup

                # Derivative on measurement (note the sign: d(error)/dt = -d(distance)/dt
                # for a fixed target), then low-pass filtered to reject pulse jitter.
                raw_derivative = -(current_distance - prev_distance) / dt
                derivative_filtered += D_FILTER_ALPHA * (raw_derivative - derivative_filtered)

                pid_power = (KP_DIST * error) + (KI_DIST * integral_error) + (KD_DIST * derivative_filtered)
                if abs(error) <= 2:
                    # Close enough - don't fight for sub-encoder-resolution precision.
                    # Must actually break here, not just zero the power: the while
                    # loop's exit condition (avg_pulses < target_pulses) may never
                    # become true if the last fractional pulse needs nonzero power
                    # to reach - zeroing power without breaking would loop forever.
                    break
                pid_power = max(0, min(max_power, pid_power))

                # Smooth acceleration - slew-limit how fast the applied power can change
                max_step = accel_rate * dt
                smoothed_power += max(-max_step, min(max_step, pid_power - smoothed_power))

                print(f"[DrivePID] error={error:7.1f}mm  power={smoothed_power:6.1f}  "
                      f"pulses M0={pulses['M0']} M1={pulses['M1']} M2={pulses['M2']} M3={pulses['M3']}")

                # Proportional-only steering correction for left/right pulse difference
                diff = pulses_right - pulses_left
                steer = KP_STEER * diff

                # Clamp each side's magnitude to >=0 BEFORE applying direction - a big
                # steer correction should only ever reduce a wheel's power toward
                # zero, never flip its sign and drive it the opposite way.
                right_magnitude = max(0, smoothed_power - steer)
                left_magnitude = max(0, smoothed_power + steer)

                set_right(direction * right_magnitude)
                set_left(direction * left_magnitude)

                prev_distance = current_distance

            elapsed = time.time() - start

    finally:
        safety_stop_event.set()
        all_stop()
        for motor in ('M0', 'M1', 'M2', 'M3'):
            motor_distance = pulses[motor] * MM_PER_PULSE
            print(f"{motor} pulses: {pulses[motor]} | Distance: {motor_distance:.1f}mm")
        distance_right = pulses_right * MM_PER_PULSE
        distance_left = pulses_left * MM_PER_PULSE
        print(f"Right avg pulses: {pulses_right:.1f} | Distance: {distance_right:.1f}mm")
        print(f"Left  avg pulses: {pulses_left:.1f} | Distance: {distance_left:.1f}mm")
        print(f"Actual time: {elapsed:.2f} seconds")
        time.sleep(.1)

    if safety_triggered.is_set():
        output_text("Emergency stop! I detected an obstacle while driving.")

    # Not used by any real command path (get_movement_type discards it) -
    # exists so tools like extra/motor_setup.py's steering-gain tuner can
    # measure the resulting left/right imbalance without re-parsing stdout.
    return {"pulses_left": pulses_left, "pulses_right": pulses_right, "pulses": dict(pulses)}


def TurnPID(angle, max_power):
    # Turns the robot in place to `angle` degrees (positive = CW, negative =
    # CCW) using a PID control loop. Mirrors DrivePID's structure, just with
    # angle/arc-length in place of distance, and the two sides driven in
    # opposite directions for a pivot turn instead of together.

    if not config.is_raspberry_pi:
        print(f"[TurnPID] Stub (not on Pi): would turn {angle} degrees at power {max_power}.")
        return

    gains = config.TURN_PID
    ACCEL_PER_SEC_FACTOR = gains["ACCEL_PER_SEC_FACTOR"]  # accel_rate = max_power / this

    # PID gains for the angle error (error = target - current_angle) -> drives motor power
    KP_TURN = gains["KP_TURN"]
    KI_TURN = gains["KI_TURN"]
    KD_TURN = gains["KD_TURN"]

    # Derivative low-pass filter coefficient (0..1). Lower = more smoothing.
    # Derivative is computed on the measurement (angle), not the error, to
    # avoid a derivative spike whenever the target/error changes ("derivative kick").
    D_FILTER_ALPHA = gains["D_FILTER_ALPHA"]

    # Proportional-only correction keeping both sides' arc length in sync
    # during the turn (the turning equivalent of DrivePID's KP_STEER).
    KP_SYNC = gains["KP_SYNC"]

    # Proportional correction between the two motors on the SAME side (front vs
    # back). They get commanded the same power, but motor-to-motor variance means
    # their actual speeds can drift apart and fight each other / scrub the wheels.
    # This nudges power away from whichever motor has more cumulative pulses
    # (faster) and toward whichever has fewer (slower).
    KP_BALANCE = gains["KP_BALANCE"]

    def _split_power(power, pulses_a, pulses_b):
        # Returns (power_a, power_b) such that the faster motor (more pulses)
        # gets less power and the slower motor gets more, magnitude-only so it
        # works the same regardless of forward/reverse sign.
        sign = -1 if power < 0 else 1
        magnitude = abs(power)
        balance = KP_BALANCE * (pulses_a - pulses_b)
        mag_a = max(0, min(max_power, magnitude - balance))
        mag_b = max(0, min(max_power, magnitude + balance))
        return int(sign * mag_a), int(sign * mag_b)

    direction = -1 if angle > 0 else 1  # positive angle = CW, negative = CCW
    target_pulses = int(abs(angle) / 360 * TURNING_CIRCUMFERENCE_MM / MM_PER_PULSE)
    accel_rate = max_power / ACCEL_PER_SEC_FACTOR  # max power change per second (smooth acceleration / slew limit)
    integral_limit = max_power / KI_TURN if KI_TURN else 0  # clamp so the integral term alone can't exceed max_power (anti-windup)

    pulses = {'M0': 0, 'M1': 0, 'M2': 0, 'M3': 0}

    def set_left(power):
        power = max(-max_power, min(max_power, power))
        power_lf, power_lb = _split_power(power, pulses[id_left_front], pulses[id_left_back])
        motor_left_front.power(power_lf)
        motor_left_back.power(power_lb)

    def set_right(power):
        power = max(-max_power, min(max_power, power))
        power_rf, power_rb = _split_power(power, pulses[id_right_front], pulses[id_right_back])
        motor_right_front.power(power_rf)
        motor_right_back.power(power_rb)

    def all_stop():
        motor_left_front.stop()
        motor_left_back.stop()
        motor_right_front.stop()
        motor_right_back.stop()

    _log_run_header("TurnPID")
    print(f"Target angle: {angle} degrees")
    print(f"Expected pulses: {target_pulses}")
    print("---")

    integral_error = 0.0    # accumulated angle error, for KI_TURN
    prev_angle = 0.0         # previous measured angle, for filtered derivative
    derivative_filtered = 0.0
    smoothed_power = 0.0    # slew-limited power actually applied (smooth acceleration)

    pulses_left = pulses_right = 0
    elapsed = 0.0

    try:
        with gpiod.request_lines(
            config.ENCODER_GPIO_CHIP,
            consumer="encoder",
            config={
                # bias=PULL_DOWN holds the line low when idle so it can't float and
                # cause spurious/missed RISING edges - adjust to PULL_UP if your
                # encoder's idle state is high instead of low.
                pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
                for pin in ENCODER_PIN.values()
            }
        ) as request:
            start = time.time()
            next_control = start + CONTROL_DT

            # Stop on the AVERAGE of both sides reaching target, so a sync
            # imbalance doesn't cut the turn short the instant the faster side
            # crosses the line. Each side's pulses are unsigned (magnitude
            # only), which is fine here since each side only ever spins one
            # way for a given turn.
            avg_pulses = 0
            while avg_pulses < target_pulses:
                # Drain encoder edges as fast as they arrive (non-blocking-ish).
                if request.wait_edge_events(timeout=0.001):
                    for event in request.read_edge_events():
                        motor = PIN_TO_MOTOR.get(event.line_offset)
                        if motor:
                            pulses[motor] += 1

                pulses_left = (pulses[id_left_front] + pulses[id_left_back]) / 2
                pulses_right = (pulses[id_right_front] + pulses[id_right_back]) / 2
                avg_pulses = (pulses_left + pulses_right) / 2

                now = time.time()
                if now < next_control:
                    continue  # not time for a control update yet; keep counting pulses
                dt = now - (next_control - CONTROL_DT)
                next_control = now + CONTROL_DT

                # PID on angle error (magnitude) -> commanded power (magnitude).
                # Arc length travelled is avg_pulses worth of wheel travel;
                # converting that to degrees turned is the inverse of the
                # target_pulses calculation above.
                current_angle = (avg_pulses * MM_PER_PULSE) * (360 / TURNING_CIRCUMFERENCE_MM)
                error = abs(angle) - current_angle

                integral_error += error * dt
                integral_error = max(-integral_limit, min(integral_limit, integral_error))  # anti-windup

                # Derivative on measurement (note the sign: d(error)/dt = -d(angle)/dt
                # for a fixed target), then low-pass filtered to reject pulse jitter.
                raw_derivative = -(current_angle - prev_angle) / dt
                derivative_filtered += D_FILTER_ALPHA * (raw_derivative - derivative_filtered)

                pid_power = (KP_TURN * error) + (KI_TURN * integral_error) + (KD_TURN * derivative_filtered)
                if abs(error) <= 1:
                    # Close enough - don't fight for sub-encoder-resolution precision.
                    # Must actually break here, not just zero the power: the while
                    # loop's exit condition (avg_pulses < target_pulses) may never
                    # become true if the last fractional pulse needs nonzero power
                    # to reach - zeroing power without breaking would loop forever.
                    break
                pid_power = max(0, min(max_power, pid_power))

                # Smooth acceleration - slew-limit how fast the applied power can change
                max_step = accel_rate * dt
                smoothed_power += max(-max_step, min(max_step, pid_power - smoothed_power))

                print(f"[TurnPID] error={error:7.1f}deg  power={smoothed_power:6.1f}  "
                      f"pulses M0={pulses['M0']} M1={pulses['M1']} M2={pulses['M2']} M3={pulses['M3']}")

                # Proportional-only correction keeping both sides' pulse counts in sync
                diff = pulses_right - pulses_left
                sync = KP_SYNC * diff

                # Clamp each side's magnitude to >=0 BEFORE applying direction - a big
                # sync correction should only ever reduce a wheel's power toward
                # zero, never flip its sign and drive it the opposite way.
                right_magnitude = max(0, smoothed_power - sync)
                left_magnitude = max(0, smoothed_power + sync)

                # Pivot turn: the two sides spin in opposite directions. CCW
                # (direction=1) drives right forward, left backward; CW
                # (direction=-1) does the opposite.
                set_right(direction * right_magnitude)
                set_left(-direction * left_magnitude)

                prev_angle = current_angle

            elapsed = time.time() - start

    finally:
        all_stop()
        for motor in ('M0', 'M1', 'M2', 'M3'):
            motor_angle = (pulses[motor] * MM_PER_PULSE) * (360 / TURNING_CIRCUMFERENCE_MM)
            print(f"{motor} pulses: {pulses[motor]} | Angle: {motor_angle:.1f} degrees")
        print(f"Right avg pulses: {pulses_right:.1f}")
        print(f"Left  avg pulses: {pulses_left:.1f}")
        print(f"Actual time: {elapsed:.2f} seconds")
        time.sleep(.1)

    # Not used by any real command path (get_movement_type discards it) -
    # exists so tools like extra/motor_setup.py's steering-gain tuner can
    # measure the resulting left/right imbalance without re-parsing stdout.
    return {"pulses_left": pulses_left, "pulses_right": pulses_right, "pulses": dict(pulses)}


def DriveTime(direction, duration, max_power):
    # Drives the robot open-loop for `duration` seconds (direction: 1=forward,
    # -1=backward). Power ramps up linearly to peak at exactly max_power at
    # the midpoint (duration/2), then back down linearly to 0 - a clean
    # triangle scaled to this command's own duration, rather than a fixed
    # acceleration rate that could plateau at max_power early (long
    # duration) or never reach it at all (short duration). Still no
    # distance target (that's what DrivePID is for - this stops on time,
    # not pulses), but applies DrivePID's same live KP_STEER left/right
    # correction from encoder pulses so it doesn't curve like a truly blind
    # open-loop drive would.
    if not config.is_raspberry_pi:
        print(f"[DriveTime] Stub (not on Pi): would drive {'forward' if direction > 0 else 'backward'} for {duration}s at power {max_power}.")
        return

    if direction > 0:
        pre_dist = _scan_distance_mm()
        if pre_dist is not None and pre_dist <= config.safety_stop_distance_mm:
            output_text(f"I can't drive forward — there's an obstacle only {int(pre_dist)}mm away! I need at least {config.safety_stop_distance_mm}mm of clearance.")
            return

    KP_STEER = config.DRIVE_PID["KP_STEER"]
    half = duration / 2.0

    pulses = {'M0': 0, 'M1': 0, 'M2': 0, 'M3': 0}

    def all_stop():
        motor_left_front.stop()
        motor_left_back.stop()
        motor_right_front.stop()
        motor_right_back.stop()

    safety_stop_event = threading.Event()
    safety_triggered  = threading.Event()
    if direction > 0:
        threading.Thread(
            target=_ultrasonic_safety_worker,
            args=(safety_stop_event, safety_triggered),
            daemon=True,
            name="drive_safety",
        ).start()

    _log_run_header("DriveTime")
    print(f"Direction: {'forward' if direction > 0 else 'backward'}, Duration: {duration}s")

    start = time.time()
    try:
        with gpiod.request_lines(
            config.ENCODER_GPIO_CHIP,
            consumer="encoder",
            config={
                pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
                for pin in ENCODER_PIN.values()
            }
        ) as request:
            while True:
                elapsed = time.time() - start
                if elapsed >= duration or safety_triggered.is_set():
                    break

                if request.wait_edge_events(timeout=0.001):
                    for event in request.read_edge_events():
                        motor = PIN_TO_MOTOR.get(event.line_offset)
                        if motor:
                            pulses[motor] += 1

                # Triangular ramp: 0 at elapsed=0, max_power at elapsed=half, 0 at elapsed=duration.
                current_power = max_power * max(0.0, 1 - abs(elapsed - half) / half) if half > 0 else 0.0

                # Same live proportional steering correction as DrivePID's
                # KP_STEER, just riding on top of the time-based ramp above
                # instead of a distance-error-driven PID power.
                pulses_left  = (pulses[id_left_front] + pulses[id_left_back]) / 2
                pulses_right = (pulses[id_right_front] + pulses[id_right_back]) / 2
                steer = KP_STEER * (pulses_right - pulses_left)
                right_magnitude = max(0, min(max_power, current_power - steer))
                left_magnitude  = max(0, min(max_power, current_power + steer))
                applied_left  = int(direction * left_magnitude)
                applied_right = int(direction * right_magnitude)

                print(f"[DriveTime] elapsed={elapsed:.2f}s  power={int(direction * current_power)}  "
                      f"L={applied_left} R={applied_right}")
                motor_left_front.power(applied_left)
                motor_left_back.power(applied_left)
                motor_right_front.power(applied_right)
                motor_right_back.power(applied_right)
                time.sleep(CONTROL_DT)
    finally:
        safety_stop_event.set()
        all_stop()
        print(f"[DriveTime] Done: {time.time() - start:.2f}s elapsed.  "
              f"pulses M0={pulses['M0']} M1={pulses['M1']} M2={pulses['M2']} M3={pulses['M3']}")

    if safety_triggered.is_set():
        output_text("Emergency stop! I detected an obstacle while driving.")


def TurnTime(direction, duration, max_power):
    # Pivots the robot open-loop for `duration` seconds (direction: 1=CCW/left,
    # -1=CW/right). Same triangular ramp profile as DriveTime - power peaks
    # at exactly max_power at the midpoint (duration/2) regardless of how
    # long duration is, then ramps back down. Still no angle target (that's
    # what TurnPID is for - this stops on time, not pulses), but applies
    # TurnPID's live KP_SYNC correction so both sides' pulse counts stay in
    # step during the turn instead of drifting apart uncorrected.
    if not config.is_raspberry_pi:
        print(f"[TurnTime] Stub (not on Pi): would turn {'left' if direction > 0 else 'right'} for {duration}s at power {max_power}.")
        return

    KP_SYNC = config.TURN_PID["KP_SYNC"]
    half = duration / 2.0

    pulses = {'M0': 0, 'M1': 0, 'M2': 0, 'M3': 0}

    def all_stop():
        motor_left_front.stop()
        motor_left_back.stop()
        motor_right_front.stop()
        motor_right_back.stop()

    _log_run_header("TurnTime")
    print(f"Direction: {'CCW/left' if direction > 0 else 'CW/right'}, Duration: {duration}s")

    start = time.time()
    try:
        with gpiod.request_lines(
            config.ENCODER_GPIO_CHIP,
            consumer="encoder",
            config={
                pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
                for pin in ENCODER_PIN.values()
            }
        ) as request:
            while True:
                elapsed = time.time() - start
                if elapsed >= duration:
                    break

                if request.wait_edge_events(timeout=0.001):
                    for event in request.read_edge_events():
                        motor = PIN_TO_MOTOR.get(event.line_offset)
                        if motor:
                            pulses[motor] += 1

                # Triangular ramp: 0 at elapsed=0, max_power at elapsed=half, 0 at elapsed=duration.
                current_power = max_power * max(0.0, 1 - abs(elapsed - half) / half) if half > 0 else 0.0

                # Same live proportional sync correction as TurnPID's
                # KP_SYNC, just riding on top of the time-based ramp above
                # instead of an angle-error-driven PID power.
                pulses_left  = (pulses[id_left_front] + pulses[id_left_back]) / 2
                pulses_right = (pulses[id_right_front] + pulses[id_right_back]) / 2
                sync = KP_SYNC * (pulses_right - pulses_left)
                right_magnitude = max(0, min(max_power, current_power - sync))
                left_magnitude  = max(0, min(max_power, current_power + sync))
                applied_right = int(direction * right_magnitude)
                applied_left  = int(-direction * left_magnitude)

                print(f"[TurnTime] elapsed={elapsed:.2f}s  power={int(current_power)}  "
                      f"L={applied_left} R={applied_right}")
                # Pivot turn: CCW (direction=1) → right forward, left backward
                motor_right_front.power(applied_right)
                motor_right_back.power(applied_right)
                motor_left_front.power(applied_left)
                motor_left_back.power(applied_left)
                time.sleep(CONTROL_DT)
    finally:
        all_stop()
        print(f"[TurnTime] Done: {time.time() - start:.2f}s elapsed.  "
              f"pulses M0={pulses['M0']} M1={pulses['M1']} M2={pulses['M2']} M3={pulses['M3']}")


# ---------------------------------------------------------------------------
# Ultrasonic obstacle sensing (HC-SR04 via an Arduino running
# extra/ultrasonic_sensor/ultrasonic_sensor.ino). The Arduino only fires the
# sensor on request ('D' byte) rather than continuously pinging, so it stays
# idle/off whenever nothing is asking it for a reading.
# ---------------------------------------------------------------------------
ULTRASONIC_PORT = config.ULTRASONIC_PORT
ULTRASONIC_BAUD = config.ULTRASONIC_BAUD
ULTRASONIC_SAMPLES = config.ULTRASONIC_SAMPLES

_ultrasonic_serial = None
_ultrasonic_lock   = threading.Lock()  # guards serial port against concurrent safety-thread access


def get_distance():
    # Lazy + persistent: opened on first use (which also triggers the
    # Arduino's auto-reset), then reused so later calls don't pay that cost.
    # Lock guards against the safety background thread and the main thread
    # both trying to read the sensor at the same time.
    if not config.is_raspberry_pi:
        print("[Ultrasonic] Stub (not on Pi): no distance reading available.")
        return None

    global _ultrasonic_serial
    with _ultrasonic_lock:
        if _ultrasonic_serial is None:
            try:
                _ultrasonic_serial = serial.Serial(ULTRASONIC_PORT, ULTRASONIC_BAUD, timeout=1)
                time.sleep(2)  # let the Arduino finish its auto-reset after the port opens
            except serial.SerialException as e:
                print(f"[Ultrasonic] Could not open {ULTRASONIC_PORT}: {e}")
                return None

        _ultrasonic_serial.reset_input_buffer()
        _ultrasonic_serial.write(b'D')
        line = _ultrasonic_serial.readline().decode('utf-8', errors='ignore').strip()
    try:
        distance = float(line)
    except ValueError:
        return None
    return distance if distance >= 0 else None


def _scan_distance_mm():
    # Retries up to ULTRASONIC_SAMPLES times for one valid reading (not an
    # average - the Arduino only pings on request, so this just rejects
    # dropped/garbled single reads). Returns None if every attempt failed.
    for _ in range(ULTRASONIC_SAMPLES):
        seed = get_distance()
        if seed is not None:
            return seed * 10
    return None


def _ultrasonic_safety_worker(stop_event, triggered_event):
    # Background thread started by DrivePID/DriveTime for forward moves.
    # Polls the ultrasonic sensor at ~20 Hz and sets triggered_event the
    # instant the robot gets within config.safety_stop_distance_mm of anything.
    # The drive loop checks triggered_event on every ~1ms iteration for a
    # near-instant emergency stop.
    limit = config.safety_stop_distance_mm
    while not stop_event.is_set():
        dist = get_distance()
        if dist is not None and dist * 10 <= limit:
            print(f"[Safety] Obstacle at {dist * 10:.0f}mm — triggering emergency stop!")
            triggered_event.set()
            return
        time.sleep(0.05)  # 20 Hz poll


def StopPID(distance, max_power):
    # Scrapped the old direct-ultrasonic PID control loop - it fought
    # constant single-ping sensor noise and needed continuous tuning
    # (median filtering, dynamic KP, brake zones...) to avoid oscillation or
    # overshooting into the obstacle. Instead, take ONE ultrasonic reading
    # up front and hand off to the already well-tuned encoder-based
    # DrivePID: drive forward by however far the obstacle currently is,
    # minus how far from it we actually want to end up.
    #   DriveDistance = ScanDistance - StopDistance
    # Then re-scan and correct with another DrivePID call if still off by
    # more than STOP_PID_CORRECTION_TOLERANCE_MM - the correction distance
    # can come out negative (drives in reverse) if the first pass overshot.
    _log_run_header("StopPID")

    scan_distance_mm = _scan_distance_mm()
    if scan_distance_mm is None:
        print("[StopPID] No valid initial reading, aborting.")
        return

    drive_distance = scan_distance_mm - distance
    print(f"[StopPID] Scan distance: {scan_distance_mm:.1f}mm  "
          f"Target stop distance: {distance}mm  -> driving {drive_distance:.1f}mm")
    DrivePID(drive_distance, max_power)

    tolerance = config.STOP_PID_CORRECTION_TOLERANCE_MM
    for correction in range(1, config.STOP_PID_MAX_CORRECTIONS + 1):
        scan_distance_mm = _scan_distance_mm()
        if scan_distance_mm is None:
            print("[StopPID] Re-scan failed - aborting further corrections.")
            return

        error = scan_distance_mm - distance
        if abs(error) <= tolerance:
            print(f"[StopPID] Re-scan {scan_distance_mm:.1f}mm - within "
                  f"{tolerance}mm of target, done.")
            return

        print(f"[StopPID] Correction {correction}: re-scan {scan_distance_mm:.1f}mm - "
              f"off by {error:.1f}mm, driving {error:.1f}mm")
        DrivePID(error, max_power)

    print(f"[StopPID] Reached max correction attempts ({config.STOP_PID_MAX_CORRECTIONS}) - stopping.")


# ---------------------------------------------------------------------------
# Photo / camera (formerly photo_analysing.py)
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
# Use os.path.join to ensure slashes work on both Windows (\) and Linux (/)
AUDIO_PATH = os.path.join(BASE_DIR, "audio", camera_sound)

_groq_client = Groq(api_key=api_key)

# Pre-load the shutter sound into a pygame Sound object so play() has
# near-zero latency (no process spawn, no file I/O at click time).
# Falls back to ffplay at play time if pygame can't load the file.
_camera_sound_obj = None

def _try_preload_camera_sound():
    global _camera_sound_obj
    if not os.path.exists(AUDIO_PATH):
        return
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        _camera_sound_obj = pygame.mixer.Sound(AUDIO_PATH)
    except Exception:
        pass  # ffplay fallback used in play_camera_sound()

_try_preload_camera_sound()


def play_camera_sound():
    if not os.path.exists(AUDIO_PATH):
        print(f"[AUDIO] Sound file not found: {AUDIO_PATH}")
        return
    if _camera_sound_obj is not None:
        try:
            _camera_sound_obj.play()
            return
        except Exception as exc:
            print(f"[AUDIO] pygame Sound play failed: {exc}")
    try:
        cmd = [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-noborder", "-probesize", "32", "-analyzeduration", "0",
            "-sync", "ext", AUDIO_PATH,
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[AUDIO] ffplay failed: {exc}")


def take_photo():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if is_raspberry_pi and use_camera:
        image_path = os.path.join(IMAGES_DIR, "capture.jpg")
        # Shutter sound right as the camera actually fires, not before.
        play_camera_sound()
        subprocess.run(
            ["libcamera-jpeg", "-o", image_path, "--nopreview", "-t", "1"],
            check=True
        )
    else:
        image_path = os.path.join(IMAGES_DIR, test_image)
        # Shutter sound right as the (test) image is actually read, not before.
        play_camera_sound()
        if not is_raspberry_pi:
            print(f"[PHOTO] Windows mode — using test image: {test_image}")
        else:
            print(f"[PHOTO] use_camera=False — using test image: {test_image}")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = _groq_client.chat.completions.create(
        model=vision_model,
        max_tokens=vision_max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": "Describe what you see in 1-2 sentences, as if you are a robot reporting back to a user."}
            ]
        }]
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# AI action dispatcher (formerly actions.py)
# ---------------------------------------------------------------------------

def is_recognised_command(action, duration, distance, stop_at, angle):
    """
    True if this is a command get_movement_type actually knows how to run.
    Used to validate a whole batch of parsed commands upfront so one bad
    command doesn't let the others run anyway (see ai_brain.parse_and_execute).
    """
    is_turn = action in ("left", "right")

    no_extra_fields = not duration and not distance and not stop_at and not angle

    if action in ("take_photo", "stop", "shutdown") and no_extra_fields:
        return True
    elif action and distance and not duration and not stop_at and not angle:
        return True
    elif action and stop_at and not duration and not distance and not angle:
        return True
    elif is_turn and angle and not duration and not distance and not stop_at:
        return True
    elif action and duration and not distance and not stop_at and not angle:
        return True
    return False


def diagnose_invalid_command(action, duration, distance, stop_at, angle):
    """Returns a short human-readable explanation of why this command is invalid."""
    is_turn = action in ("left", "right")
    if is_turn and distance:
        return "turns use degrees, not distance — try something like 'turn left 90 degrees'"
    if is_turn and stop_at:
        return "turns use angles only, stop distances don't apply to turning — try 'turn left 90 degrees'"
    if distance and stop_at:
        return "a command can't have both a travel distance and a stop-at distance at the same time"
    if duration and (distance or stop_at or angle):
        return "time-based commands can't also specify distance, stop-at, or angle — use just the time, or just the distance/angle"
    return "that combination of fields isn't valid — try rephrasing"


def motor_command_kind(action, duration, distance, stop_at, angle):
    """
    Returns "drive", "turn", "ultrasonic", or None (not motor-dependent —
    take_photo/stop/shutdown work without motors). Used to bail out of a
    whole batch with one clear, specific message instead of announcing a
    movement and then quietly no-opping it when motors aren't available
    (see ai_brain.parse_and_execute), and to pick the right "Sorry, X only
    works in Raspberry Pi mode!" wording for it.
    """
    is_turn = action in ("left", "right")
    if action and distance:
        return "drive"
    if action and stop_at:
        return "ultrasonic"
    if is_turn and angle:
        return "turn"
    if action and duration:
        return "turn" if is_turn else "drive"
    return None


PI_ONLY_MESSAGES = {
    "drive":      "Sorry, driving only works in Raspberry Pi mode!",
    "turn":       "Sorry, turning only works in Raspberry Pi mode!",
    "ultrasonic": "Sorry, using the ultrasonic sensor only works in Raspberry Pi mode!",
}
PI_ONLY_FALLBACK_MESSAGE = "Sorry, motor commands only work in Raspberry Pi mode!"


def describe_action(action, duration, distance, stop_at, angle):
    """
    Returns a short, lowercase, no-trailing-period phrase describing a single
    parsed command, suitable for joining multiple actions into one sentence
    (see ai_brain.parse_and_execute). Returns None for shutdown (excluded -
    its own "Goodbye!" is announced separately, right before it executes) and
    for unrecognised commands (those get their own immediate error message).
    """
    is_turn = action in ("left", "right")
    no_extra_fields = not duration and not distance and not stop_at and not angle

    if action == "take_photo" and no_extra_fields:
        return "taking a photo"
    elif action == "stop" and no_extra_fields:
        return "stopping"
    elif action == "shutdown" and no_extra_fields:
        return None
    elif action and distance and not duration and not stop_at and not angle:
        return f"driving {action} for {distance} millimeters"
    elif action and stop_at and not duration and not distance and not angle:
        return f"driving {action} until an obstacle is {stop_at} millimeters away"
    elif is_turn and angle and not duration and not distance and not stop_at:
        return f"turning {action} for {angle} degrees"
    elif action and duration and not distance and not stop_at and not angle:
        if is_turn:
            return f"turning {action} for {duration} seconds"
        return f"driving {action} for {duration} seconds"
    else:
        return None


def get_movement_type(action, duration, distance, stop_at, angle, announce=True):
    is_forward  = action in ("forwards", "left")
    is_turn     = action in ("left", "right")
    direction   = 1 if is_forward else -1
    no_extra_fields = not duration and not distance and not stop_at and not angle

    if action == "take_photo" and no_extra_fields:
        if announce:
            output_text("Taking photo.")
        show("photo")
        output_text(take_photo())

    elif action == "stop" and no_extra_fields:
        if announce:
            output_text("Stopping.")
        show("driving")
        stop()

    elif action == "shutdown" and no_extra_fields:
        shutdown()

    elif action and distance and not duration and not stop_at and not angle:
        if not is_raspberry_pi:
            output_text(PI_ONLY_MESSAGES["drive"])
        else:
            if announce:
                output_text(f"Driving {action} for {distance} millimeters.")
            show("driving")
            DrivePID(distance * direction, config.motor_max_power)

    elif action and stop_at and not duration and not distance and not angle:
        if not is_raspberry_pi:
            output_text(PI_ONLY_MESSAGES["ultrasonic"])
        else:
            if announce:
                output_text(f"Driving {action} until an obstacle is {stop_at} mm away.")
            show("driving")
            StopPID(stop_at, config.motor_max_power)

    elif is_turn and angle and not duration and not distance and not stop_at:
        if not is_raspberry_pi:
            output_text(PI_ONLY_MESSAGES["turn"])
        else:
            if announce:
                output_text(f"Turning {action} for {angle} degrees.")
            show("driving")
            # TurnPID's own sign convention (positive angle = CW) is the
            # opposite of `direction` here (1 = left/CCW, matching TurnTime
            # just below) - negate so "left"/"right" still turn the same
            # real-world way regardless of which of the two actually runs.
            TurnPID(angle * -direction, config.motor_max_power)

    elif action and duration and not distance and not stop_at and not angle:
        if not is_raspberry_pi:
            output_text(PI_ONLY_MESSAGES["turn"] if is_turn else PI_ONLY_MESSAGES["drive"])
        else:
            if announce:
                if is_turn:
                    output_text(f"Turning {action} for {duration} seconds.")
                else:
                    output_text(f"Driving {action} for {duration} seconds.")
            show("driving")
            if is_turn:
                TurnTime(direction, duration, config.motor_max_power)
            else:
                DriveTime(direction, duration, config.motor_max_power)

    # elif action == "follow_me":
    #     follow_me()

    # elif action == "stay":
    #     stay()

    # elif action == "come_closer":
    #     come_closer()

    # elif action == "keep_distance":
    #     keep_distance()

    else:
        reason = diagnose_invalid_command(action, duration, distance, stop_at, angle)
        output_text(f"I can't run that — {reason}. Want to try again?")


def output_text(text):
    if enable_text_to_speech:
        print(f"{name}: {text}")
        speak_with_display(text)
    else:
        print(text)


def shutdown():
    stop()  # kill motors first - everything below is just for show
    output_text("Goodbye!")
    show("sad")
    time.sleep(1)
    stop_display()
    # os._exit, not sys.exit: this can be called from the button daemon's
    # background thread (LargeCircle, see _handle_press() below), where
    # sys.exit() would only end that thread and leave the rest of the
    # program (and main.py's main loop) running.
    os._exit(0)


# ---------------------------------------------------------------------------
# Button-box daemon (formerly button_control.py)
# ---------------------------------------------------------------------------
# Standalone, always-on process (run via systemd, see extra/bmo-buttons.service
# - update its ExecStart to point at this file) that reads button events from
# the Arduino Micro (see config.BUTTON_PINS) and acts on them immediately:
#
#   UpArrow/DownArrow/LeftArrow/RightArrow -> move the mouse cursor
#   LargeCircle                            -> stop the motors, and if main.py
#                                              is running, tell it to shut down
#   While main.py is running:
#     SmallCircle -> left click
#     Triangle    -> skip wake word detection
#   While main.py is NOT running (mouse-only remote control mode):
#     SmallCircle -> right click
#     Triangle    -> left click
#
# This runs as its own process (via `if __name__ == "__main__"` below, not a
# thread inside main.py) so the mouse/click buttons work even when main.py
# isn't running, e.g. 24/7 on boot. Only one process can own the Arduino's
# serial port at a time, so main.py no longer reads it - this is the sole
# owner. Talking to main.py (Triangle/LargeCircle) therefore happens over OS
# signals + a pidfile rather than an in-process function call - see main.py's
# signal handlers.
#
# Mouse movement uses a virtual uinput device (evdev), not pyautogui - this
# robot's Pi runs labwc (Wayland), and pyautogui's cursor-warp only works on
# X11. uinput acts at the kernel input layer, so it works regardless of
# display server (X11, Wayland, or none).

MOUSE_STEP_PX = config.mouse_step_px
MOUSE_REPEAT_SECS = config.mouse_repeat_secs

MAIN_PID_PATH = Path(__file__).resolve().parent / "main.pid"

PIN_TO_NAME = {pin: btn_name for btn_name, pin in config.BUTTON_PINS.items()}

DIRECTION_NAMES = {"UpArrow", "DownArrow", "LeftArrow", "RightArrow"}

_ui = None  # created in main(), so import alone can't fail

# Direction buttons currently held down, repeated by _move_loop() until
# released - set/read from two different threads (serial reader + mover),
# so guarded by a lock rather than relying on the GIL for set mutation.
_held = set()
_held_lock = threading.Lock()


def _move_mouse(dx, dy):
    _ui.write(e.EV_REL, e.REL_X, dx)
    _ui.write(e.EV_REL, e.REL_Y, dy)
    _ui.syn()


def _click(button=None):
    if _ui is None:
        return  # virtual mouse unavailable, see warning printed in main()
    if button is None:
        button = e.BTN_LEFT
    _ui.write(e.EV_KEY, button, 1)
    _ui.syn()
    _ui.write(e.EV_KEY, button, 0)
    _ui.syn()


def _main_pid():
    # Returns main.py's pid if it's alive, else None. Used both to decide
    # whether we're in "mouse-only remote control mode" and to know who to
    # signal.
    try:
        pid = int(MAIN_PID_PATH.read_text().strip())
        os.kill(pid, 0)  # signal 0: liveness check only, doesn't actually signal
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def _signal_main(sig):
    # No-op if main.py isn't running - nothing to bypass/shut down.
    pid = _main_pid()
    if pid is not None:
        os.kill(pid, sig)


def _resolve_move(held):
    # Left/Right take priority over Up/Down - movement is always a single
    # axis at a time, never diagonal, and never both directions of the same
    # axis at once (e.g. Left+Right held together cancels to 0, not a jitter
    # between the two).
    if "LeftArrow" in held or "RightArrow" in held:
        dx = (-MOUSE_STEP_PX if "LeftArrow" in held else 0) + (MOUSE_STEP_PX if "RightArrow" in held else 0)
        return dx, 0
    dy = (-MOUSE_STEP_PX if "UpArrow" in held else 0) + (MOUSE_STEP_PX if "DownArrow" in held else 0)
    return 0, dy


def _move_loop():
    while True:
        with _held_lock:
            held = set(_held)
        if held:
            dx, dy = _resolve_move(held)
            if dx or dy:
                _move_mouse(dx, dy)
        time.sleep(MOUSE_REPEAT_SECS)


def _handle_press(button_name):
    main_running = _main_pid() is not None
    if button_name == "SmallCircle":
        # main.py running -> left click; not running (mouse-only mode) -> right click
        _click(e.BTN_LEFT if main_running else e.BTN_RIGHT)
    elif button_name == "LargeCircle":
        print("[ButtonControl] LargeCircle pressed - stopping motors and signalling shutdown.")
        stop()  # immediate, regardless of whether main.py is running
        _signal_main(signal.SIGTERM)  # if main.py is running, let it shut down gracefully
    elif button_name == "Triangle":
        if main_running:
            _signal_main(signal.SIGUSR1)  # tell main.py to skip wake word detection
        else:
            _click(e.BTN_LEFT)  # mouse-only mode -> left click


def main():
    """The button-box daemon's entry point - see the module-level comment above."""
    global _ui
    try:
        _ui = UInput({
            e.EV_REL: [e.REL_X, e.REL_Y],
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        }, name="bmo-button-mouse")
    except OSError as exc:
        print(f"[ButtonControl] Could not create virtual mouse device: {exc}")
        print("[ButtonControl] Mouse/click buttons disabled, Triangle/LargeCircle still work.")
        _ui = None

    if _ui is not None:
        threading.Thread(target=_move_loop, daemon=True, name="mouse_mover").start()

    ser = serial.Serial(config.BUTTON_PORT, config.BUTTON_BAUD, timeout=1)
    time.sleep(2)  # let the Arduino finish its auto-reset after the port opens
    ser.reset_input_buffer()

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

        button_name = PIN_TO_NAME.get(pin)
        if button_name is None:
            continue

        if button_name in DIRECTION_NAMES:
            if _ui is None:
                continue  # virtual mouse unavailable, see warning above
            with _held_lock:
                if state == 0:
                    _held.add(button_name)
                else:
                    _held.discard(button_name)
            continue

        if state != 0:
            continue  # only act on press, not release, for the remaining buttons

        if button_name == "SmallCircle" and _ui is None:
            continue  # virtual mouse unavailable, see warning above
        _handle_press(button_name)


if __name__ == "__main__":
    main()
