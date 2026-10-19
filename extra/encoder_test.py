#!/usr/bin/env python3
# Calibration test: drives the robot a short distance, then asks you to
# physically measure how far it actually went, so you can compare real
# mm/pulse against the theoretical value computed from wheel/gear specs.
import math
from time import time, sleep
from fusion_hat.motor import Motor
import gpiod
from gpiod.line import Edge, Bias

# Theoretical constants (same derivation as motortest2.py)
WHEEL_DIAMETER_MM = 55
WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM
ENCODER_PPR = 11
GEAR_RATIO = 21.3
PULSES_PER_REV = ENCODER_PPR * GEAR_RATIO
THEORETICAL_MM_PER_PULSE = WHEEL_CIRCUMFERENCE_MM / PULSES_PER_REV

TEST_POWER = 40      # Fixed power, no PID/steering - just enough to roll straight-ish
TEST_DURATION_S = 0.5  # How long to drive for

# Same trim value tuned in motortest2.py - keeps this run straight so the
# tape-measured distance actually matches the wheels' real travel.
TRIM = 8.8

# Pin mapping (from encoder_identify_test.py):
# 17 = M0 encoder (front right)
# 27 = M1 encoder (front left)
# 4  = M2 encoder (back right)
# 22 = M3 encoder (back left)
ENCODER_PIN = {'M0': 17, 'M1': 27, 'M2': 4, 'M3': 22}
pin_to_motor = {pin: motor for motor, pin in ENCODER_PIN.items()}

motor0 = Motor('M0', is_reversed=True)   # Front right
motor1 = Motor('M1', is_reversed=False)  # Front left
motor2 = Motor('M2', is_reversed=True)   # Back right
motor3 = Motor('M3', is_reversed=False)  # Back left

def set_left(power):
    motor1.power(int(power))
    motor3.power(int(power))

def set_right(power):
    motor0.power(int(power))
    motor2.power(int(power))

def all_stop():
    motor0.stop()
    motor1.stop()
    motor2.stop()
    motor3.stop()

pulses = {'M0': 0, 'M1': 0, 'M2': 0, 'M3': 0}

print(f"Driving at power {TEST_POWER} for {TEST_DURATION_S}s. Mark the start position now.")
sleep(2)

try:
    with gpiod.request_lines(
        "/dev/gpiochip4",
        consumer="encoder_test",
        config={
            pin: gpiod.LineSettings(edge_detection=Edge.RISING, bias=Bias.PULL_DOWN)
            for pin in ENCODER_PIN.values()
        }
    ) as request:
        start = time()
        set_left(TEST_POWER - TRIM)
        set_right(TEST_POWER + TRIM)

        while time() - start < TEST_DURATION_S:
            if request.wait_edge_events(timeout=0.001):
                for event in request.read_edge_events():
                    motor = pin_to_motor.get(event.line_offset)
                    if motor:
                        pulses[motor] += 1
finally:
    all_stop()
    sleep(.1)

pulses_left = (pulses['M1'] + pulses['M3']) / 2
pulses_right = (pulses['M0'] + pulses['M2']) / 2
avg_pulses = (pulses_right + pulses_left) / 2
theoretical_mm = avg_pulses * THEORETICAL_MM_PER_PULSE

print("---")
for motor in ('M0', 'M1', 'M2', 'M3'):
    print(f"{motor} pulses: {pulses[motor]}")
print(f"Right avg pulses: {pulses_right:.1f}")
print(f"Left  avg pulses: {pulses_left:.1f}")
print(f"Average pulses: {avg_pulses:.1f}")
print(f"Theoretical distance for that many pulses: {theoretical_mm:.1f}mm")
print("---")

measured_mm = float(input("Measure the ACTUAL distance traveled (mm) with a tape measure, then enter it: "))

actual_mm_per_pulse = measured_mm / avg_pulses
calibration_factor = actual_mm_per_pulse / THEORETICAL_MM_PER_PULSE

print("---")
print(f"Theoretical MM_PER_PULSE: {THEORETICAL_MM_PER_PULSE:.4f}")
print(f"Actual      MM_PER_PULSE: {actual_mm_per_pulse:.4f}")
print(f"Calibration factor: {calibration_factor:.4f}")
print(f"Set MM_PER_PULSE_CALIBRATION = {calibration_factor:.4f} in motortest2.py")
