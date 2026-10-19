#!/usr/bin/env python3
# Standalone test for the HC-SR04 ultrasonic sensor. The Arduino runs
# ultrasonic_sensor/ultrasonic_sensor.ino (flash that first), which does the
# microsecond-precision trigger/echo timing itself via pulseIn() and only
# fires the sensor when it receives a 'D' request byte - it doesn't
# continuously ping, so the sensor stays idle/off between requests.
import time
import serial

PORT = "/dev/ttyACM0"
BAUD = 9600

if __name__ == "__main__":
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # let the Arduino finish its auto-reset after the port opens
    try:
        while True:
            ser.reset_input_buffer()
            ser.write(b"D")
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                print("no response")
            else:
                try:
                    d = float(line)
                    print(f"{d:.1f} cm" if d >= 0 else "out of range")
                except ValueError:
                    pass
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
