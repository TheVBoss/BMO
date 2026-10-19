// HC-SR04 ultrasonic distance sensor, read directly on the Arduino.
// pulseIn() does the microsecond-precision echo timing in firmware, which
// Firmata/pyfirmata can't do reliably over serial (its pin reads only
// reflect the last periodically-reported value, far too coarse for this
// sensor's timing).
//
// On-demand: the sensor only fires when the Pi sends a 'D' byte over
// serial - it does NOT continuously ping. This keeps the sensor (and the
// Arduino's time spent driving it) idle/off unless something actually asks
// for a reading, to save power.
//
// Wiring: TRIG -> pin 9, ECHO -> pin 10, VCC -> 5V, GND -> GND.
//
// Also reports 7 ground-based buttons wired to pins 2-8 (other leg to
// GND, read with INPUT_PULLUP so idle = HIGH, pressed = LOW). Unlike the
// ultrasonic sensor these aren't polled - the instant a button's state
// changes, this sketch sends a line "B<pin>:<state>" (state 0 = pressed,
// 1 = released) on its own, so the Pi finds out as soon as it happens
// (this is what makes a kill-switch button useful). See
// config.BUTTON_PINS / extra/button_test.py on the Pi side.

const int TRIG_PIN = 9;
const int ECHO_PIN = 10;
const unsigned long ECHO_TIMEOUT_US = 30000UL;  // ~5m max range

const int BUTTON_PINS[] = {2, 3, 4, 5, 6, 7, 8};
const int NUM_BUTTONS = sizeof(BUTTON_PINS) / sizeof(BUTTON_PINS[0]);
int button_state[NUM_BUTTONS];
unsigned long button_last_change_ms[NUM_BUTTONS];
const unsigned long DEBOUNCE_MS = 30;

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  for (int i = 0; i < NUM_BUTTONS; i++) {
    pinMode(BUTTON_PINS[i], INPUT_PULLUP);
    button_state[i] = digitalRead(BUTTON_PINS[i]);
    button_last_change_ms[i] = 0;
  }
}

void poll_buttons() {
  unsigned long now = millis();
  for (int i = 0; i < NUM_BUTTONS; i++) {
    int reading = digitalRead(BUTTON_PINS[i]);
    if (reading != button_state[i] && now - button_last_change_ms[i] > DEBOUNCE_MS) {
      button_state[i] = reading;
      button_last_change_ms[i] = now;
      Serial.print('B');
      Serial.print(BUTTON_PINS[i]);
      Serial.print(':');
      Serial.println(reading);
    }
  }
}

void handle_distance_request() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);

  if (duration == 0) {
    Serial.println(-1);  // out of range / no echo
  } else {
    float distance_cm = duration * 0.0343 / 2.0;
    Serial.println(distance_cm);
  }
}

void loop() {
  poll_buttons();

  if (Serial.available() > 0 && Serial.read() == 'D') {
    handle_distance_request();
  }
}
