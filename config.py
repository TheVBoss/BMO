# ============================================================
# config.py — Central configuration for the BMO Robot
# ============================================================
# The QUICK SETTINGS section has the things you're most likely to change.
# Hardware and tuning details live further down under ADVANCED SETTINGS,
# grouped by the same sections.
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# QUICK SETTINGS
# ============================================================

# ------------------------------------------------------------
# Platform Selection
# ------------------------------------------------------------
# Set to True when running on a Raspberry Pi 5.
# Set to False for Windows/macOS development.
#
# Effects when False (Windows):
#   - Camera always uses the test image (libcamera not available)
#   - GPIO motor control is replaced with print stubs
#   - Wake word import fails gracefully if openwakeword isn't installed
is_raspberry_pi = True


# ------------------------------------------------------------
# Robot Identity
# ------------------------------------------------------------
# The robot's name. Changing this also changes the AI persona.
# Note: if you change the name you will also need to train your own
# openwakeword model to match — the built-in "hey_jarvis" model
# won't respond to a different name.
name = "BMO"


# ------------------------------------------------------------
# Feature Toggles
# ------------------------------------------------------------
# Set to True to enable microphone input via Whisper/Groq
enable_speech_to_text = True

# Which voice-activity-detection backend decides when the user has stopped
# talking, during listen():
#   "energy" (default) - plain RMS/volume threshold (silence_threshold below).
#                         Unchanged legacy behavior.
#   "silero"            - Silero VAD, a small neural model that scores actual
#                         speech-likelihood per chunk instead of just volume.
#                         Opt-in; see the Silero VAD section further down for
#                         tuning knobs. Requires audio/silero_vad.onnx (vendored).
stt_vad_backend = "silero"

# Set to True to enable Piper TTS audio output
enable_text_to_speech = True    

# Set to True to enable wake word detection
enable_wake_word = True

# Set to True to let main.py respond to the button daemon's signals
# (actions.py, run via extra/bmo-buttons.service). The daemon itself always
# runs independently — this only controls whether main.py listens for it.
enable_button_control = True

# Set to True to capture from the Pi camera.
# NOTE: Only works when is_raspberry_pi = True.
# On Windows this is ignored — the test image is always used.
use_camera = False

# Set to True to enable the pygame display (faces, animations).
# Set to False to run headless (no screen required).
enable_display = True

# Set to True to announce each queued command in detail before running them
# (e.g. "driving forward for 200 millimeters, then turning left for 90
# degrees..."). Set to False for a generic "Running the full sequence now!"
# instead.
announce_command_details = False


# ------------------------------------------------------------
# Command Limits
# ------------------------------------------------------------
# Python enforces these — the AI is told not to check them itself.
# Change them to suit your space and safety requirements.

# Maximum travel distance per drive command (mm)
max_drive_distance_mm     = 2000

# Maximum turn angle per command (degrees; 720 = two full rotations)
max_turn_angle_deg        = 720

# Maximum duration for time-based drive/turn commands (seconds)
max_command_duration_secs = 20

# Minimum obstacle clearance for stop-at commands (mm)
# Values below this risk hitting the obstacle before the robot stops.
min_stop_at_distance_mm   = 100

# Emergency stop threshold (mm). DrivePID and DriveTime monitor the ultrasonic
# sensor while driving forward and cut motors instantly if anything comes within
# this distance. Also used as a pre-drive abort if already too close.
safety_stop_distance_mm   = 30


# ------------------------------------------------------------
# AI / LLM
# ------------------------------------------------------------
# Where you are - given to the AI as context (e.g. for weather/time/local
# search questions asked without specifying a place).
user_location = "Auckland, New Zealand"

# Max tokens for the AI response (higher = longer replies, more latency)
max_tokens = 150

# Groq model used for chat responses
llm_model = "llama-3.3-70b-versatile"

# Free backup model, auto-switched to for the rest of the session once
# llm_model's daily quota gets rate-limited (see ai_brain.ask_ai). Groq's
# free-tier limits are per-model - llama-3.1-8b-instant has a much bigger
# free daily allowance (500K tokens/14.4K requests vs llm_model's
# 100K/1K), so it's a reasonable same-account fallback rather than the bot
# just going silent for the rest of the day.
llm_fallback_model = "llama-3.1-8b-instant"


# ------------------------------------------------------------
# Text-to-Speech (Piper)
# ------------------------------------------------------------
# Path to the ONNX voice model file
tts_voice_path = "audio/bmo/bmo.onnx"

# Playback speed multiplier (1.0 = normal speed)
tts_speed = 0.99


# ------------------------------------------------------------
# Wake Word
# ------------------------------------------------------------
# Wake word model(s) to load. Built-in options:
#   "alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy", "ok_nabu"
# Or provide a path to a custom .onnx model file.
wake_word_models = ["audio/wakeword.onnx"]

# Confidence threshold for wake word detection (0.0 – 1.0)
wake_word_threshold = 0.3


# ------------------------------------------------------------
# Camera / Photo
# ------------------------------------------------------------
# Test image filename (relative to the images/ folder).
# Used when is_raspberry_pi = False, or when use_camera = False on Pi.
test_image = "dog.jpg"


# ============================================================
# ADVANCED SETTINGS
# ============================================================

# ------------------------------------------------------------
# Groq API Key
# ------------------------------------------------------------
# Set GROQ_API_KEY in a .env file (same folder as this file) or as an
# environment variable. Get a key from https://console.groq.com
# DO NOT share or commit your key to version control.
api_key = os.environ["GROQ_API_KEY"]


# ------------------------------------------------------------
# AI / LLM (advanced)
# ------------------------------------------------------------
# Number of retry attempts if the Groq API call fails
llm_retry_attempts = 20

# Seconds to wait between retries on API failure
llm_retry_delay = 10

# Groq vision model used for photo analysis
vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"

# Max tokens for the vision/photo response
vision_max_tokens = 150


# ------------------------------------------------------------
# Text-to-Speech (advanced)
# ------------------------------------------------------------
# Seconds of silence appended after each spoken phrase to prevent audio
# cutoff — the output device buffer flush can clip the last few milliseconds
# of real audio otherwise.
tts_trailing_silence_secs = 0.3


# ------------------------------------------------------------
# Speech-to-Text (Whisper / Groq)
# ------------------------------------------------------------
# Sample rate — 16000 Hz is optimal for Whisper/openwakeword
fs = 16000

# The Google VoiceHAT sound card only supports 48000 Hz natively (confirmed
# via sd.check_input_settings - 16000/22050/44100 all fail with
# "Invalid sample rate"). Mic capture happens at this rate and gets
# downsampled to `fs` before being handed to the model/Whisper.
device_fs = 48000

# ALSA card index + mixer control name for the VoiceHAT mic, used to force
# capture volume to max before listening (see speech.set_mic_volume_max).
mic_alsa_card    = 2
mic_mixer_control = "Fusion Hat"

# sounddevice/PortAudio's own "default output device" heuristic ignores
# ALSA's `pcm.!default` (set to "hat" in /etc/asound.conf, routing through
# the dmix + softvol chain to the real VoiceHAT card) and instead picks the
# first raw hw playback device it enumerates - the Pi's HDMI output - which
# then fails with "Sample format not supported" for speech audio. Naming the
# device explicitly on every sd.play() call sidesteps that wrong default.
speaker_device = "hat"

# Same idea as speaker_device, but for capture - names the sounddevice input
# device explicitly instead of relying on PortAudio's default guess. None
# means "use the default" (see audio_setup.py to pick and test one).
mic_device = "mic"

# RMS silence detection threshold.
# Raise if recording won't stop; lower if it cuts off too early.
silence_threshold = 0.05

# Seconds of silence before recording stops
silence_limit = 1.5

# Number of speech chunks (see stt_chunk_secs) required before silence
# detection activates — prevents cutting off before the user has started.
min_speech_chunks = 1

# Duration of each recorded audio chunk in seconds
stt_chunk_secs = 0.2


# ------------------------------------------------------------
# Speech-to-Text — Silero VAD (advanced)
# ------------------------------------------------------------
# Only used when stt_vad_backend = "silero" above.

# Path to the vendored Silero VAD ONNX model.
silero_vad_model_path = "audio/silero_vad.onnx"

# Speech-probability threshold (0.0-1.0). Chunks scoring above this are
# treated as speech, at/below as silence. Silero's own docs call 0.5 a good
# default for most datasets - tune from there if needed.
silero_vad_threshold = 0.5

# Number of Silero chunks (each a fixed 512 samples / 32ms at 16kHz - not
# configurable, the ONNX model hard-requires exactly 512 samples per call)
# required before silence detection activates - same idea as
# min_speech_chunks above, but in Silero's much smaller chunk granularity.
# 6 chunks ~= 192ms, roughly matching min_speech_chunks=1 at the energy
# backend's default stt_chunk_secs (200ms).
silero_min_speech_chunks = 6


# ------------------------------------------------------------
# Wake Word (advanced)
# ------------------------------------------------------------
# Audio chunk size in seconds fed to the wake word model (80 ms recommended)
wake_word_chunk_secs = 0.08

# Seconds to pause after detection so the audio tail can clear
wake_word_cooldown = 0.5


# ------------------------------------------------------------
# Camera / Photo (advanced)
# ------------------------------------------------------------
# Filename of the camera shutter sound, in the audio/ folder.
camera_sound = "camera.mp3"


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------
# Screen resolution used only for the windowed dev preview (Windows/macOS,
# is_raspberry_pi = False). On the Pi, fullscreen mode auto-detects the real
# screen's resolution instead — these two values are never used there.
display_width  = 800
display_height = 480

# Fill colour (hex) shown in any gaps around the face image - e.g. if the
# auto-detected screen's aspect ratio doesn't match the image's, the image is
# centered and scaled to fit without stretching, leaving bars on two sides.
display_bg_color = "#bbfdc9"

# Shifts the whole face (mouth, eyes, static images, listening overlay) away
# from dead-center, in pixels. Positive x moves it right, positive y moves it
# down; negative moves left/up. The background stays filled with
# display_bg_color across the whole screen - only the face itself moves.
face_offset_x = 59
face_offset_y = 53

# Seconds between switching images in the loading animation
display_loading_cycle_interval = 0.3

# Minimum seconds to show the loading animation at startup, regardless of
# how fast everything else loads. Purely cosmetic — without this the loading
# screen can flash by almost instantly on fast machines.
startup_loading_hold = 1.5

# Seconds to pause after a wake trigger so the thinking animation is visible
# before "Listening!" is announced.
wake_think_pause_secs = 0.5

# Pygame display refresh rate (frames per second). Verified this Pi 5 can
# sustain 60fps fullscreen with vsync on comfortably - lower if the real
# panel/hardware struggles.
display_fps = 60

# Idle eye animation timings (seconds per step)
eye_open_secs     = 2.0   # how long each "open" or "open2" step is held
eye_closed_secs   = 0.35  # how long each "closed" step flanking a blink is held
eye_blink_secs    = 0.2   # how long the blink flash (Eyes_Blinked) is shown
eye_thinking_secs = 0.25  # how long each thinking/thinking2 flicker step is held


# ------------------------------------------------------------
# Display Driver (Pi 5)
# ------------------------------------------------------------
# DRI device for kmsdrm on Pi 5 (card2 = vc4-drm display controller)
dri_device = "/dev/dri/card2"


# ------------------------------------------------------------
# Motor Configuration
# ------------------------------------------------------------
# Maps each physical wheel position to its Fusion Hat motor ID (M0-M3) and
# whether that motor needs to be flipped so that a positive power always
# means "this wheel spins forward". Figured out with extra/motor_setup.py.
#
# Left/Right are as if you're standing behind the robot, looking the same
# way it faces (toward its front) - not standing in front of it looking
# back. That matches the robot's own left/right, same as a car.
#
# If a wheel spins backwards when it should go forward, flip its "reversed"
# value here (or use extra/motor_setup.py's Toggle option).
MOTORS = {
    "LeftFront":  {"id": "M2", "reversed": True},
    "LeftBack":   {"id": "M0", "reversed": False},
    "RightFront": {"id": "M1", "reversed": True},
    "RightBack":  {"id": "M3", "reversed": False},
}

# GPIO chip the encoders are wired to.
ENCODER_GPIO_CHIP = "/dev/gpiochip4"

# Maps each wheel position to the GPIO pin its encoder is wired to. Figured
# out with extra/motor_setup.py.
ENCODER_PINS = {
    "LeftFront":  22,
    "LeftBack":   27,
    "RightFront": 4,
    "RightBack":  4,
}

# Wheel and encoder hardware constants - shared by every PID drive function
# (DrivePID, TurnPID, StopPID) since they describe the hardware, not a
# specific maneuver.
WHEEL_DIAMETER_MM = 55
ENCODER_PPR = 11           # Basic pulses per revolution from encoder
GEAR_RATIO = 21.3          # Motor gear reduction ratio

# Turning geometry, used by TurnPID. ROBOT_WIDTH_MM is the left/right wheel
# center-to-center track width; WHEEL_WIDTH_MM is the tire contact-patch
# width. Subtracting it approximates the effective pivot circle the wheels
# actually trace on the ground (rather than the wheels' centerline circle).
ROBOT_WIDTH_MM = 240.5
WHEEL_WIDTH_MM = 6

# Fixed control-loop period. Pulses are counted continuously, but the PID
# update only runs every CONTROL_DT seconds so the math sees a roughly
# constant timestep instead of a wildly variable per-edge one.
CONTROL_DT = 0.02  # 50 Hz

# Maximum motor power (0–100) used for all PID drive/turn/stop commands.
motor_max_power = 60

# DriveTime/TurnTime have no separate ramp-rate setting - power always
# ramps linearly from 0 up to motor_max_power at exactly the midpoint of
# whatever duration is requested, then back down to 0, a pure triangle
# scaled to the command's own length rather than a fixed acceleration rate.


# ------------------------------------------------------------
# Motor PID Gains
# ------------------------------------------------------------
# DrivePID gains - driving in a straight line for a target distance.
DRIVE_PID = {
    # KP_DIST switches between two fixed values depending on how long the
    # requested drive is - a single KP_DIST tuned for short, precise moves
    # was too gentle to get a long drive up to speed promptly, and one tuned
    # for long drives overshot/oscillated on short ones.
    "KP_DIST_THRESHOLD_MM": 1000,  # drives longer than this use KP_DIST_LONG, otherwise KP_DIST_SHORT
    "KP_DIST_LONG": 0.3,           # Proportional gain on distance error -> power, for drives > KP_DIST_THRESHOLD_MM
    "KP_DIST_SHORT": 0.7,          # Proportional gain on distance error -> power, for drives <= KP_DIST_THRESHOLD_MM
    "KI_DIST": 0.3,            # Integral gain on distance error
    "KD_DIST": 0.43,           # Derivative gain on distance error
    "D_FILTER_ALPHA": 0.3,     # Derivative low-pass filter coefficient (0..1, lower = more smoothing)
    "KP_STEER": 2.6,           # Left/right steering correction (pulse difference between sides)
    "KP_BALANCE": 0.3,         # Front/back balance correction (pulse difference within a side)
    "ACCEL_PER_SEC_FACTOR": 3.0,  # accel_rate = max_power / this (seconds to ramp 0->max_power)
}

# TurnPID gains - pivoting in place to a target angle.
TURN_PID = {
    "KP_TURN": 1.0,            # Proportional gain on angle error -> power
    "KI_TURN": 0.3,            # Integral gain on angle error
    "KD_TURN": 0.3,            # Derivative gain on angle error
    "D_FILTER_ALPHA": 0.3,     # Derivative low-pass filter coefficient (0..1, lower = more smoothing)
    "KP_SYNC": 2.6,            # Keeps both sides' arc length in sync during the turn
    "KP_BALANCE": 0.3,         # Front/back balance correction (pulse difference within a side)
    "ACCEL_PER_SEC_FACTOR": 3.0,  # accel_rate = max_power / this (seconds to ramp 0->max_power)
}

# StopPID no longer runs its own PID loop against the ultrasonic sensor (that
# approach needed constant tuning to avoid sensor-noise-driven oscillation/
# overshoot). It now just takes one ultrasonic reading and hands off to the
# already-tuned DrivePID above, so there are no StopPID-specific gains left
# to configure - just the re-scan/correction settings below.

# After the initial DrivePID, StopPID re-scans and runs another DrivePID
# correction if it's still off target by more than this many mm - the
# correction distance can be negative (drives in reverse) if it overshot.
STOP_PID_CORRECTION_TOLERANCE_MM = 50

# Safety cap on correction passes, in case the ultrasonic reading or the
# drive itself never converges within tolerance.
STOP_PID_MAX_CORRECTIONS = 5


# ------------------------------------------------------------
# Ultrasonic Sensor Settings
# ------------------------------------------------------------
# HC-SR04 via an Arduino running extra/ultrasonic_sensor/ultrasonic_sensor.ino.
ULTRASONIC_PORT = "/dev/ttyACM0"
ULTRASONIC_BAUD = 9600
ULTRASONIC_SAMPLES = 5   # readings attempted per distance check

# Arduino digital pins the HC-SR04 is wired to. Documentation only - these
# aren't read by the Python side (the Pi just sends 'D' and reads back a
# distance over serial); change them here and in ultrasonic_sensor.ino
# together if the sensor is ever rewired.
TRIG_PIN = 9
ECHO_PIN = 10


# ------------------------------------------------------------
# Button Settings
# ------------------------------------------------------------
# 7 ground-based momentary buttons wired directly to digital pins 2-8 on
# the same Arduino Micro that runs the ultrasonic sketch (see
# ULTRASONIC_PORT above). Each pin is configured INPUT_PULLUP in
# firmware, so it reads HIGH at rest and LOW while the button is pressed
# (the other leg of each button just goes to GND).
#
# Firmware reports presses/releases the instant they happen (not on
# request, unlike the ultrasonic sensor) as lines like "B2:0" (pin 2 just
# went LOW / pressed) or "B2:1" (released) - see
# extra/ultrasonic_sensor/ultrasonic_sensor.ino. Wire up / re-map buttons
# interactively with extra/button_setup.py (press each button when
# prompted - it auto-detects pins and writes BUTTON_PINS below for you).
BUTTON_PORT = ULTRASONIC_PORT
BUTTON_BAUD = ULTRASONIC_BAUD

# Maps each button's logical name to the Arduino digital pin it's wired to.
BUTTON_PINS = {
    "Triangle":    2,
    "LargeCircle": 7,
    "SmallCircle": 3,
    "UpArrow":     5,
    "LeftArrow":   8,
    "RightArrow":  6,
    "DownArrow":   4,
}

# Mouse cursor movement settings (used by the button daemon in actions.py).
mouse_step_px    = 6     # pixels moved per repeat tick while a direction button is held
mouse_repeat_secs = 0.03  # seconds between repeat ticks (~33 ticks/sec)
