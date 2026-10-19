import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Mirror everything this process prints (including C-level output from
# libraries like ALSA/PortAudio/onnxruntime, which bypass Python's print) to
# debug.log, overwritten fresh on every run, so terminal output can be read
# back after the fact without copy-pasting it.
_debug_log_path = Path(__file__).resolve().with_name("debug.log")
_tee = subprocess.Popen(["tee", str(_debug_log_path)], stdin=subprocess.PIPE)
os.dup2(_tee.stdin.fileno(), sys.stdout.fileno())
os.dup2(_tee.stdin.fileno(), sys.stderr.fileno())
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from actions import shutdown
from ai_brain import ask_ai
from config import name, enable_text_to_speech, enable_speech_to_text, enable_wake_word, enable_display, enable_button_control, startup_loading_hold, is_raspberry_pi, wake_think_pause_secs, stt_vad_backend

if is_raspberry_pi:
    from fusion_hat.battery import Battery
from speech import listen, listen_silero, wait_for_wake_word, request_bypass
from display_change import start, speak_with_display, wait, listen as display_listen, load, preload_speech, think

if is_raspberry_pi:
    # The Fusion Hat's speaker amp is off by default each boot - enable it
    # before anything tries to play audio (TTS, camera shutter sound, etc).
    try:
        subprocess.run(["fusion_hat", "speaker", "enable"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[MAIN] Could not enable Fusion Hat speaker: {e}")

greeting = "Bee-p. Boo-p. Hi, I'm Bee-Mo, I can drive, I can talk, I can help. What do you need?"

if enable_display:
    start()
    load()  # show loading animation - held for at least startup_loading_hold seconds, see below
    if enable_text_to_speech:
        # Synthesize + lip-sync-analyze the greeting now, in the background,
        # so the loading hold below covers that cost - speak_with_display()
        # picks up the cached result instead of redoing it right before
        # actually speaking.
        preload_speech(greeting)
    time.sleep(startup_loading_hold)

if enable_button_control and is_raspberry_pi:
    # The button daemon (actions.py's main(), see extra/bmo-buttons.service)
    # runs as its own always-on process, not a thread in here - it owns the
    # Arduino's serial port so mouse/click buttons work even when main.py
    # isn't running. It reaches this process via OS signals: SIGTERM
    # (LargeCircle) for a graceful shutdown, SIGUSR1 (Triangle) to bypass
    # wake word detection. The pidfile is how it finds this process.
    # SIGUSR1 doesn't exist on Windows, so this whole block is Pi-only.
    Path(__file__).resolve().with_name("main.pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: shutdown())
    signal.signal(signal.SIGUSR1, lambda *_: request_bypass())

print("[MAIN] Robot AI started. Type something.")

if is_raspberry_pi:
    battery = Battery()
    print(f"[MAIN] Battery: {battery.capacity}%")

print(greeting)
if enable_text_to_speech:
    speak_with_display(greeting)

WAKE_PHRASE = f"hey {name}".lower()

while True:
    wait()
    if enable_speech_to_text:
        wait_for_wake_word()
        think()
        time.sleep(wake_think_pause_secs)
        # speak_with_display (not display_listen() + speak()) so the
        # listening overlay turns on at the exact instant "Listening!" is
        # actually said - see talk()'s keep_listening handling - rather
        # than the overlay appearing early while it's still being
        # synthesized.
        if enable_text_to_speech:
            speak_with_display("Listening!", keep_listening=True)
        else:
            display_listen()
        user_input = listen_silero() if stt_vad_backend == "silero" else listen()
    elif enable_wake_word:
        # No microphone (enable_speech_to_text=False), but wake word is still
        # wanted - type the wake phrase instead of saying it, then the next
        # line typed is treated as the actual command, same as the mic flow.
        typed = input(f"Type 'Hey {name}' to wake: ")
        if typed.strip().lower() != WAKE_PHRASE:
            continue
        think()
        time.sleep(wake_think_pause_secs)
        if enable_text_to_speech:
            speak_with_display("Listening!", keep_listening=True)
        else:
            display_listen()
        user_input = input("Prompt: ")
    else:
        user_input = input("Prompt: ")

    think()  # gap between the user's input and the AI actually responding
    response = ask_ai(user_input)  # Driving logic is triggered inside ask_ai if an ACTION is detected

    if response:
        print(f"{name}: {response}")
        if enable_text_to_speech:
            speak_with_display(response)