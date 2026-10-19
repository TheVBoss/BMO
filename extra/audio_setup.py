"""
Interactive audio setup menu: test/assign the speaker and mic, and measure
a silence_threshold from real ambient noise. Lists every audio device
sounddevice/PortAudio can see (including the ALSA plugin variants defined in
/etc/asound.conf, e.g. "hat", "speaker", "speaker_hw", "mic"). Assigning a
speaker/mic or saving a threshold writes straight into config.py.

Run with: python3 extra/audio_setup.py
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from config import device_fs, fs, stt_chunk_secs

if config.is_raspberry_pi:
    # The Fusion Hat's speaker amp is off by default each boot - enable it
    # before testing playback below, or every speaker test would be silent.
    try:
        subprocess.run(["fusion_hat", "speaker", "enable"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Could not enable Fusion Hat speaker: {e}")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.py"
DOWNSAMPLE = device_fs // fs
SILENCE_FINDER_GET_READY_SECS = 3
SILENCE_FINDER_QUIET_SECS     = 7
SILENCE_FINDER_BREAK_SECS     = 3
SILENCE_FINDER_SPEECH_SECS    = 7
MEASURE_DRAIN_SECS            = 2.0  # settling time to discard when a capture stream first opens


def _list_devices(channel_key):
    devices = sd.query_devices()
    return [d["name"] for d in devices if d[channel_key] > 0]


def _print_devices(names):
    for n, name in enumerate(names, start=1):
        print(f"  {n}. {name}")


def _select_device(names, label):
    while True:
        print(f"\n{label} devices:")
        _print_devices(names)
        choice = input(f"Pick a {label.lower()} by number (or 'c' to cancel): ").strip().lower()
        if choice == "c":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Invalid choice, try again.")


def _test_output(name):
    t = np.arange(int(device_fs * 1.5)) / device_fs
    tone = (0.25 * np.sin(2 * np.pi * 440 * t) * np.iinfo(np.int16).max).astype(np.int16)
    print(f"Playing a test tone through '{name}'...")
    sd.play(tone, device_fs, device=name)
    sd.wait()


def _test_input(name, playback_device):
    print(f"Recording 3 seconds from '{name}' - say something...")
    rec = sd.rec(int(device_fs * 3), samplerate=device_fs, channels=1, dtype="int16", device=name)
    sd.wait()
    print(f"Playing it back through '{playback_device}'...")
    sd.play(rec, device_fs, device=playback_device)
    sd.wait()


def _countdown(seconds, message):
    print(message)
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1)


def _measure_levels(mic_name, seconds):
    """
    Records from mic_name for `seconds` and returns the same "level" per
    chunk that speech.listen() computes (mean abs of the float32 mic
    signal, downsampled from device_fs to fs).
    """
    chunk_frames = int(device_fs * stt_chunk_secs)
    n_chunks = int(seconds / stt_chunk_secs)
    levels = []
    with sd.InputStream(samplerate=device_fs, channels=1, dtype="float32", device=mic_name) as stream:
        # Drain stale audio - freshly opening a capture stream on this
        # hardware produces a loud settling artifact for a couple of
        # seconds, which would otherwise get counted as real ambient/speech
        # data (same fix as speech.listen()'s LISTEN_DRAIN_CHUNKS).
        for _ in range(int(round(MEASURE_DRAIN_SECS / stt_chunk_secs))):
            stream.read(chunk_frames)

        for i in range(n_chunks):
            raw_audio, _ = stream.read(chunk_frames)
            audio = resample_poly(raw_audio.flatten(), up=1, down=DOWNSAMPLE)
            level = np.abs(audio).mean()
            levels.append(level)
            remaining = seconds - i * stt_chunk_secs
            print(f"\r  {remaining:4.1f}s left - level: {level:.5f}", end="", flush=True)
    print()
    return np.array(levels)


def _drop_anomalies(levels):
    """Drops values outside the standard Q1/Q3 +- 1.5*IQR range."""
    q1, q3 = np.percentile(levels, [25, 75])
    iqr = q3 - q1
    clean = levels[(levels >= q1 - 1.5 * iqr) & (levels <= q3 + 1.5 * iqr)]
    return clean if len(clean) else levels


def _find_silence_threshold(mic_name):
    """
    Measures real mic levels in two phases - quiet, then you talking -
    drops anomalous chunks (IQR outliers - a click, an echo, a breathing
    gap) from each, and picks silence_threshold as the midpoint between the
    two phases' averages.
    """
    _countdown(SILENCE_FINDER_GET_READY_SECS, f"\nGet ready - stay quiet and stop typing on '{mic_name}'...")
    print(f"Measuring ambient noise for {SILENCE_FINDER_QUIET_SECS}s...")
    quiet_avg = float(_drop_anomalies(_measure_levels(mic_name, SILENCE_FINDER_QUIET_SECS)).mean())
    print(f"Quiet average: {quiet_avg:.5f}")

    _countdown(SILENCE_FINDER_BREAK_SECS, "\nNow get ready to talk normally, like you would to BMO...")
    print(f"Recording your voice for {SILENCE_FINDER_SPEECH_SECS}s - talk now...")
    speech_avg = float(_drop_anomalies(_measure_levels(mic_name, SILENCE_FINDER_SPEECH_SECS)).mean())
    print(f"Speech average: {speech_avg:.5f}")

    threshold = (quiet_avg + speech_avg) / 2
    print(f"\nDone. Quiet average: {quiet_avg:.5f}, speech average: {speech_avg:.5f} "
          f"-> threshold (midpoint): {threshold:.5f}")
    return threshold


def _write_config(key, value):
    text = CONFIG_PATH.read_text()
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"couldn't find a '{key} =' line in config.py to update")
    CONFIG_PATH.write_text(pattern.sub(f"{key} = {json.dumps(value)}", text, count=1))


def _print_menu(speaker, mic, threshold):
    print("\n=== Audio Setup ===")
    print(f"[1] Speaker Test              (current: {speaker})")
    print("[2] Assign Speaker")
    print(f"[3] Mic Test                  (current: {mic})")
    print("[4] Assign Mic")
    print(f"[5] Silence Threshold Finder  (current: {threshold})")
    print("[q] Quit")


def main():
    outputs = _list_devices("max_output_channels")
    inputs = _list_devices("max_input_channels")
    if not outputs:
        sys.exit("No output devices found.")
    if not inputs:
        sys.exit("No input devices found.")

    speaker   = getattr(config, "speaker_device", None) or outputs[0]
    mic       = getattr(config, "mic_device", None) or inputs[0]
    threshold = getattr(config, "silence_threshold", None)

    while True:
        _print_menu(speaker, mic, threshold)
        choice = input("Choose an option: ").strip().lower()

        if choice == "1":
            try:
                _test_output(speaker)
            except Exception as e:
                print(f"Failed to play through '{speaker}': {e}")

        elif choice == "2":
            picked = _select_device(outputs, "Speaker")
            if picked:
                speaker = picked
                _write_config("speaker_device", speaker)
                print(f"Saved speaker_device = {speaker!r}")

        elif choice == "3":
            try:
                _test_input(mic, speaker)
            except Exception as e:
                print(f"Failed to record from '{mic}': {e}")

        elif choice == "4":
            picked = _select_device(inputs, "Microphone")
            if picked:
                mic = picked
                _write_config("mic_device", mic)
                print(f"Saved mic_device = {mic!r}")

        elif choice == "5":
            try:
                threshold = round(float(_find_silence_threshold(mic)), 5)
            except Exception as e:
                print(f"Failed to record from '{mic}': {e}")
                continue
            if input(f"Save silence_threshold = {threshold} to config.py? [y/N]: ").strip().lower() == "y":
                _write_config("silence_threshold", threshold)
                print("Saved.")

        elif choice == "q":
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
