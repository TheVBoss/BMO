"""
Shared persistent audio output stream - opened once here and reused for
every phrase, instead of each caller doing its own sd.play() (which fully
opens and tears down the ALSA/dmix connection every single call). Repeated
stream open/close under CPU load - the render loop, motor control, and ML
inference all sharing this one process - is a real, distinct source of
occasional playback glitches, separate from the mouth-sync timing tuned in
display_change.talk() (which this doesn't touch - no buffering/latency
change, just avoiding repeated stream teardown).
"""
import threading

import numpy as np
import sounddevice as sd

from config import device_fs, speaker_device

_lock = threading.Lock()
_stream = sd.OutputStream(samplerate=device_fs, device=speaker_device, channels=1, dtype="int16")
_stream.start()


def play_blocking(audio: np.ndarray):
    """
    Writes int16 mono audio to the persistent stream and blocks until it's
    been handed to the device - same calling convention as the
    sd.play(audio, ...); sd.wait() pattern this replaces.
    """
    with _lock:
        _stream.write(np.ascontiguousarray(audio, dtype=np.int16))
