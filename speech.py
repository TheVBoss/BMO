"""
speech.py — Speech-to-text (Whisper via Groq), text-to-speech (Piper), and
wake word detection (openWakeWord), merged from the former speech_to_text.py,
text_to_speech.py, and wake_word.py.
"""
import io
import subprocess
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from groq import Groq
from piper import PiperVoice
import librosa

from audio_output import play_blocking
from config import (
    api_key, fs, device_fs, silence_threshold, silence_limit, min_speech_chunks,
    mic_alsa_card, mic_mixer_control, tts_voice_path, tts_speed,
    wake_word_threshold, wake_word_chunk_secs, wake_word_cooldown,
    enable_wake_word, is_raspberry_pi, wake_word_models,
    tts_trailing_silence_secs, stt_chunk_secs, speaker_device, mic_device,
    stt_vad_backend, silero_vad_model_path, silero_vad_threshold, silero_min_speech_chunks,
)

client = Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Text-to-speech (Piper)
# ---------------------------------------------------------------------------
voice = PiperVoice.load(tts_voice_path)

# librosa.effects.time_stretch JIT-compiles its numba-accelerated code on
# first use (~2s, GIL-stalling), same issue as the Piper model load. Warm it
# up here at import time instead of on the first real speak() call.
librosa.effects.time_stretch(np.zeros(4096, dtype=np.float32), rate=tts_speed)


def speak(text: str):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    wav_buffer.seek(0)
    with wave.open(wav_buffer, "rb") as wav_file:
        audio      = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        samplerate = wav_file.getframerate()

    # Resample to the device's native rate (it rejects the Piper voice's
    # native rate otherwise), then apply tts_speed via a true time-stretch
    # (librosa, phase vocoder) rather than resampling - this changes tempo
    # without shifting pitch, unlike the old resample-based speed trick.
    audio = resample_poly(audio, up=device_fs, down=samplerate)
    audio_float = (audio / 32768.0).astype(np.float32)

    # time_stretch's STFT/ISTFT can drop part of the final frame right at the
    # array boundary, clipping the last bit of audio. Pad with silence first
    # so the real tail isn't sitting at the edge, then trim back down to the
    # expected stretched length afterward.
    pad = int(device_fs * 0.1)
    expected_len = int(round(len(audio_float) / tts_speed))
    audio_float = librosa.effects.time_stretch(np.pad(audio_float, (0, pad)), rate=tts_speed)[:expected_len]
    audio = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)

    # play_blocking() can return before the output device has actually
    # finished flushing its buffer through the speakers, clipping the last
    # bit of real audio. Append trailing silence so there's always a buffer
    # of "nothing" after the real content for that flush to land in.
    audio = np.concatenate([audio, np.zeros(int(device_fs * tts_trailing_silence_secs), dtype=np.int16)])

    play_blocking(audio)


# ---------------------------------------------------------------------------
# Speech-to-text (Whisper via Groq)
# ---------------------------------------------------------------------------
CHUNK_SECS = stt_chunk_secs  # seconds per recording chunk
DOWNSAMPLE = device_fs // fs
LISTEN_DRAIN_CHUNKS = 3  # ~0.6s at the default CHUNK_SECS - see listen()


def set_mic_volume_max():
    """Force the mic's ALSA capture volume to 100% before listening."""
    try:
        subprocess.run(
            ["amixer", "-c", str(mic_alsa_card), "sset", mic_mixer_control, "Capture", "100%"],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[SpeechToText] Could not set mic volume: {exc}")


def listen():
    # Deferred import - display_change.py imports `voice` from this module
    # at its own top level, so importing display_change up there would be
    # circular. This is only needed once listen() actually runs, by which
    # point display_change is already fully imported anyway.
    from display_change import think

    audio_buffer  = []
    silence_time  = 0.0
    speech_chunks = 0

    set_mic_volume_max()
    print("Listening... speak now")

    with sd.InputStream(samplerate=device_fs, channels=1, dtype="float32", device=mic_device) as stream:
        # Drain stale audio from buffer - without this, listen() picks up
        # the acoustic tail of the "Listening!" prompt that was just spoken
        # (speaker and mic sit right next to each other on the hat), reading
        # it as the very first chunk(s) of real speech. Same fix already
        # applied in wait_for_wake_word() below, for the same reason.
        for _ in range(LISTEN_DRAIN_CHUNKS):
            stream.read(int(device_fs * CHUNK_SECS))

        while True:
            raw_audio, _ = stream.read(int(device_fs * CHUNK_SECS))
            audio = resample_poly(raw_audio.flatten(), up=1, down=DOWNSAMPLE)
            audio_buffer.append(audio)

            level = np.abs(audio).mean()
            print(f"[DEBUG] level={level:.4f}  speech_chunks={speech_chunks}  silence_time={silence_time:.1f}s")

            if level < silence_threshold:
                if speech_chunks >= min_speech_chunks:
                    silence_time += CHUNK_SECS
            else:
                speech_chunks += 1
                silence_time   = 0.0

            if silence_time > silence_limit:
                print("Stopped recording (silence detected)")
                think()
                break

    # Convert to WAV in memory
    audio        = np.concatenate(audio_buffer)
    audio_int16  = (audio * 32767).astype(np.int16)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_int16.tobytes())
    wav_buffer.seek(0)
    wav_buffer.name = "audio.wav"  # Groq requires a filename

    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=wav_buffer,
        response_format="text"
    )
    print(f"[SPEECH TO TEXT]: {transcription}")
    return transcription


# ---------------------------------------------------------------------------
# Speech-to-text (Silero VAD backend) - opt-in alternative stopping logic for
# listen(). See stt_vad_backend in config.py. listen() above is untouched;
# this is a fully separate code path selected by main.py's dispatch.
# ---------------------------------------------------------------------------
_silero_session = None

if stt_vad_backend == "silero":
    try:
        import onnxruntime as ort
        _sess_opts = ort.SessionOptions()
        _sess_opts.inter_op_num_threads = 1
        _sess_opts.intra_op_num_threads = 1
        # Built once here (same rationale as oww_model below - session
        # construction has real setup cost, do it once, not per listen() call).
        _silero_session = ort.InferenceSession(
            silero_vad_model_path, sess_options=_sess_opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        if is_raspberry_pi:
            raise
        else:
            print(f"[SileroVAD] WARNING: could not load Silero VAD ({exc}). "
                  f"listen_silero() will fail if called.")

SILERO_CHUNK_SAMPLES   = 512   # fixed by the model - 16kHz only, do not change
SILERO_CONTEXT_SAMPLES = 64    # fixed by the model - 16kHz context window
SILERO_RAW_CHUNK = SILERO_CHUNK_SAMPLES * DOWNSAMPLE   # samples to read at device_fs per chunk
SILERO_DRAIN_CHUNKS = 19       # ~0.6s at 32ms/chunk - mirrors LISTEN_DRAIN_CHUNKS's ~0.6s


def _silero_infer(chunk_16k, state, context):
    """
    One Silero VAD forward pass on a 512-sample/16kHz chunk. Silero was
    tuned expecting a 64-sample "context" prefix (the trailing samples of
    the previous chunk) in front of each new chunk, plus a small persistent
    recurrent state threaded between calls - see silero-vad's own
    OnnxWrapper for the reference implementation of this contract.
    Returns (speech_probability, new_state, new_context).
    """
    x = np.concatenate([context, chunk_16k]).reshape(1, -1).astype(np.float32)
    sr = np.array(16000, dtype=np.int64)
    out, new_state = _silero_session.run(None, {"input": x, "state": state, "sr": sr})
    new_context = chunk_16k[-SILERO_CONTEXT_SAMPLES:]
    return float(out[0, 0]), new_state, new_context


def _finish_transcription(audio_buffer):
    """
    Shared WAV-assembly + Groq transcription tail, used by listen_silero().
    listen() above keeps its own copy of this same logic untouched, so it
    doesn't need to change at all to support this new backend.
    """
    audio        = np.concatenate(audio_buffer)
    audio_int16  = (audio * 32767).astype(np.int16)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_int16.tobytes())
    wav_buffer.seek(0)
    wav_buffer.name = "audio.wav"  # Groq requires a filename

    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=wav_buffer,
        response_format="text"
    )
    print(f"[SPEECH TO TEXT]: {transcription}")
    return transcription


def listen_silero():
    """
    Same job as listen() - records until the user stops talking, then
    transcribes via Groq Whisper - but decides "stopped talking" using
    Silero VAD's speech-probability score per chunk instead of a plain
    volume threshold. Opt-in via config.stt_vad_backend = "silero".
    """
    if _silero_session is None:
        raise RuntimeError(
            "listen_silero() called but the Silero VAD session isn't loaded "
            "(stt_vad_backend must be \"silero\" at import time)."
        )

    from display_change import think

    audio_buffer  = []
    silence_time  = 0.0
    speech_chunks = 0
    state   = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(SILERO_CONTEXT_SAMPLES, dtype=np.float32)

    set_mic_volume_max()
    print("Listening (Silero VAD)... speak now")

    with sd.InputStream(samplerate=device_fs, channels=1, dtype="float32", device=mic_device) as stream:
        # Drain stale audio from buffer - same reason as listen()'s
        # LISTEN_DRAIN_CHUNKS, just expressed in Silero's smaller chunk size.
        for _ in range(SILERO_DRAIN_CHUNKS):
            stream.read(SILERO_RAW_CHUNK)

        while True:
            raw_audio, _ = stream.read(SILERO_RAW_CHUNK)
            audio = resample_poly(raw_audio.flatten(), up=1, down=DOWNSAMPLE).astype(np.float32)
            audio_buffer.append(audio)

            prob, state, context = _silero_infer(audio, state, context)
            print(f"[DEBUG-Silero] prob={prob:.3f}  speech_chunks={speech_chunks}  silence_time={silence_time:.2f}s")

            if prob < silero_vad_threshold:
                if speech_chunks >= silero_min_speech_chunks:
                    silence_time += SILERO_CHUNK_SAMPLES / fs
            else:
                speech_chunks += 1
                silence_time   = 0.0

            if silence_time > silence_limit:
                print("Stopped recording (Silero VAD: silence detected)")
                think()
                break

    return _finish_transcription(audio_buffer)


# ---------------------------------------------------------------------------
# Wake word detection (openWakeWord)
# ---------------------------------------------------------------------------
_oww_available = False
oww_model = None

if enable_wake_word:
    try:
        import openwakeword
        from openwakeword.model import Model
        # Built once at import time - constructing Model() takes ~0.5s (onnxruntime
        # session setup), which was happening on every single wait_for_wake_word()
        # call and adding to the perceived detection delay. reset() (called per-call
        # below) just clears the prediction buffer - effectively free.
        oww_model = Model(wakeword_model_paths=wake_word_models)
        _oww_available = True
    except ImportError:
        if is_raspberry_pi:
            raise
        else:
            print(
                "[WakeWord] WARNING: openwakeword is not installed.\n"
                "           Wake word detection is disabled for this session.\n"
                "           Install it with: pip install openwakeword"
            )

WAKE_CHUNK = int(fs * wake_word_chunk_secs)
WAKE_RAW_CHUNK = WAKE_CHUNK * DOWNSAMPLE  # samples to read at the mic's native rate per chunk

# Set by the button daemon (actions.py, Triangle button) from its own thread
# to skip wake word detection entirely - either before the next
# wait_for_wake_word() call, or mid-wait if one is already blocked listening
# for it.
_bypass_event = threading.Event()


def request_bypass():
    _bypass_event.set()


def wait_for_wake_word():
    if not enable_wake_word:
        return

    if not _oww_available:
        print("[WakeWord] Skipping wake word (openwakeword not available).")
        return

    if _bypass_event.is_set():
        _bypass_event.clear()
        print("[WakeWord] Bypassed via button.")
        return

    set_mic_volume_max()
    oww_model.reset()  # clear prediction buffer from any previous call, no stale state
    print("Waiting for wake word 'Hey BMO'...")

    with sd.InputStream(samplerate=device_fs, channels=1, dtype="int16", device=mic_device) as stream:
        # Drain stale audio from buffer
        for _ in range(8):
            stream.read(WAKE_RAW_CHUNK)

        while True:
            if _bypass_event.is_set():
                _bypass_event.clear()
                print("[WakeWord] Bypassed via button.")
                return

            raw_audio, _ = stream.read(WAKE_RAW_CHUNK)
            audio = resample_poly(raw_audio.flatten(), up=1, down=DOWNSAMPLE).astype(np.int16)

            predictions = oww_model.predict(audio)

            for wake_word, score in predictions.items():
                if isinstance(score, (int, float, np.floating)) and score > wake_word_threshold:
                    print(f"Wake word detected! ({score:.2f})")
                    time.sleep(wake_word_cooldown)
                    return
