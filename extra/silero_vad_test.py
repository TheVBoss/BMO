"""
Offline verification for the Silero VAD backend (see speech.listen_silero()).
Doesn't need a live mic - checks the model's I/O contract, that pure noise
scores low, and that real speech (synthesized via Piper, the same TTS engine
this project already uses) scores high, all without touching hardware.

Run with: python3 extra/silero_vad_test.py
"""
import io
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
# Force the Silero session to load for this script regardless of what
# config.py's stt_vad_backend is currently set to - this is a standalone
# check of the Silero code path, not a run of the live app.
config.stt_vad_backend = "silero"

import speech  # noqa: E402 - needs the sys.path insert + config override above first


def check_io_contract():
    print("=== 1. Model I/O contract ===")
    sess = speech._silero_session
    assert sess is not None, "Silero session failed to load"
    inputs  = {i.name: i.shape for i in sess.get_inputs()}
    outputs = {o.name: o.shape for o in sess.get_outputs()}
    assert inputs["input"] == [None, None]
    assert inputs["state"] == [2, None, 128]
    assert inputs["sr"] == []
    assert outputs["output"] == [None, 1]
    print("  OK -", inputs, "->", outputs)


def check_silence():
    print("=== 2. Synthetic silence/noise ===")
    rng = np.random.default_rng(0)
    state   = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(speech.SILERO_CONTEXT_SAMPLES, dtype=np.float32)
    probs = []
    for _ in range(50):
        chunk = rng.normal(0, 0.01, speech.SILERO_CHUNK_SAMPLES).astype(np.float32)
        prob, state, context = speech._silero_infer(chunk, state, context)
        probs.append(prob)
    probs = np.array(probs)
    print(f"  max prob over 50 quiet chunks: {probs.max():.4f} (threshold {config.silero_vad_threshold})")
    assert probs.max() < config.silero_vad_threshold, "quiet noise scored above the speech threshold!"
    print("  OK - stayed below threshold")


def _piper_wav(text):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        speech.voice.synthesize_wav(text, wav_file)
    wav_buffer.seek(0)
    with wave.open(wav_buffer, "rb") as wav_file:
        audio      = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        samplerate = wav_file.getframerate()
    return audio, samplerate


def _score_chunks(audio_16k_float):
    """
    Runs Silero over audio_16k_float in fixed 512-sample chunks with proper
    state/context threading (zero-padding the tail to a full chunk).
    Returns an array of per-chunk probabilities.
    """
    n = speech.SILERO_CHUNK_SAMPLES
    pad = (-len(audio_16k_float)) % n
    padded = np.concatenate([audio_16k_float, np.zeros(pad, dtype=np.float32)])
    state   = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(speech.SILERO_CONTEXT_SAMPLES, dtype=np.float32)
    probs = []
    for i in range(0, len(padded), n):
        chunk = padded[i:i + n]
        prob, state, context = speech._silero_infer(chunk, state, context)
        probs.append(prob)
    return np.array(probs)


def check_real_speech():
    print("=== 3. Synthetic speech (Piper TTS) ===")
    audio_int16, samplerate = _piper_wav("Testing one two three, this is Bee-Mo talking normally.")
    lead = np.zeros(int(samplerate * 0.5), dtype=np.int16)
    audio_int16 = np.concatenate([lead, audio_int16, lead])
    audio_16k = resample_poly(audio_int16, up=config.fs, down=samplerate)
    audio_16k_float = (audio_16k / 32768.0).astype(np.float32)

    probs = _score_chunks(audio_16k_float)
    lead_chunks = int(0.4 * config.fs / speech.SILERO_CHUNK_SAMPLES)
    lead_prob   = probs[:lead_chunks].mean()
    speech_prob = probs[lead_chunks:-lead_chunks].max()
    print(f"  lead-in silence avg prob: {lead_prob:.4f}, peak prob during speech: {speech_prob:.4f}")
    assert lead_prob < config.silero_vad_threshold, "lead-in silence scored as speech!"
    assert speech_prob > config.silero_vad_threshold, "synthesized speech never crossed the threshold!"
    print("  OK - silence low, speech high")


def check_chunked_vs_whole_resample():
    print("=== 4. Per-chunk vs whole-file 48k->16k resample ===")
    audio_int16, samplerate = _piper_wav("Checking resample consistency across chunk boundaries.")

    # Shared baseline: upsample the whole file to device_fs (48kHz) once,
    # same as real mic audio arrives at. From here the two paths only differ
    # in how the 48k->16k downsample step is done.
    audio_devicefs = resample_poly(audio_int16, up=config.device_fs, down=samplerate)

    # "Whole": downsample the entire 48kHz signal to 16kHz in one shot.
    audio_16k_whole = resample_poly(audio_devicefs, up=1, down=speech.DOWNSAMPLE)
    probs_whole = _score_chunks((audio_16k_whole / 32768.0).astype(np.float32))

    # "Chunked": slice the same 48kHz signal into listen_silero()'s exact
    # per-read chunk size and resample each piece independently, like the
    # real mic-capture loop does.
    raw_chunk_len = speech.SILERO_RAW_CHUNK
    n_chunks = len(audio_devicefs) // raw_chunk_len
    pieces = [
        resample_poly(audio_devicefs[i * raw_chunk_len:(i + 1) * raw_chunk_len], up=1, down=speech.DOWNSAMPLE)
        for i in range(n_chunks)
    ]
    audio_16k_chunked = np.concatenate(pieces) if pieces else np.array([])
    probs_chunked = _score_chunks((audio_16k_chunked / 32768.0).astype(np.float32))

    n = min(len(probs_whole), len(probs_chunked))
    diff  = np.abs(probs_whole[:n] - probs_chunked[:n])
    flips = np.sum((probs_whole[:n] > config.silero_vad_threshold) != (probs_chunked[:n] > config.silero_vad_threshold))
    print(f"  max abs prob diff: {diff.max():.4f}, classification flips: {flips}/{n}")
    assert flips == 0, "per-chunk resampling flipped speech/silence classification vs whole-file resample!"
    print("  OK - no classification flips")


if __name__ == "__main__":
    check_io_contract()
    check_silence()
    check_real_speech()
    check_chunked_vs_whole_resample()
    print("\nAll offline Silero VAD checks passed.")
