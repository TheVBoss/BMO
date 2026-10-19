"""
display_change.py — BMO Robot Display Controller
=================================================
Controls the DSI LCD screen for the BMO robot project. Works on both
Windows (testing) and Raspberry Pi 5 (deployment). Merged from the former
display_change.py — the BMO mouth+eyes lip-sync engine renders into the
single shared window (merged from the former talking_face.py).

USAGE:
------
1. Drop this file into the root of your robot project (same folder as config.py).
2. Import and use in your project:

    from display_change import start, show, talk, speak_with_display

    start()                          # Open the display window (non-blocking)
    show("photo")                    # Show a static face by key or filename
    talk(audio_array, samplerate)    # Animate the BMO mouth/eyes while speaking
    stop()                           # Close the display

TALKING / WAITING / LISTENING:
-------------------------------
wait(), listen(), and talk() all render the BMO mouth+eyes face (idle blink
cycle + lip-synced mouth, driven by rhubarb when available) instead of
swapping between static images. listen() additionally overlays the
"listening" graphic on top. The other states below (load, show) still use
the static images in FACE_MAP.

Mouth (lip-sync): talk()/speak_with_display() drive the mouth from real
Rhubarb Lip Sync cues (generate manually with:
rhubarb -f json -o out.json speech.wav) when the `rhubarb` binary is
available (see RHUBARB_PATH/_find_rhubarb()), falling back to a rough
RMS-amplitude approximation otherwise. Idle/rest mouth is "A".

Eyes (independent of audio), one background thread:
    - Idle cycle (IDLE_EYE_STEPS): Eyes_Open -> Eyes_Closed -> Eyes_Blinked
      (quick flash) -> Eyes_Closed -> Eyes_Open2 -> Eyes_Closed ->
      Eyes_Blinked (quick flash) -> Eyes_Closed -> (repeat). The blink is
      baked into the cycle itself rather than a separate random timer.
    set_expression("happy") pauses the idle cycle and holds Eyes_Closed
    persistently instead. Eyes_Closed2.png isn't wired up to anything yet -
    free for a future expression.

CYCLE ANIMATIONS (interruptible — stop instantly when anything else runs):
------------------
    load()     — cycles: loading    → loading2    → loading3 → ...

INTEGRATION WITH speech.py:
-------------------------------------
    from display_change import start, speak_with_display as speak
    start()
    speak("Hello!")   # synthesises, plays, and animates all in one call

CONFIG SETTINGS (in config.py):
---------------------------------
    display_width           = 800    # match your DSI screen (4.3" = 800x480)
    display_height          = 480
    dri_device              = "/dev/dri/card2"  # Pi 5 display controller

PI 5 NOTES:
-----------
- Pi 5 dropped the legacy framebuffer (fbdev). This file auto-detects
  whether a desktop session is running and uses kmsdrm if not.
- Pi 5 has no 3.5mm jack — use USB audio, HDMI audio, or a speaker HAT.
- The DSI connector on Pi 5 is smaller — you need the Pi 5 ribbon adapter.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Config import
# ---------------------------------------------------------------------------
# Each config setting falls back to a sane default if missing from config.py,
# so this module works standalone even with a partial/older config file.
import config

# Each setting reads from config.py, falling back to a sane default so this
# module works standalone even with a partial/older config file.
enable_display         = getattr(config, "enable_display",                 True)
tts_speed              = getattr(config, "tts_speed",                      1.0)
device_fs              = getattr(config, "device_fs",                      48000)
speaker_device          = getattr(config, "speaker_device",                 None)
display_width          = getattr(config, "display_width",                  800)
display_height         = getattr(config, "display_height",                 480)
_bg_color_hex          = getattr(config, "display_bg_color",               "#bbfdc9").lstrip("#")
display_bg_color       = tuple(int(_bg_color_hex[i:i + 2], 16) for i in (0, 2, 4))
face_offset_x          = getattr(config, "face_offset_x",                  0)
face_offset_y          = getattr(config, "face_offset_y",                  0)
LOADING_CYCLE_INTERVAL = getattr(config, "display_loading_cycle_interval", 0.3)
DISPLAY_FPS            = getattr(config, "display_fps",                    30)
TRAILING_SILENCE_SECS  = getattr(config, "tts_trailing_silence_secs",      0.3)
dri_device             = getattr(config, "dri_device",                     "/dev/dri/card2")
IDLE_STEP_NORMAL           = getattr(config, "eye_open_secs",     2.0)
IDLE_STEP_CLOSED           = getattr(config, "eye_closed_secs",   0.35)
IDLE_STEP_QUICK            = getattr(config, "eye_blink_secs",    0.2)
IDLE_STEP_THINKING_FLICKER = getattr(config, "eye_thinking_secs", 0.25)

# Fullscreen on the Pi (real DSI screen), windowed everywhere else (desktop dev).
display_fullscreen = getattr(config, "is_raspberry_pi", sys.platform == "linux")

# ---------------------------------------------------------------------------
# Piper TTS voice — reuse the same instance speech.py already loads eagerly
# at import time, rather than loading a second copy of the ONNX model here.
# ---------------------------------------------------------------------------
try:
    from speech import voice as _piper_voice
except ImportError as e:
    _piper_voice = None
    print(f"[Display] TTS not available: {e}")


def _find_rhubarb():
    """
    Look for the rhubarb binary on PATH first; if not found, fall back to a
    local extract under tools/ (see README - download from
    https://github.com/DanielSWolf/rhubarb-lip-sync/releases and unzip into
    tools/ without needing to touch PATH). Returns None if neither exists.
    """
    on_path = shutil.which("rhubarb")
    if on_path:
        return on_path
    exe_name = "rhubarb.exe" if sys.platform == "win32" else "rhubarb"
    candidates = glob.glob(os.path.join(os.path.dirname(__file__), "tools", "Rhubarb-Lip-Sync-*", exe_name))
    return candidates[0] if candidates else None


RHUBARB_PATH = _find_rhubarb()  # None if not installed - see _drive_mouth()/_compute_mouth_cues()

# ---------------------------------------------------------------------------
# Face registries — maps short names to image file paths
# ---------------------------------------------------------------------------
FACES_DIR  = os.path.join(os.path.dirname(__file__), "faces")
MOUTHS_DIR = os.path.join(os.path.dirname(__file__), "faces", "mouths")
EYES_DIR   = os.path.join(os.path.dirname(__file__), "faces", "eyes")

FACE_MAP = {
    "sad":       "sad/sad.png",
    "loading":   "loading/loading.png",
    "loading2":  "loading/loading2.png",
    "loading3":  "loading/loading3.png",
    "driving":   "driving/driving.png",
    "listening": "listening/listening.png",
    "photo":     "photo/photo.png",
}

# Reverse lookup so show() also accepts filenames
REVERSE_MAP = {v: k for k, v in FACE_MAP.items()}

# Rhubarb viseme -> mouth image. X (silence/rest) reuses A.
MOUTH_FILES = {
    "A": "A.png", "B": "B.png", "C": "C.png", "D": "D.png",
    "E": "E.png", "F": "F.png", "G": "G.png", "H": "H.png",
    "X": "A.png",
}

EYE_FILES = {
    "open":      "Eyes_Open.png",
    "open2":     "Eyes_Open2.png",
    "closed":    "Eyes_Closed.png",
    "blink":     "Eyes_Blinked.png",
    "happy":     "Eyes_Closed.png",
    "thinking":  "Eyes_Thinking.png",
    "thinking2": "Eyes_Thinking2.png",
}

# Idle eyes step through one of these two fixed sequences (see _eye_mode),
# each step held for its own duration - normal pace for the open/thinking
# looks, shorter for closed (so the closed->blink->closed flutter doesn't
# drag), and a quick flash for the actual blink (Eyes_Blinked) sandwiched
# between those closed frames so it reads as a real blink rather than a
# sudden cut.
IDLE_EYE_STEPS_NORMAL = [
    ("open",   IDLE_STEP_NORMAL),
    ("closed", IDLE_STEP_CLOSED),
    ("blink",  IDLE_STEP_QUICK),
    ("closed", IDLE_STEP_CLOSED),
    ("open2",  IDLE_STEP_NORMAL),
    ("closed", IDLE_STEP_CLOSED),
    ("blink",  IDLE_STEP_QUICK),
    ("closed", IDLE_STEP_CLOSED),
]

# Thinking: rapidly flicker thinking<->thinking2 a few times, then do one
# closed->blink->closed, then repeat - rather than IDLE_EYE_STEPS_NORMAL's
# slower open/open2 hold-then-blink rhythm.
IDLE_STEP_THINKING_FLICKER = 0.25  # seconds per thinking/thinking2 flicker - short, per request

IDLE_EYE_STEPS_THINKING = [
    ("thinking",  IDLE_STEP_THINKING_FLICKER),
    ("thinking2", IDLE_STEP_THINKING_FLICKER),
    ("thinking",  IDLE_STEP_THINKING_FLICKER),
    ("thinking2", IDLE_STEP_THINKING_FLICKER),
    ("thinking",  IDLE_STEP_THINKING_FLICKER),
    ("thinking2", IDLE_STEP_THINKING_FLICKER),
    ("thinking",  IDLE_STEP_THINKING_FLICKER),
    ("thinking2", IDLE_STEP_THINKING_FLICKER),
    ("closed",    IDLE_STEP_CLOSED),
    ("blink",     IDLE_STEP_QUICK),
    ("closed",    IDLE_STEP_CLOSED),
]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_lock         = threading.Lock()  # guards display state: _current_face/_face_mode/_talking/_listening
_current_face = "loading"  # used only in "legacy" mode - a static FACE_MAP image
_face_mode    = "legacy"   # "legacy" (static FACE_MAP image) or "bmo" (mouth+eyes)
_talking      = False
_listening    = False     # when True, the "listening" overlay is drawn on top
_surfaces     = {}        # name -> pygame.Surface, populated on start() - legacy FACE_MAP faces
_bmo_mouth_surfaces = {}  # mouth key -> pygame.Surface, populated on start()
_bmo_eye_surfaces   = {}  # eye key -> pygame.Surface, populated on start()
_screen       = None
_running      = False

# Shared cycle-animation state — only one cycle runs at a time (load() is the
# only thing that still uses this; wait()/listen()/talk() all render the
# "bmo" mouth+eyes face instead, driven by the idle-cycle thread below).
_cycle_stop   = threading.Event()
_cycle_thread = None

# preload_speech() cache - see that function and speak_with_display().
_preload_cache   = {}  # text -> {"audio", "samplerate", "cues"}, once ready
_preload_threads = {}  # text -> Thread, while still being prepared

_NOT_GIVEN = object()  # sentinel for talk()'s cues param - None is a valid "rhubarb unavailable" value

# BMO mouth/eyes state - guarded by its own lock since it's mutated from the
# idle-cycle thread, the mouth-driving thread, and set_expression(), somewhat
# independently of the display state above.
_face_lock         = threading.Lock()
_current_mouth_key = "A"
_idle_index        = 0       # position in the active idle steps list (see _eye_mode)
_eye_mode          = "normal"  # "normal" or "thinking" - selects which idle steps list plays
_expression         = None  # None (idle cycle runs normally) or "happy"

_idle_stop   = threading.Event()
_idle_thread = None

# ---------------------------------------------------------------------------
# Pi 5 display environment setup
# ---------------------------------------------------------------------------

def _configure_sdl_for_platform():
    """
    Set SDL2 environment variables for the current platform.

    Pi 5 removed the legacy framebuffer (fbdev) — it uses KMS/DRM instead.
    We auto-detect the right driver:

      Non-Linux               → SDL auto-detects (Windows / macOS dev machine)
      Linux + desktop session → SDL auto-detects X11 or Wayland
      Linux + no desktop      → kmsdrm (headless Pi 5 boot to console)
    """
    if sys.platform != "linux":
        return

    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

    has_x11     = bool(os.environ.get("DISPLAY"))
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))

    # A desktop session (e.g. labwc) can be running on the Pi's physical
    # console even when this script is launched from an SSH session that
    # never inherited DISPLAY/WAYLAND_DISPLAY. Check for the actual socket
    # under XDG_RUNTIME_DIR before assuming we're headless - kmsdrm wants
    # exclusive control of the DRM device and fails outright if a running
    # compositor already holds it.
    if not (has_x11 or has_wayland):
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")  # type: ignore[attr-defined]
        for candidate in ("wayland-0", "wayland-1"):
            if os.path.exists(os.path.join(runtime_dir, candidate)):
                os.environ["WAYLAND_DISPLAY"] = candidate
                os.environ.setdefault("XDG_RUNTIME_DIR", runtime_dir)
                has_wayland = True
                print(f"[Display] Found running Wayland session at {candidate} (not inherited via env) — using it.")
                break

    if has_x11 or has_wayland:
        # Desktop session running — SDL will find X11/Wayland automatically.
        print("[Display] Desktop session detected — SDL will auto-select driver.")
    else:
        # Headless / console boot. Pi 5 requires kmsdrm; fbdev does not exist.
        os.environ["SDL_VIDEODRIVER"]         = "kmsdrm"
        os.environ["SDL_DRMSCANOUT"]          = "1"
        os.environ["SDL_VIDEO_KMSDRM_DEVICE"] = dri_device
        print(f"[Display] No desktop session — using kmsdrm on {dri_device}.")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_to_fit(surface):
    """Scale a surface to fit the screen while preserving aspect ratio."""
    w, h  = surface.get_size()
    scale = min(display_width / w, display_height / h)
    return pygame.transform.smoothscale(surface, (int(w * scale), int(h * scale)))


def current_mouth_key():
    with _face_lock:
        return _current_mouth_key


def _active_idle_steps():
    return IDLE_EYE_STEPS_THINKING if _eye_mode == "thinking" else IDLE_EYE_STEPS_NORMAL


def current_eye_key():
    """
    The "happy" expression (if set) holds closed eyes persistently;
    otherwise the active idle step sequence plays normally - either the
    normal open/closed/blink/closed/open2/... cycle, or (see set_eye_mode())
    the thinking/closed/blink/closed/thinking2/... cycle.
    """
    with _face_lock:
        if _expression:
            return _expression
        steps = _active_idle_steps()
        return steps[_idle_index % len(steps)][0]


def set_mouth(viseme: str):
    global _current_mouth_key
    with _face_lock:
        _current_mouth_key = viseme if viseme in MOUTH_FILES else "A"


def set_eye_mode(mode: str):
    """
    mode: "normal" (open/open2 idle cycle) or "thinking" (flickers between
    Eyes_Thinking/Eyes_Thinking2 instead, same blink rhythm). Only resets the
    step index when actually switching to a different sequence than the one
    already playing - callers like wait()/listen() call this on every phase
    change even when the mode isn't changing (e.g. wait() -> listen() are
    both "normal"), and resetting unconditionally snapped the blink rhythm
    back to step 0 every time, making blinking look like it restarts on each
    phase change instead of continuing on its own cadence.
    """
    global _eye_mode, _idle_index
    if mode not in ("normal", "thinking"):
        raise ValueError(f"Unknown eye mode: {mode!r}. Supported: 'normal', 'thinking'.")
    with _face_lock:
        if _eye_mode != mode:
            _idle_index = 0
        _eye_mode = mode


def set_expression(name: str = None):
    """
    name: None for neutral (the idle step sequence plays normally), or "happy"
    for persistently closed smiling eyes. Independent of the mouth.
    """
    global _expression
    if name not in (None, "happy"):
        raise ValueError(f"Unknown expression: {name!r}. Supported: None, 'happy'.")
    with _face_lock:
        _expression = name


def get_mouth_cues(wav_path: str):
    """
    Returns wav_path's Rhubarb "mouthCues" (via the rhubarb binary), or None
    if RHUBARB_PATH isn't set - callers should fall back to
    amplitude_viseme()-based mouth driving in that case.
    """
    if not RHUBARB_PATH:
        return None
    fd, json_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        subprocess.run(
            # "phonetic" instead of the default "pocketSphinx" recognizer -
            # the latter loads a full speech-recognition dictionary on every
            # single call (this runs rhubarb fresh each time, not as a
            # persistent server), adding a multi-second delay before any
            # cues are ready. "phonetic" analyzes the audio's formants
            # directly with no dictionary, so it's much faster - worth it
            # since BMO's mouth shapes don't need word-level accuracy.
            [RHUBARB_PATH, "-r", "phonetic", "-f", "json", "-o", json_path, wav_path],
            check=True, capture_output=True,
        )
        with open(json_path, "r") as f:
            return json.load(f)["mouthCues"]
    finally:
        os.remove(json_path)


def amplitude_viseme(rms: float) -> str:
    """Rough RMS-amplitude -> mouth-shape bucketing, used when rhubarb isn't installed."""
    if rms < 0.01:
        return "A"
    elif rms < 0.04:
        return "C"
    elif rms < 0.08:
        return "D"
    return "G"


def _idle_cycle_loop():
    """
    Steps through the active idle steps list (see _active_idle_steps()),
    each held for its own duration (a quick flash for "blink", normal pace
    otherwise). Keeps running even during "happy" - current_eye_key() just
    ignores _idle_index while that's set, so the cycle is always picked
    back up at the right point afterward.

    Polls in <=50ms ticks (rather than one long wait per step) so a
    set_eye_mode() switch is picked up almost instantly instead of only
    after the current step's full duration elapses.
    """
    global _idle_index
    tick = 0.05
    while not _idle_stop.is_set():
        with _face_lock:
            mode_at_start = _eye_mode
        steps = _active_idle_steps()
        _, duration = steps[_idle_index % len(steps)]
        deadline = time.monotonic() + duration

        while not _idle_stop.is_set():
            with _face_lock:
                mode_changed = _eye_mode != mode_at_start
            if mode_changed:
                break  # set_eye_mode() already reset _idle_index to 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(tick, remaining))

        if _idle_stop.is_set():
            break

        with _face_lock:
            if _eye_mode == mode_at_start:
                _idle_index = (_idle_index + 1) % len(steps)
            # else: mode switched mid-step - _idle_index is already 0, don't double-advance


def _render():
    """Draw the current face to the screen, then the listening overlay on top if active."""
    _screen.fill(display_bg_color)
    with _lock:
        mode      = _face_mode
        name      = _current_face
        listening = _listening

    center = (display_width // 2 + face_offset_x, display_height // 2 + face_offset_y)

    if mode == "bmo":
        mouth_surf = _bmo_mouth_surfaces[current_mouth_key()]
        eye_surf   = _bmo_eye_surfaces[current_eye_key()]
        _screen.blit(mouth_surf, mouth_surf.get_rect(center=center))
        _screen.blit(eye_surf, eye_surf.get_rect(center=center))
    else:
        surf = _surfaces.get(name)
        if surf:
            _screen.blit(surf, surf.get_rect(center=center))
        else:
            # Fallback: render face name as green text
            font = pygame.font.SysFont("monospace", 36)
            s = font.render(f"[ {name} ]", True, (0, 255, 0))
            _screen.blit(s, s.get_rect(center=center))

    if listening:
        overlay = _surfaces.get("listening")
        if overlay:
            _screen.blit(overlay, overlay.get_rect(center=center))

    pygame.display.flip()


def _run_loop():
    """Main pygame event + render loop. Runs in a background thread."""
    global _screen, _running, display_width, display_height

    _configure_sdl_for_platform()

    pygame.init()
    pygame.display.set_caption("BMO Robot Display")
    flags = pygame.FULLSCREEN if display_fullscreen else 0

    if display_fullscreen:
        # (0, 0) + FULLSCREEN tells SDL to use whatever the current screen's
        # real resolution actually is, instead of relying on
        # config.py's display_width/height matching the physical screen.
        # vsync=1 syncs flip() to the display's actual refresh instead of
        # firing on our own clock.tick() schedule - without it, frames land
        # at inconsistent points relative to the real refresh (tearing, and
        # frames effectively shown for 1 or 2 refreshes unevenly), which
        # reads as "not smooth" even when clock.tick() itself is rock
        # steady. Degrades gracefully (no exception) if the backend can't
        # honor it.
        _screen = pygame.display.set_mode((0, 0), flags, vsync=1)
        display_width, display_height = _screen.get_size()
        print(f"[Display] Auto-detected screen size: {display_width}x{display_height}")
    else:
        # No real screen to detect on a dev machine - use the configured
        # preview window size instead.
        _screen = pygame.display.set_mode((display_width, display_height), flags, vsync=1)

    pygame.mouse.set_visible(False)   # hide cursor on touchscreen / kmsdrm
    clock   = pygame.time.Clock()

    # Pre-load all static face images
    for name, filename in FACE_MAP.items():
        path = os.path.join(FACES_DIR, filename)
        if os.path.exists(path):
            try:
                _surfaces[name] = _scale_to_fit(pygame.image.load(path).convert_alpha())
            except pygame.error as e:
                print(f"[Display] Could not load '{path}': {e}")
        else:
            print(f"[Display] Missing image: {path}")

    # Pre-load the BMO mouth/eyes images used by wait()/listen()/talk().
    for key, filename in MOUTH_FILES.items():
        path = os.path.join(MOUTHS_DIR, filename)
        _bmo_mouth_surfaces[key] = _scale_to_fit(pygame.image.load(path).convert_alpha())
    for key, filename in EYE_FILES.items():
        path = os.path.join(EYES_DIR, filename)
        _bmo_eye_surfaces[key] = _scale_to_fit(pygame.image.load(path).convert_alpha())

    print(f"[Display] Loaded {len(_surfaces)} face(s), "
          f"{len(_bmo_mouth_surfaces)} BMO mouth(s), {len(_bmo_eye_surfaces)} BMO eye state(s).")

    while _running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                _running = False
        _render()
        clock.tick(DISPLAY_FPS)

    pygame.quit()


def _compute_mouth_cues(audio, samplerate):
    """
    Writes audio to a temp wav and returns get_mouth_cues() for it (or None
    if rhubarb isn't available). This is the slow part (real rhubarb
    analysis can take ~1s+) - talk() runs it in a background thread in
    parallel with the resample/time-stretch step, then joins it before
    starting playback, so the cues are ready in advance and _drive_mouth()
    just plays them back in lockstep with the audio instead of computing
    them concurrently with (and lagging behind) already-started playback.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(wav_path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(samplerate)
            f.writeframes(audio.tobytes())
        return get_mouth_cues(wav_path)
    finally:
        os.remove(wav_path)


def _drive_mouth(audio, samplerate, cues):
    """
    Plays back pre-computed cues (see _compute_mouth_cues()), or an
    RMS-amplitude approximation if cues is None (rhubarb unavailable).
    Runs in a background thread during talk(), started right alongside
    play_blocking() so `start` below lines up with the actual audio's first
    sample - cue/chunk timestamps are divided by tts_speed since that's how
    much faster (or slower) the played-back audio runs versus this
    original.
    """
    start = time.monotonic()

    if cues is not None:
        for cue in cues:
            with _lock:
                if not _talking:
                    break
            target = start + cue["start"] / tts_speed
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            set_mouth(cue["value"])
    else:
        audio_float = audio.astype(np.float32) / 32768.0
        chunk_size  = max(1, int(samplerate * 0.08))  # ~80ms steps
        pos = 0
        while pos < len(audio_float):
            with _lock:
                if not _talking:
                    break
            chunk = audio_float[pos:pos + chunk_size]
            pos  += chunk_size
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
            set_mouth(amplitude_viseme(rms))
            target = start + (pos / samplerate) / tts_speed
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)


def _stop_cycle():
    """
    Signal any running cycle animation to stop and wait for it to exit.
    Safe to call even when no cycle is running.
    The cycle thread wakes every 50 ms, so this returns almost instantly.

    Also clears the listening overlay - whatever is interrupting the cycle
    (show, talk, load, wait, or a fresh listen()) isn't the listening phase
    anymore, or listen() will set it again right after calling this.
    """
    global _cycle_thread, _listening
    _cycle_stop.set()
    if _cycle_thread is not None and _cycle_thread.is_alive():
        _cycle_thread.join(timeout=LOADING_CYCLE_INTERVAL + 0.2)
    _cycle_thread = None
    with _lock:
        _listening = False


def _run_cycle(faces, interval):
    """
    Rotate through *faces* (list of face-name strings) every *interval* seconds.
    Sleeps in <=50 ms ticks so _cycle_stop is noticed near-instantly.

    Uses an absolute deadline (time.monotonic() + interval) rather than
    accumulating fixed-size sleep ticks - sleep() typically overshoots its
    requested duration slightly, and that error compounds across the ticks
    needed for one interval, making consecutive face durations uneven.
    """
    global _current_face
    idx = 0
    tick = 0.05
    while not _cycle_stop.is_set():
        with _lock:
            _current_face = faces[idx % len(faces)]
        idx += 1

        deadline = time.monotonic() + interval
        while not _cycle_stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(tick, remaining))


def _start_cycle(faces, interval=None):
    """
    Stop any current cycle and start a new one with the given face list.
    Non-blocking — the cycle runs in a daemon thread.
    """
    global _cycle_thread
    if not enable_display:
        return
    if interval is None:
        interval = LOADING_CYCLE_INTERVAL

    _stop_cycle()
    _cycle_stop.clear()

    _cycle_thread = threading.Thread(
        target=_run_cycle,
        args=(faces, interval),
        daemon=True
    )
    _cycle_thread.start()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start():
    """Open the display window in a background thread. Call once at startup."""
    global _running, _idle_thread
    if not enable_display:
        print("[Display] Display disabled (enable_display=False). Skipping start().")
        return
    if _running:
        return
    _running = True
    threading.Thread(target=_run_loop, daemon=True).start()
    # Slightly longer sleep on Pi 5 — kmsdrm initialisation takes a moment.
    time.sleep(0.5)

    _idle_stop.clear()
    _idle_thread = threading.Thread(target=_idle_cycle_loop, daemon=True)
    _idle_thread.start()


def stop():
    """Close the display window."""
    global _running
    if not enable_display:
        return
    _idle_stop.set()
    if _idle_thread is not None and _idle_thread.is_alive():
        _idle_thread.join(timeout=1)
    _stop_cycle()
    _running = False
    time.sleep(0.4)
    pygame.quit()


def show(face_name: str):
    """
    Show a static FACE_MAP face by key or filename. Stops any running cycle
    animation instantly, and switches out of "bmo" mode if that was active.

    Examples:
        show("sad")
    """
    if not enable_display:
        return
    global _current_face, _talking, _face_mode
    _stop_cycle()
    name = REVERSE_MAP.get(face_name, face_name)
    if name not in FACE_MAP:
        print(f"[Display] Unknown face '{face_name}'. Available: {list(FACE_MAP.keys())}")
        return
    print(f"[Display] Showing: {name} ({FACE_MAP[name]})")
    with _lock:
        _current_face = name
        _face_mode    = "legacy"
        _talking      = False


def listen():
    """
    Show the BMO mouth+eyes face (idle blink cycle, mouth at rest) with the
    "listening" graphic overlaid on top.
    Stops instantly when show(), talk(), load(), or wait() is called.
    """
    print("[Display] Phase: listening")
    _stop_cycle()
    global _face_mode, _listening
    with _lock:
        _face_mode = "bmo"
        _listening = True
    set_eye_mode("normal")
    set_mouth("A")


def think():
    """
    Show the BMO mouth+eyes face in "thinking" mode: mouth at rest, eyes
    flicker between Eyes_Thinking/Eyes_Thinking2 (same blink rhythm as the
    normal idle cycle - see IDLE_EYE_STEPS_THINKING). Meant for the gap
    between a user's input and the system actually responding with
    audio/display - see main.py.
    Stops instantly when show(), talk(), listen(), load(), or wait() is called.
    """
    print("[Display] Phase: thinking")
    _stop_cycle()
    global _face_mode
    with _lock:
        _face_mode = "bmo"
    set_eye_mode("thinking")
    set_mouth("A")


_warmed_up = False


def _warmup_lipsync():
    """
    Synthesizes and runs rhubarb on a short throwaway phrase (audio is
    discarded, nothing is played) to eat the first-call cold-start costs -
    ONNX runtime's first real inference, OS disk-cache miss on rhubarb.exe -
    during the loading screen's hold time instead of during the first real
    talk(). See load(), which fires this in the background.
    """
    global _warmed_up
    if _warmed_up or _piper_voice is None:
        return
    _warmed_up = True
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(wav_path, "wb") as f:
                _piper_voice.synthesize_wav("Hello", f)
            get_mouth_cues(wav_path)  # discarded - just warms rhubarb up
        finally:
            os.remove(wav_path)
    except Exception as e:
        print(f"[Display] Lip-sync warmup failed (non-fatal): {e}")


def load(interval: float = None):
    """
    Start the loading animation: cycles loading → loading2 → loading3 → …
    Non-blocking. Stops instantly when show(), talk(), listen(), or wait() is called.

    Also kicks off a one-time background warmup (see _warmup_lipsync()) so
    the loading screen's hold time absorbs the first-call cold-start cost
    instead of the first real talk() being the slow one.
    """
    if interval is None:
        interval = LOADING_CYCLE_INTERVAL
    print("[Display] Phase: loading")
    global _face_mode
    with _lock:
        _face_mode = "legacy"
    threading.Thread(target=_warmup_lipsync, daemon=True).start()
    _start_cycle(["loading", "loading2", "loading3"], interval)


def wait():
    """
    Show the BMO mouth+eyes face (idle blink cycle, mouth at rest).
    Stops instantly when show(), talk(), listen(), or load() is called.
    """
    print("[Display] Phase: waiting")
    _stop_cycle()
    global _face_mode
    with _lock:
        _face_mode = "bmo"
    set_eye_mode("normal")
    set_mouth("A")


def talk(audio: np.ndarray, samplerate: int, keep_listening: bool = False, cues=_NOT_GIVEN):
    """
    Play audio and lip-sync the BMO mouth+eyes face to it (see
    _drive_mouth()). Blocks until audio finishes. Stops any running cycle
    animation before playing.

    Args:
        audio:          numpy int16 array of PCM audio samples.
        samplerate:     sample rate of the audio (e.g. 22050).
        keep_listening: if True, the "listening" overlay stays drawn on top
                         of the talking animation and the idle face
                         afterward, instead of being cleared like any other
                         talk() call would. Used for announcing "Listening!"
                         without the overlay disappearing while it's said.
        cues:           pre-computed mouth cues (see preload_speech()) -
                         skips this call's own cue computation if given.
    """
    global _talking, _listening, _face_mode
    from scipy.signal import resample_poly
    import librosa
    from audio_output import play_blocking

    # Compute mouth cues (real rhubarb analysis - can take ~1s+) in the
    # background NOW, in parallel with the resample/time-stretch below,
    # rather than after playback has already started. That previous
    # approach made the mouth visibly lag behind already-playing audio by
    # however long analysis took; this way the cues are ready and waiting
    # by the time play_blocking() actually runs, for genuinely synced playback
    # from the first frame - at the cost of a brief pre-roll while both run,
    # mostly hidden behind the resample/time-stretch work that already
    # happened serially before playback even without this. Skipped
    # entirely if `cues` was already supplied (see preload_speech()).
    cue_result = {"cues": cues} if cues is not _NOT_GIVEN else {}
    cue_thread = None
    if cues is _NOT_GIVEN and enable_display:
        def _prep_cues():
            cue_result["cues"] = _compute_mouth_cues(audio, samplerate)
        cue_thread = threading.Thread(target=_prep_cues, daemon=True)
        cue_thread.start()

    # This sound card only accepts device_fs (e.g. 48000Hz) exactly - no
    # arbitrary rates - so tts_speed can't be applied by lying about the
    # samplerate. Resample to device_fs, then apply tts_speed via a true
    # time-stretch (librosa, phase vocoder) so pitch isn't affected.
    playback_audio = resample_poly(audio, up=device_fs, down=samplerate)
    playback_float = (playback_audio / 32768.0).astype(np.float32)

    # time_stretch's STFT/ISTFT can drop part of the final frame right at the
    # array boundary, clipping the last bit of audio. Pad with silence first
    # so the real tail isn't sitting at the edge, then trim back down to the
    # expected stretched length afterward.
    pad = int(device_fs * 0.1)
    expected_len = int(round(len(playback_float) / tts_speed))
    playback_float = librosa.effects.time_stretch(np.pad(playback_float, (0, pad)), rate=tts_speed)[:expected_len]
    playback_audio = np.clip(playback_float * 32767, -32768, 32767).astype(np.int16)

    # play_blocking() can return before the output device has actually
    # finished flushing its buffer through the speakers, clipping the last
    # bit of real audio. Append trailing silence so there's always a buffer
    # of "nothing" after the real content for that flush to land in.
    playback_audio = np.concatenate([playback_audio, np.zeros(int(device_fs * TRAILING_SILENCE_SECS), dtype=np.int16)])

    if not enable_display:
        play_blocking(playback_audio)
        return

    _stop_cycle()  # also clears _listening - restored below, right alongside playback, if requested

    with _lock:
        _talking = True
        # Deliberately NOT touching _face_mode/eye-mode/_listening here -
        # whatever the caller had active (e.g. think()'s thinking flicker)
        # stays untouched through the whole prep phase below, so it's only
        # ever "thinking" before audio is heard, never something else.

    if cue_thread is not None:
        cue_thread.join()  # cues are ready (or this just waits out whichever of the two took longer)

    # Flip to the talking face (and the "listening" overlay, if requested)
    # at the exact instant playback starts below - not a moment before -
    # so e.g. the listening overlay and "Listening!" being said start
    # together instead of the overlay jumping the gun during synthesis/cue
    # prep above.
    set_eye_mode("normal")
    with _lock:
        _face_mode = "bmo"
        if keep_listening:
            _listening = True

    # Start the mouth thread right alongside play_blocking() (nothing else
    # between the two) so _drive_mouth's `start = time.monotonic()` anchor
    # actually matches the audio onset, instead of drifting ahead of it by
    # however long set_eye_mode()/the face-state lock above took. The
    # persistent stream play_blocking() writes to is already open/running,
    # so this call has less startup overhead than the old sd.play() (which
    # opened a fresh ALSA connection every time) - if anything, tighter
    # timing than before, not looser.
    mouth_thread = threading.Thread(target=_drive_mouth, args=(audio, samplerate, cue_result["cues"]), daemon=True)
    mouth_thread.start()
    play_blocking(playback_audio)

    with _lock:
        _talking = False
    mouth_thread.join(timeout=1)
    set_mouth("A")

    if not keep_listening:
        with _lock:
            _listening = False
    # Stay on the BMO idle face (still blinking) rather than switching to a
    # static "done talking" image.


def preload_speech(text: str):
    """
    Synthesizes `text` and pre-computes its mouth cues right now, in a
    background thread, caching the result so a later speak_with_display(text)
    call with the EXACT same text skips straight to playback instead of
    redoing that ~1s+ work right before actually speaking. For text that
    isn't known in advance (e.g. AI responses) this can't help - it's meant
    for a fixed string known ahead of time, like main.py's startup greeting,
    called during the loading screen so its hold time covers the cost.
    Non-blocking; safe to call more than once for the same text.
    """
    if _piper_voice is None or text in _preload_cache or text in _preload_threads:
        return

    def _do_preload():
        import io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as f:
            _piper_voice.synthesize_wav(text, f)
        buf.seek(0)
        with wave.open(buf, "rb") as f:
            audio      = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            samplerate = f.getframerate()
        cues = _compute_mouth_cues(audio, samplerate)
        _preload_cache[text] = {"audio": audio, "samplerate": samplerate, "cues": cues}
        del _preload_threads[text]

    t = threading.Thread(target=_do_preload, daemon=True)
    _preload_threads[text] = t
    t.start()


def speak_with_display(text: str, keep_listening: bool = False):
    """
    Drop-in replacement for speech.speak().
    Synthesises speech with Piper and calls talk() to animate the display.

    Usage in main.py:
        from display_change import start, speak_with_display as speak
        start()
        speak("Hello!")

    Pass keep_listening=True to keep the "listening" overlay on top while
    this is said (see talk()) - e.g. announcing "Listening!" itself.

    If preload_speech(text) was already called for this exact text, reuses
    that cached synthesis + cues instead of redoing the work now.
    """
    if _piper_voice is None:
        print("[Display] TTS not available (voice failed to load at import).")
        return

    if text in _preload_threads:
        _preload_threads[text].join()

    cached = _preload_cache.pop(text, None)
    if cached is not None:
        talk(cached["audio"], cached["samplerate"], keep_listening=keep_listening, cues=cached["cues"])
        return

    voice = _piper_voice
    import io
    buf   = io.BytesIO()
    with wave.open(buf, "wb") as f:
        voice.synthesize_wav(text, f)
    buf.seek(0)
    with wave.open(buf, "rb") as f:
        audio      = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        samplerate = f.getframerate()
    talk(audio, samplerate, keep_listening=keep_listening)


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to preview faces/animations, then
# interactively type lines for BMO to say and watch the mouth/eyes lip-sync.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("BMO Display Test — press Ctrl+C to exit.")
    print(f"Display: {display_width}x{display_height}  Fullscreen: {display_fullscreen}")
    start()
    try:
        print("\n-- Static faces --")
        for face in FACE_MAP:
            print(f"  Showing: {face}")
            show(face)
            time.sleep(1)

        print("\n-- listen() for 3s (BMO face + listening overlay) --")
        listen()
        time.sleep(3)

        print("\n-- load() for 3s --")
        load()
        time.sleep(3)

        print("\n-- wait() for 3s (BMO face, idle blink cycle) --")
        wait()
        time.sleep(3)

        print("\n-- show() interrupts wait() immediately --")
        wait()
        time.sleep(1)
        show("sad")   # cuts the BMO idle face mid-blink
        time.sleep(2)

        wait()
        if RHUBARB_PATH:
            print(f"\n[Display] Using rhubarb at {RHUBARB_PATH} for real lip-sync.")
        else:
            print("\n[Display] rhubarb not on PATH - mouth movement will be an "
                  "amplitude-based approximation, not real lip-sync.")
        print("Type a line for BMO to say (empty line to quit).")
        while True:
            text = input("Say: ").strip()
            if not text:
                break
            speak_with_display(text)

    except KeyboardInterrupt:
        pass
    stop()
