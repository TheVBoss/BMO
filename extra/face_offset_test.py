#!/usr/bin/env python3
"""
Interactive face_offset_x/face_offset_y tuner. Opens the real display
(same fullscreen/kmsdrm detection as display_change.py) showing the idle
BMO face, and lets you nudge it around live with either the keyboard
(arrow keys) or the D-Pad buttons (UpArrow/DownArrow/LeftArrow/RightArrow -
see config.BUTTON_PINS) wired to the Arduino Micro. Press 'r' to reset to
0,0. Triangle (or Enter on the keyboard) saves straight into config.py and
exits immediately, no prompt. Esc or close the window quits WITHOUT saving
by default, but still asks first in case that was a misclick.

Run with: python3 extra/face_offset_test.py
"""
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import pygame
from display_change import (
    _configure_sdl_for_platform, display_fullscreen, display_width, display_height,
    display_bg_color, MOUTHS_DIR, EYES_DIR, MOUTH_FILES, EYE_FILES,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.py"

PIXELS_PER_SEC = 120  # how fast the face moves while a key/button is held
DPAD_NAMES = ("UpArrow", "DownArrow", "LeftArrow", "RightArrow")
DPAD_TO_DELTA = {
    "UpArrow":    (0, -1),
    "DownArrow":  (0, 1),
    "LeftArrow":  (-1, 0),
    "RightArrow": (1, 0),
}
SAVE_BUTTON_NAME = "Triangle"


def _scale_to_fit(surface, max_w, max_h):
    w, h = surface.get_size()
    scale = min(max_w / w, max_h / h)
    return pygame.transform.smoothscale(surface, (int(w * scale), int(h * scale)))


def _write_scalar(key, value):
    text = CONFIG_PATH.read_text()
    pattern = re.compile(rf"^{key}\s*=.*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"couldn't find a '{key} =' line in config.py to update")
    backup_path = CONFIG_PATH.with_suffix(".py.bak")
    backup_path.write_text(text)
    CONFIG_PATH.write_text(pattern.sub(f"{key} = {value}", text, count=1))
    print(f"(previous config.py saved to {backup_path.name})")


# ---------------------------------------------------------------------------
# D-Pad + Triangle button listener - runs in a background thread, updates
# `held` in place under `lock` and sets `save_requested` on a Triangle
# press. Harmless no-op if the Arduino isn't connected - keyboard-only
# still works fine.
# ---------------------------------------------------------------------------

def _button_thread(held, lock, stop_event, save_requested):
    try:
        import serial
        ser = serial.Serial(config.BUTTON_PORT, config.BUTTON_BAUD, timeout=1)
    except Exception as e:
        print(f"[FaceOffsetTest] D-Pad/Triangle unavailable ({e}) - keyboard still works.")
        return

    time.sleep(2)  # let the Arduino finish its auto-reset after the port opens
    ser.reset_input_buffer()
    pin_to_name = {pin: name for name, pin in config.BUTTON_PINS.items()}

    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            break
        if not line.startswith("B") or ":" not in line:
            continue
        try:
            pin_str, state_str = line[1:].split(":")
            pin, state = int(pin_str), int(state_str)
        except ValueError:
            continue
        name = pin_to_name.get(pin)
        if name == SAVE_BUTTON_NAME:
            if state == 0:  # pressed
                save_requested.set()
        elif name in DPAD_NAMES:
            with lock:
                held[name] = (state == 0)  # 0 = pressed, 1 = released

    ser.close()


def main():
    offset_x = config.face_offset_x
    offset_y = config.face_offset_y

    held = {name: False for name in DPAD_NAMES}
    lock = threading.Lock()
    stop_event = threading.Event()
    save_requested = threading.Event()
    threading.Thread(target=_button_thread, args=(held, lock, stop_event, save_requested), daemon=True).start()

    _configure_sdl_for_platform()
    pygame.init()
    pygame.display.set_caption("Face Offset Tuner")
    flags = pygame.FULLSCREEN if display_fullscreen else 0
    if display_fullscreen:
        screen = pygame.display.set_mode((0, 0), flags, vsync=1)
        screen_w, screen_h = screen.get_size()
    else:
        screen = pygame.display.set_mode((display_width, display_height), flags, vsync=1)
        screen_w, screen_h = display_width, display_height
    pygame.mouse.set_visible(False)

    mouth_surf = _scale_to_fit(
        pygame.image.load(os.path.join(MOUTHS_DIR, MOUTH_FILES["A"])).convert_alpha(), screen_w, screen_h)
    eye_surf = _scale_to_fit(
        pygame.image.load(os.path.join(EYES_DIR, EYE_FILES["open"])).convert_alpha(), screen_w, screen_h)
    font = pygame.font.SysFont("monospace", 28)

    print(f"\nArrow keys or D-Pad buttons to move the face. 'r' resets to 0,0. "
          f"{SAVE_BUTTON_NAME} (or Enter) saves & exits immediately. "
          f"Esc or close the window quits (asks first).\n")

    clock = pygame.time.Clock()
    running = True
    save_now = False
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    offset_x, offset_y = 0, 0
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    save_now = True
                    running = False

        if save_requested.is_set():
            save_now = True
            running = False

        keys = pygame.key.get_pressed()
        step = PIXELS_PER_SEC * dt
        if keys[pygame.K_LEFT]:
            offset_x -= step
        if keys[pygame.K_RIGHT]:
            offset_x += step
        if keys[pygame.K_UP]:
            offset_y -= step
        if keys[pygame.K_DOWN]:
            offset_y += step

        with lock:
            for name, (dx, dy) in DPAD_TO_DELTA.items():
                if held[name]:
                    offset_x += dx * step
                    offset_y += dy * step

        screen.fill(display_bg_color)
        center = (screen_w // 2 + int(offset_x), screen_h // 2 + int(offset_y))
        screen.blit(mouth_surf, mouth_surf.get_rect(center=center))
        screen.blit(eye_surf, eye_surf.get_rect(center=center))

        label = font.render(f"face_offset_x = {int(offset_x)}   face_offset_y = {int(offset_y)}",
                             True, (0, 0, 0))
        screen.blit(label, (10, 10))
        hint = font.render(f"{SAVE_BUTTON_NAME}/Enter = save & exit   Esc = quit", True, (0, 0, 0))
        screen.blit(hint, (10, 10 + label.get_height() + 4))

        pygame.display.flip()

    stop_event.set()
    pygame.quit()

    offset_x, offset_y = int(offset_x), int(offset_y)
    print(f"\nFinal: face_offset_x = {offset_x}   face_offset_y = {offset_y}")

    if save_now or input("Save to config.py? [y/N] ").strip().lower() == "y":
        _write_scalar("face_offset_x", offset_x)
        _write_scalar("face_offset_y", offset_y)
        print("config.py updated.")
    else:
        print("Not saved.")


if __name__ == "__main__":
    main()
