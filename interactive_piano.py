import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pygame
import tkinter as tk
from tkinter import ttk

from piano_utils import PianoRegion, midi_note_to_name


@dataclass
class PianoConfig:
    num_keys: int = 14
    base_midi_note: int = 60  # Middle C (C4)
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    cooldown_s: float = 0.05
    samples_dir: Path = Path("sounds")
    volume: float = 0.8


@dataclass
class ControlState:
    volume: float
    cooldown_s: float
    show_landmarks: bool = True
    should_quit: bool = False
    reset_calibration: bool = False

    def update_volume(self, value: float) -> None:
        self.volume = value

    def update_cooldown(self, value: float) -> None:
        self.cooldown_s = value


class SamplePlayer:
    def __init__(self, base_note: int, num_keys: int, samples_dir: Path, volume: float) -> None:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.samples: Dict[int, pygame.mixer.Sound] = {}
        self.samples_dir = samples_dir
        self._load_samples(base_note, num_keys)
        self.set_volume(volume)

    def _load_samples(self, base_note: int, num_keys: int) -> None:
        for key_index in range(num_keys):
            midi_note = base_note + key_index
            name = midi_note_to_name(midi_note)
            sample_path = self.samples_dir / f"{name}.wav"
            if sample_path.exists():
                self.samples[key_index] = pygame.mixer.Sound(str(sample_path))
            else:
                print(f"[warn] Missing sample: {sample_path}")

    def play(self, key_index: int) -> None:
        sample = self.samples.get(key_index)
        if sample is None:
            return
        sample.play()

    def set_volume(self, volume: float) -> None:
        for sample in self.samples.values():
            sample.set_volume(volume)

    def close(self) -> None:
        pygame.mixer.quit()


class InteractivePiano:
    def __init__(self, config: PianoConfig, controls: ControlState) -> None:
        self.config = config
        self.controls = controls
        self.region: Optional[PianoRegion] = None
        self.keys_down: Set[int] = set()
        self.last_trigger: Dict[int, float] = {}
        self.player = SamplePlayer(
            config.base_midi_note, config.num_keys, config.samples_dir, config.volume
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            max_num_hands=2,
        )

    def set_region(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> None:
        self.region = PianoRegion(top_left, bottom_right)

    def get_key_index(self, point: Tuple[int, int]) -> Optional[int]:
        if self.region is None or not self.region.contains(point):
            return None
        rel_x = point[0] - self.region.top_left[0]
        key_width = self.region.width / self.config.num_keys
        index = int(rel_x / key_width)
        if 0 <= index < self.config.num_keys:
            return index
        return None

    def trigger_notes(self, active_keys: Set[int]) -> None:
        now = time.time()
        new_keys = active_keys - self.keys_down

        for key in new_keys:
            last = self.last_trigger.get(key, 0)
            if now - last >= self.controls.cooldown_s:
                self.player.play(key)
                self.last_trigger[key] = now

        self.keys_down = active_keys

    def apply_controls(self) -> None:
        self.player.set_volume(self.controls.volume)

    def shutdown(self) -> None:
        self.player.close()
        self.hands.close()


def build_controls(config: PianoConfig) -> ControlState:
    return ControlState(volume=config.volume, cooldown_s=config.cooldown_s)


def launch_gui(controls: ControlState) -> None:
    root = tk.Tk()
    root.title("Interactive Piano Controls")

    volume_var = tk.DoubleVar(value=controls.volume)
    cooldown_var = tk.DoubleVar(value=controls.cooldown_s)
    show_landmarks_var = tk.BooleanVar(value=controls.show_landmarks)

    def on_volume_change(_event: object = None) -> None:
        controls.update_volume(volume_var.get())

    def on_cooldown_change(_event: object = None) -> None:
        controls.update_cooldown(cooldown_var.get())

    def on_landmarks_toggle() -> None:
        controls.show_landmarks = show_landmarks_var.get()

    def on_reset_calibration() -> None:
        controls.reset_calibration = True

    def on_quit() -> None:
        controls.should_quit = True
        root.destroy()

    ttk.Label(root, text="Volume").pack(anchor="w", padx=10, pady=(10, 0))
    volume_slider = ttk.Scale(
        root, from_=0.0, to=1.0, orient="horizontal", variable=volume_var, command=on_volume_change
    )
    volume_slider.pack(fill="x", padx=10)

    ttk.Label(root, text="Cooldown (seconds)").pack(anchor="w", padx=10, pady=(10, 0))
    cooldown_slider = ttk.Scale(
        root,
        from_=0.01,
        to=0.3,
        orient="horizontal",
        variable=cooldown_var,
        command=on_cooldown_change,
    )
    cooldown_slider.pack(fill="x", padx=10)

    landmarks_check = ttk.Checkbutton(
        root, text="Show fingertip markers", variable=show_landmarks_var, command=on_landmarks_toggle
    )
    landmarks_check.pack(anchor="w", padx=10, pady=(10, 0))

    ttk.Button(root, text="Reset calibration", command=on_reset_calibration).pack(
        fill="x", padx=10, pady=(10, 0)
    )
    ttk.Button(root, text="Quit", command=on_quit).pack(fill="x", padx=10, pady=(10, 10))

    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()


def draw_region(frame: np.ndarray, region: PianoRegion, num_keys: int) -> None:
    cv2.rectangle(frame, region.top_left, region.bottom_right, (0, 255, 0), 2)
    key_width = region.width / num_keys
    for i in range(1, num_keys):
        x = int(region.top_left[0] + i * key_width)
        cv2.line(frame, (x, region.top_left[1]), (x, region.bottom_right[1]), (0, 255, 0), 1)


def main() -> None:
    config = PianoConfig()
    controls = build_controls(config)
    piano = InteractivePiano(config, controls)

    gui_thread = threading.Thread(target=launch_gui, args=(controls,), daemon=True)
    gui_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    calibration_points: List[Tuple[int, int]] = []

    def handle_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal calibration_points
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_points.append((x, y))
            if len(calibration_points) == 2:
                top_left = (
                    min(calibration_points[0][0], calibration_points[1][0]),
                    min(calibration_points[0][1], calibration_points[1][1]),
                )
                bottom_right = (
                    max(calibration_points[0][0], calibration_points[1][0]),
                    max(calibration_points[0][1], calibration_points[1][1]),
                )
                piano.set_region(top_left, bottom_right)

    cv2.namedWindow("Interactive Piano")
    cv2.setMouseCallback("Interactive Piano", handle_mouse)

    try:
        while True:
            if controls.should_quit:
                break

            if controls.reset_calibration:
                calibration_points = []
                piano.region = None
                controls.reset_calibration = False

            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = piano.hands.process(rgb)

            active_keys: Set[int] = set()

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for tip_id in FINGERTIP_IDS:
                        lm = hand_landmarks.landmark[tip_id]
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        key_index = piano.get_key_index((x, y))
                        if key_index is not None:
                            active_keys.add(key_index)
                        if controls.show_landmarks:
                            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

            piano.trigger_notes(active_keys)
            piano.apply_controls()

            if piano.region:
                draw_region(frame, piano.region, config.num_keys)
            else:
                cv2.putText(
                    frame,
                    "Press 'c' then click top-left and bottom-right of piano area",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Interactive Piano", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibration_points = []

    finally:
        cap.release()
        cv2.destroyAllWindows()
        piano.shutdown()


FINGERTIP_IDS = [4, 8, 12, 16, 20]


if __name__ == "__main__":
    main()
