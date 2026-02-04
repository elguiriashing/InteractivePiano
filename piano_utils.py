from dataclasses import dataclass
from typing import Tuple

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class PianoRegion:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    def contains(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return (
            self.top_left[0] <= x <= self.bottom_right[0]
            and self.top_left[1] <= y <= self.bottom_right[1]
        )

    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]

    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]


def midi_note_to_name(midi_note: int) -> str:
    name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{name}{octave}"
