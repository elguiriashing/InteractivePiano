import unittest

from piano_utils import PianoRegion, midi_note_to_name


class TestPianoUtils(unittest.TestCase):
    def test_midi_note_to_name(self) -> None:
        self.assertEqual(midi_note_to_name(60), "C4")
        self.assertEqual(midi_note_to_name(61), "C#4")
        self.assertEqual(midi_note_to_name(72), "C5")

    def test_region_contains(self) -> None:
        region = PianoRegion((10, 20), (110, 120))
        self.assertTrue(region.contains((10, 20)))
        self.assertTrue(region.contains((50, 50)))
        self.assertFalse(region.contains((5, 50)))
        self.assertFalse(region.contains((200, 200)))


if __name__ == "__main__":
    unittest.main()
