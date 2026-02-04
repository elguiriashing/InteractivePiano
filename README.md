# Interactive Piano (Projector + Webcam)

This project provides a small Python app that detects your hands from a webcam feed and triggers piano **audio samples** when your fingertips enter mapped key regions. It is designed to be used alongside projector mapping tools (e.g., map.club) so you can project a piano layout onto a surface.

## What this does
- Uses **MediaPipe Hands** to track fingertips.
- Maps fingertip positions into **virtual piano key regions**.
- Plays **polyphonic chords** by triggering multiple audio files at once.
- Provides a **calibration step** so the mapped key area matches your projection.
- Offers a **control GUI** for volume, cooldown, and calibration reset.

## Quick start

1. **Create a virtual environment** (recommended).
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Add audio samples** (see [Audio samples](#audio-samples)).
4. **Run the app**:

```bash
python interactive_piano.py
```

## Controls
- **`c`**: Start calibration (click top-left, then bottom-right of the projected piano area).
- **`q`**: Quit.

The control window lets you:
- Adjust **volume**.
- Change the **cooldown** between retriggers.
- Toggle **fingertip markers**.
- **Reset calibration**.

## Calibration
When you press **`c`**, the app asks you to click two points in the camera view:
1. **Top-left** corner of the projected piano.
2. **Bottom-right** corner of the projected piano.

It then divides the rectangle into equal-width keys.

## Audio samples
The app expects `.wav` samples in a `sounds/` folder next to `interactive_piano.py`. File names are derived from MIDI note names, starting at `C4` by default:

```
sounds/
  C4.wav
  C#4.wav
  D4.wav
  D#4.wav
  E4.wav
  F4.wav
  F#4.wav
  G4.wav
  G#4.wav
  A4.wav
  A#4.wav
  B4.wav
  C5.wav
  C#5.wav
```

You can change the number of keys, base note, or sample folder in `interactive_piano.py`.

## Notes / Tips
- Try to keep the camera stable and perpendicular to the projection surface.
- Increase lighting if tracking is unstable.
- If you see `[warn] Missing sample: ...`, add the matching `.wav` file.

## Requirements
- Python 3.9+ recommended
- A webcam
- Audio output (speakers or headphones)

