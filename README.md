# VoxGrip - VS Code Software-Only Backend

This is a hardware-free prototype of the software pipeline for VoxGrip.

## Pipeline

Microphone
-> Offline STT (Vosk)
-> Intent Parser
-> Object Detection (YOLO)
-> Arm Brain
-> Simulated Servo Commands
-> TTS

## 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Install an offline Vosk English model

Download an English Vosk model from the official Vosk models page.

Extract the model folder into:

```text
models/
```

The default expected path is:

```text
models/vosk-model-small-en-us-0.15
```

If your downloaded model has a different folder name, change `VOSK_MODEL_PATH` at the top of `main.py`.

## 4. Run

```bash
python main.py
```

The first YOLO run may download the model specified by `YOLO_MODEL`.

## Example commands

- Open the hand
- Close the hand
- Grab the bottle
- Pick up the cup
- Release object
- Rotate wrist left
- Rotate wrist right
- Center wrist
- Stop

## Important

This version does NOT control real hardware.

The `SimulatedArm` class prints the equivalent servo movement. Later it can be replaced with actual PCA9685/servo control.

The object detector uses a pretrained general-purpose YOLO model. It is not a custom prosthetic-hand/object-grasping model.

## Troubleshooting

### Microphone problem

Check Windows Settings -> System -> Sound -> Input and make sure the correct microphone is selected.

### Camera problem

Try:

```python
CAMERA_INDEX = 1
```

instead of 0.

### Vosk model problem

Make sure the extracted folder exactly matches `VOSK_MODEL_PATH`.

### pyttsx3 voice problem

On Windows, pyttsx3 normally uses the installed Windows speech engine. Check that Windows has an available speech voice.

### YOLO download problem

If automatic model download is blocked, download the Ultralytics model separately and set `YOLO_MODEL` to its local path.
