import json
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO


# ============================================================
# VOXGRIP - SOFTWARE-ONLY BACKEND DEMO
# ------------------------------------------------------------
# Pipeline:
# Microphone -> Offline STT -> Intent Parser
#            -> Object Detection (when required)
#            -> Arm Brain -> Simulated Servos
#            -> TTS feedback
#
# NO REAL HARDWARE IS REQUIRED FOR THIS VERSION.
# ============================================================


# -----------------------------
# 1. CONFIGURATION
# -----------------------------

VOSK_MODEL_PATH = Path("models/vosk-model-small-en-us-0.15")
YOLO_MODEL = "yolo11n.pt"   # Ultralytics will download it on first run
CAMERA_INDEX = 0

SAMPLE_RATE = 16000
COMMAND_SECONDS = 5

# Object classes from the COCO-trained YOLO model that we allow as targets.
TARGET_OBJECTS = {
    "bottle",
    "cup",
    "bowl",
    "apple",
    "banana",
    "orange",
    "book",
    "cell phone",
    "laptop",
    "keyboard",
    "mouse",
    "backpack",
    "scissors",
    "remote",
    "sports ball",
}


# -----------------------------
# 2. SIMULATED SERVO POSITIONS
# -----------------------------
# These are ONLY software demo values.
# They are NOT calibrated values for MG996R/RDS3115.

OPEN_POSITION = {
    "thumb": 10,
    "index": 10,
    "middle": 10,
    "ring": 10,
    "little": 10,
}

CLOSE_POSITION = {
    "thumb": 150,
    "index": 150,
    "middle": 150,
    "ring": 150,
    "little": 150,
}

WRIST_CENTER = 90
WRIST_LEFT = 45
WRIST_RIGHT = 135


# -----------------------------
# 3. TTS
# -----------------------------

def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
    return engine


tts = initialize_tts()


def speak(text):
    """Convert text to spoken audio."""
    print(f"[TTS] {text}")
    tts.say(text)
    tts.runAndWait()


# -----------------------------
# 4. OFFLINE STT USING VOSK
# -----------------------------

def load_stt_model():
    if not VOSK_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nVosk model not found at:\n"
            f"  {VOSK_MODEL_PATH}\n\n"
            f"Download an English Vosk model and extract it into:\n"
            f"  {VOSK_MODEL_PATH.parent}\n"
        )

    print("[STT] Loading Vosk model...")
    return Model(str(VOSK_MODEL_PATH))


stt_model = load_stt_model()


def record_audio(seconds=COMMAND_SECONDS):
    """Record microphone audio from the default input device."""
    print(f"\n🎤 Speak now... ({seconds} seconds)")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    return audio.flatten().tobytes()

command_grammar = json.dumps([
    "open the hand",
    "close the hand",
    "open hand",
    "close hand",
    "shut the hand",
    "grab the bottle",
    "grab bottle",
    "pick up the bottle",
    "grab the cup",
    "grab cup",
    "pick up the cup",
    "release object",
    "release",
    "rotate wrist left",
    "rotate wrist right",
    "center wrist",
    "stop"
])

def speech_to_text(audio_bytes):
    """Convert recorded audio to text using offline Vosk STT."""

    recognizer = KaldiRecognizer(
        stt_model,
        SAMPLE_RATE,
        command_grammar
    )

    recognizer.SetWords(True)

    recognizer.AcceptWaveform(audio_bytes)

    result = json.loads(recognizer.FinalResult())
    text = result.get("text", "").strip().lower()

    return text

# -----------------------------
# 5. INTENT PARSER
# -----------------------------

def normalize_text(text):
    return " ".join(text.lower().strip().split())


def extract_target_object(text):
    """Find a supported object name in the recognized command."""
    text = normalize_text(text)

    # Check longer names first.
    for obj in sorted(TARGET_OBJECTS, key=len, reverse=True):
        if obj in text:
            return obj

    return None


def parse_intent(text):
    """
    Convert recognized speech into a structured command.

    Returns:
        {
            "intent": ...,
            "target": ...,
            "raw_text": ...
        }
    """
    text = normalize_text(text)

    if not text:
        return {
            "intent": "UNKNOWN",
            "target": None,
            "raw_text": text,
        }

    # Emergency/stop first.
    if any(p in text for p in [
        "stop",
        "emergency stop",
        "halt",
        "freeze",
    ]):
        return {
            "intent": "STOP",
            "target": None,
            "raw_text": text,
        }

    # Open hand.
    if any(p in text for p in [
        "open hand",
        "open the hand",
        "open",
        "release hand",
    ]):
        return {
            "intent": "OPEN_HAND",
            "target": None,
            "raw_text": text,
        }

    # Release object.
    if any(p in text for p in [
        "release",
        "let go",
        "drop object",
        "release object",
    ]):
        return {
            "intent": "RELEASE_OBJECT",
            "target": None,
            "raw_text": text,
        }

    # Wrist commands.
    if any(p in text for p in [
        "rotate wrist left",
        "turn wrist left",
        "wrist left",
        "rotate left",
        "turn left",
    ]):
        return {
            "intent": "WRIST_LEFT",
            "target": None,
            "raw_text": text,
        }

    if any(p in text for p in [
        "rotate wrist right",
        "turn wrist right",
        "wrist right",
        "rotate right",
        "turn right",
    ]):
        return {
            "intent": "WRIST_RIGHT",
            "target": None,
            "raw_text": text,
        }

    if any(p in text for p in [
        "center wrist",
        "straighten wrist",
        "wrist center",
    ]):
        return {
            "intent": "WRIST_CENTER",
            "target": None,
            "raw_text": text,
        }

    # Grasp/pick commands.
    if any(p in text for p in [
        "pick up",
        "pickup",
        "pick",
        "grab",
        "grasp",
        "hold",
        "take",
    ]):
        target = extract_target_object(text)

        return {
            "intent": "GRASP_OBJECT",
            "target": target,
            "raw_text": text,
        }

    # Generic close.
    if any(p in text for p in [
        "close hand",
        "close the hand",
        "close",
        "clench",
    ]):
        return {
            "intent": "CLOSE_HAND",
            "target": None,
            "raw_text": text,
        }

    return {
        "intent": "UNKNOWN",
        "target": None,
        "raw_text": text,
    }


# -----------------------------
# 6. SIMULATED ACTUATION
# -----------------------------

class SimulatedArm:
    """
    Software-only replacement for the real PCA9685 + servos.

    Later, this class can be replaced by real hardware code.
    """

    def __init__(self):
        self.finger_positions = OPEN_POSITION.copy()
        self.wrist_position = WRIST_CENTER
        self.stopped = False

    def print_state(self):
        print("\n[SIMULATED ARM STATE]")
        for finger, angle in self.finger_positions.items():
            print(f"  {finger:>6}: {angle:>3}°")
        print(f"  wrist : {self.wrist_position:>3}°")

    def stop(self):
        self.stopped = True
        print("\n[ARM] EMERGENCY STOP / MOTION STOPPED")

    def reset_stop(self):
        self.stopped = False

    def open_hand(self):
        if self.stopped:
            print("[ARM] Stopped. Say 'open hand' or restart the program.")
            return

        print("\n[ARM] Opening hand...")
        self.finger_positions = OPEN_POSITION.copy()
        self.print_state()

    def close_hand(self):
        if self.stopped:
            print("[ARM] Stopped.")
            return

        print("\n[ARM] Closing hand...")
        self.finger_positions = CLOSE_POSITION.copy()
        self.print_state()

    def release_object(self):
        self.open_hand()

    def rotate_wrist(self, angle):
        if self.stopped:
            print("[ARM] Stopped.")
            return

        print(f"\n[ARM] Rotating wrist to {angle}°...")
        self.wrist_position = angle
        self.print_state()

    def grasp(self, target):
        if self.stopped:
            print("[ARM] Stopped.")
            return

        print(f"\n[ARM] Starting grasp sequence for: {target}")
        print("[ARM] Step 1: Position hand")
        time.sleep(0.5)

        print("[ARM] Step 2: Open fingers")
        self.open_hand()
        time.sleep(0.5)

        print("[ARM] Step 3: Approach object (SIMULATED)")
        time.sleep(0.8)

        print("[ARM] Step 4: Close fingers")
        self.close_hand()
        time.sleep(0.5)

        print(f"[ARM] Step 5: Grip of '{target}' completed (SIMULATED)")


arm = SimulatedArm()


# -----------------------------
# 7. OBJECT DETECTION
# -----------------------------

print("\n[VISION] Loading YOLO model...")
detector = YOLO(YOLO_MODEL)


def detect_target_with_camera(target=None, display_seconds=5):
    """
    Open webcam and detect objects.

    If target is supplied, try to find that object.
    Returns:
        {
            "found": bool,
            "object": str | None,
            "confidence": float | None,
            "box": tuple | None
        }
    """

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[VISION] ERROR: Could not open webcam.")
        return {
            "found": False,
            "object": None,
            "confidence": None,
            "box": None,
        }

    print("\n[VISION] Camera opened.")
    print("[VISION] Show the object to the camera.")
    print("[VISION] Press Q to stop vision early.")

    start_time = time.time()
    best_detection = None

    while time.time() - start_time < display_seconds:
        ret, frame = cap.read()

        if not ret:
            print("[VISION] Could not read camera frame.")
            break

        results = detector.predict(
            source=frame,
            conf=0.45,
            verbose=False,
        )

        annotated = frame.copy()

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = names[cls_id].lower()

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Draw all detections.
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )

                label = f"{object_name} {confidence:.2f}"

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Match requested target.
                if target is not None and object_name == target:
                    if best_detection is None or confidence > best_detection["confidence"]:
                        best_detection = {
                            "found": True,
                            "object": object_name,
                            "confidence": confidence,
                            "box": (x1, y1, x2, y2),
                        }

        if best_detection:
            x1, y1, x2, y2 = best_detection["box"]

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                4,
            )

            cv2.putText(
                annotated,
                f"TARGET: {best_detection['object']}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3,
            )

        cv2.imshow("VoxGrip - Object Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if best_detection:
        return best_detection

    return {
        "found": False,
        "object": None,
        "confidence": None,
        "box": None,
    }


# -----------------------------
# 8. ARM BRAIN
# -----------------------------

def execute_command(command):
    """
    This is the central decision layer.
    It receives the structured intent and decides what the arm should do.
    """

    intent = command["intent"]
    target = command["target"]

    print("\n" + "=" * 60)
    print("[ARM BRAIN]")
    print(f"Raw command : {command['raw_text']}")
    print(f"Intent      : {intent}")
    print(f"Target      : {target}")
    print("=" * 60)

    if intent == "UNKNOWN":
        speak(
            "I did not understand the command. "
            "Please try again."
        )
        return

    if intent == "STOP":
        arm.stop()
        speak("Motion stopped.")
        return

    if intent == "OPEN_HAND":
        arm.reset_stop()
        arm.open_hand()
        speak("Opening hand.")
        return

    if intent == "CLOSE_HAND":
        arm.reset_stop()
        arm.close_hand()
        speak("Closing hand.")
        return

    if intent == "RELEASE_OBJECT":
        arm.reset_stop()
        arm.release_object()
        speak("Releasing object.")
        return

    if intent == "WRIST_LEFT":
        arm.reset_stop()
        arm.rotate_wrist(WRIST_LEFT)
        speak("Rotating wrist left.")
        return

    if intent == "WRIST_RIGHT":
        arm.reset_stop()
        arm.rotate_wrist(WRIST_RIGHT)
        speak("Rotating wrist right.")
        return

    if intent == "WRIST_CENTER":
        arm.reset_stop()
        arm.rotate_wrist(WRIST_CENTER)
        speak("Centering wrist.")
        return

    if intent == "GRASP_OBJECT":
        if target is None:
            speak(
                "Please specify an object, "
                "for example, grab the bottle."
            )
            return

        arm.reset_stop()

        speak(f"Looking for the {target}.")

        detection = detect_target_with_camera(target)

        if not detection["found"]:
            print(f"[VISION] Target '{target}' was not detected.")
            speak(f"I could not find the {target}.")
            return

        confidence = detection["confidence"]
        box = detection["box"]

        print(
            f"[VISION] Target found: {target} | "
            f"confidence={confidence:.2f} | "
            f"box={box}"
        )

        speak(
            f"{target} detected. "
            f"Preparing to grasp."
        )

        arm.grasp(target)

        speak(f"Grasp sequence for the {target} completed.")
        return


# -----------------------------
# 9. MAIN PROGRAM
# -----------------------------

def main():
    print("\n")
    print("=" * 70)
    print("                 VOXGRIP SOFTWARE DEMO")
    print("=" * 70)
    print("Offline STT + Intent Recognition + Object Detection")
    print("+ Simulated Arm Control + TTS")
    print("\nExample commands:")
    print('  "Open the hand"')
    print('  "Close the hand"')
    print('  "Grab the bottle"')
    print('  "Pick up the cup"')
    print('  "Release object"')
    print('  "Rotate wrist left"')
    print('  "Rotate wrist right"')
    print('  "Center wrist"')
    print('  "Stop"')
    print("\nPress Ctrl+C in the terminal to exit.")
    print("=" * 70)

    speak(
        "VoxGrip software demo is ready. "
        "Please give a command."
    )

    while True:
        try:
            audio = record_audio()
            text = speech_to_text(audio)

            print(f"\n[STT] Recognized text: {text if text else '[nothing detected]'}")

            if not text:
                speak("I did not hear a command.")
                continue

            command = parse_intent(text)

            execute_command(command)

        except KeyboardInterrupt:
            print("\n\n[VoxGrip] Program stopped by user.")
            break

        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            speak("An error occurred. Please check the terminal.")


if __name__ == "__main__":
    main()
