#!/usr/bin/env python3
"""
Janet V13.30 - object memory, object tab, face sample delete, return-home object scan
- OAK-D (FRONT)
- PiCam (REAR) - Index optimization
- Dual-view layout with "Slightly Smaller Rear-Mirror" PIP
- Side-panel for Sensors and Detections
- Motor control via I2C
- Color-coded HC-SR04 sonar distance readings
- Detection list in sidebar
"""

import time
import threading
import signal
import sys
import os
import logging
import queue
import subprocess
import re
import wave
import math
import shutil
import io
import random
import json

# Reduce noisy OpenCV/V4L warnings before cv2 is used.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import depthai as dai
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    GPIO = None

# ----------------------------
# Motor control via Arduino/I2C
# ----------------------------
# V13 keeps the V10 camera/HTML code intact and only makes motor control
# self-contained. This avoids relying on a separate motors_i2c.py file.
try:
    import smbus
    I2C_AVAILABLE = True
except ImportError:
    try:
        import smbus2 as smbus
        I2C_AVAILABLE = True
    except ImportError:
        smbus = None
        I2C_AVAILABLE = False

MOTORS_AVAILABLE = I2C_AVAILABLE

MOTOR_I2C_BUS = 1
MOTOR_I2C_ADDR = 0x08
MOTOR_DEFAULT_DURATION = 0.25
MOTOR_DEFAULT_ACCELERATION = 0.0
MOTOR_MIN_DURATION = 0.05
MOTOR_MAX_DURATION = 3.0
MOTOR_MIN_ACCELERATION = 0.0
MOTOR_MAX_ACCELERATION = 1.0
MOTOR_PRESETS = {
    "soft": {"duration": 0.35, "acceleration": 0.75, "label": "Soft acceleration"},
    "normal": {"duration": 0.25, "acceleration": 0.25, "label": "Normal"},
    "racing": {"duration": 0.15, "acceleration": 0.0, "label": "Racing mode"},
}

# ----------------------------
# Motor dance routines
# ----------------------------
# Cable-safe motor routines. Each routine runs for 20 seconds and is
# home-balanced: complete cycles always contain matching forward/backward
# steps and matching left/right steps. Janet should therefore finish back
# near her start point, which is important while she is still on cables.
MOTOR_DANCE_DURATION_SECONDS = 20.0
MOTOR_DANCE_STEP_GAP = 0.08
MOTOR_DANCE_MAX_CONSECUTIVE_STEPS = 5
MOTOR_DANCE_RETURN_HOME = True
MOTOR_DANCE_ROUTINES = {
    "cable_wiggle": {
        "name": "Cable Wiggle",
        "emoji": "〰️",
        "description": "Small forward/back steps with gentle left/right wiggles; returns to start.",
        "pattern": ["forward", "backward", "left", "right", "left", "right", "backward", "forward"],
    },
    "box_shuffle": {
        "name": "Box Shuffle",
        "emoji": "◼️",
        "description": "A tiny box-shaped shuffle: forward, right, back, left; returns to start.",
        "pattern": ["forward", "right", "backward", "left", "forward", "left", "backward", "right"],
    },
    "happy_bounce": {
        "name": "Happy Bounce",
        "emoji": "🤖",
        "description": "A playful bounce with short forward/back steps and quick turns; returns to start.",
        "pattern": ["forward", "backward", "forward", "backward", "left", "right", "left", "right"],
    },
    "disco_twist": {
        "name": "Disco Twist",
        "emoji": "🕺",
        "description": "More turning than travelling, with short cable-safe twists; returns to start.",
        "pattern": ["left", "right", "left", "right", "forward", "backward", "right", "left"],
    },
    "rock_music": {
        "name": "Rock Music",
        "emoji": "🎸",
        "description": "Punchy head-bang style moves: quick forward/back hits and left/right snaps; returns to start.",
        "pattern": ["forward", "backward", "forward", "backward", "left", "right", "right", "left"],
    },
    "salsa_dance": {
        "name": "Salsa Dance",
        "emoji": "💃",
        "description": "Side-to-side salsa feel with small forward/back checks; returns to start.",
        "pattern": ["left", "right", "forward", "backward", "right", "left", "forward", "backward"],
    },
    "waltz": {
        "name": "Waltz",
        "emoji": "🎻",
        "description": "Gentle box-step rhythm: forward, side, back, side; returns to start.",
        "pattern": ["forward", "left", "backward", "right", "forward", "right", "backward", "left"],
    },
    "dance": {
        "name": "Dance",
        "emoji": "🎶",
        "description": "A simple all-round dance mix using all four directions; returns to start.",
        "pattern": ["forward", "backward", "left", "right", "right", "left", "backward", "forward"],
    },
    "drum_and_bass": {
        "name": "Drum and Bass",
        "emoji": "🥁",
        "description": "Fast twitchy turns and short bassline bounces; returns to start.",
        "pattern": ["left", "right", "left", "right", "forward", "backward", "forward", "backward", "right", "left", "backward", "forward"],
    },
}

# Set this to False if the arrows are backwards on the robot.
# Your V9/V11 had the physical wiring reversed, so V13 defaults to the fixed mapping.
REVERSE_MOTOR_LOGIC = True

# Rear camera handling. Your log shows /dev/video14 timeouts and /dev/video1/2
# are not capture devices. Leaving rear probing enabled makes the page pause and
# floods the terminal every time the browser opens/reloads.
# Set REAR_CAMERA_ENABLED = True once the rear USB/Pi camera is connected and
# put the known working index first in REAR_CAMERA_INDEXES.
REAR_CAMERA_ENABLED = False
REAR_CAMERA_INDEXES = [0, 14, 1, 2]
REAR_CAMERA_WIDTH = 640
REAR_CAMERA_HEIGHT = 480

# ----------------------------
# Detection / model info
# ----------------------------
# Keep the known-good V10 model as the default. The Detection tab can inspect
# this setup and scan local model caches/candidate names, but it does not
# hot-swap the running OAK-D pipeline. That keeps the video feed stable.
DETECTION_MODEL_NAME = "yolov6-nano"
DETECTION_CONFIDENCE = 0.50
DETECTION_MODEL_FAMILY = "YOLOv6"
DETECTION_MODEL_PURPOSE = "Fast general object detection on OAK-D"
DETECTION_RUNTIME = "DepthAI DetectionNetwork"
DETECTION_MODEL_CANDIDATES = [
    {"name": "yolov6-nano", "type": "known-good", "note": "Current stable Janet model; fastest/safest baseline."},
    {"name": "yolov6-tiny", "type": "candidate", "note": "Possible next step if you want more accuracy with modest extra load."},
    {"name": "yolov6-s", "type": "candidate", "note": "Potentially better accuracy, but may reduce FPS on Raspberry Pi/OAK-D."},
    {"name": "yolov8n", "type": "candidate", "note": "Modern nano detector candidate if available in your DepthAI model source/cache."},
    {"name": "yolov8s", "type": "candidate", "note": "Potentially stronger detector; test only after backing up this working version."},
    {"name": "yolo11n", "type": "candidate", "note": "Newer YOLO-family nano candidate if supported by your installed DepthAI tools."},
]
DETECTION_CACHE_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),
    os.path.expanduser("~/.cache"),
    os.path.expanduser("~/.depthai"),
    os.path.expanduser("~/.luxonis"),
    os.path.expanduser("~/depthai"),
]

# ----------------------------
# Face recognition
# ----------------------------
# Lightweight local face recognition using OpenCV only. It stores small
# normalised face templates in known_faces/known_faces.npz, so there is no
# heavy dlib/face_recognition dependency to install on the Pi.
FACE_ENABLED = True
FACE_RECOGNITION_ENABLED = True
FACE_DETECTION_EVERY_N_FRAMES = 4
FACE_RECOGNITION_THRESHOLD = 55.0
FACE_TEMPLATE_SIZE = (80, 80)
FACE_MIN_SIZE = (48, 48)
FACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
FACE_DB_FILE = os.path.join(FACE_DIR, "known_faces.npz")
FACE_IMAGE_DIR = os.path.join(FACE_DIR, "images")
FACE_OVERLAY_COLOR_KNOWN = (87, 242, 135)
FACE_OVERLAY_COLOR_UNKNOWN = (255, 204, 102)

# Face greeting routines. Janet only performs a greeting if she has not seen
# that specific known person for FACE_GREETING_COOLDOWN_SECONDS.
FACE_GREETING_ENABLED = False
FACE_GREETING_COOLDOWN_SECONDS = 10 * 60
FACE_GREETING_STEP_GAP = 0.08
FACE_GREETING_ROUTINES = {
    "nico": {
        "display": "Nico",
        "role": "Primary user",
        "description": "move back 0.5s, then forward 0.5s",
        "steps": [("backward", 0.5), ("forward", 0.5)],
    },
    "paul": {
        "display": "Paul",
        "role": "Secondary user",
        "description": "turn right a little, then left a little",
        "steps": [("right", 0.25), ("left", 0.25)],
    },
}

# Spoken face greetings. This is deliberately separate from the Nico/Paul
# motor greeting so Janet can say hello to any known face without needing a
# special motor routine for that person. Cooldown is based on when that face
# was last seen, so she says hello when someone arrives/returns, not every
# recognition frame.
FACE_SPEECH_GREETING_ENABLED = True
FACE_SPEECH_GREETING_COOLDOWN_SECONDS = 10 * 60
FACE_SPEECH_GREETING_TEMPLATE = "Hi {name}"

# Face/object centering and object speech acknowledgement.
# Only left/right turns are used here: no forward/back movement.
TARGET_CENTERING_ENABLED = True
TARGET_CENTERING_DEADBAND = 0.12          # 12% either side of centre = good enough
TARGET_CENTERING_TURN_DURATION = 0.16     # small cable-safe left/right turn
TARGET_CENTERING_COOLDOWN_SECONDS = 1.8
TARGET_CENTERING_MIN_CONFIDENCE = 50

OBJECT_SPEECH_GREETING_ENABLED = True
OBJECT_SPEECH_GREETING_COOLDOWN_SECONDS = 30 * 60
OBJECT_SPEECH_GREETING_TEMPLATE = "Hi {name}"
OBJECT_SPEECH_SKIP_LABELS = {"person"}
OBJECT_CENTERING_ENABLED = True

# Object memory. When Janet acknowledges an object she saves a cropped
# thumbnail, then treats that label as known/acknowledged so YOLO jitter does
# not make her keep recognising the same object over and over.
OBJECT_MEMORY_ENABLED = True
OBJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_objects")
OBJECT_DB_FILE = os.path.join(OBJECT_DIR, "known_objects.json")
OBJECT_IMAGE_DIR = os.path.join(OBJECT_DIR, "images")
OBJECT_MAX_SAMPLES_PER_LABEL = 12
OBJECT_ROOM_SCAN_IDLE_RETURN_SECONDS = 8.0
OBJECT_ROOM_SCAN_MAX_RETURN_SECONDS = 2.0
OBJECT_CENTERING_TARGET_SECONDS = 6.0

# ----------------------------
# Voice control
# ----------------------------
VOICE_ENABLED = True
VOICE_WAKE_WORD = "janet"
VOICE_FORWARD_PHRASES = ("move forward", "forward", "go forward")
VOICE_BACKWARD_PHRASES = ("move back", "move backward", "backward", "go back")
VOICE_LEFT_PHRASES = ("turn left", "left")
VOICE_RIGHT_PHRASES = ("turn right", "right")
VOICE_STOP_PHRASES = ("stop", "halt")
VOICE_LISTEN_TIMEOUT = 3
VOICE_PHRASE_TIME_LIMIT = 4
VOICE_ARECORD_DEVICE = "auto"
VOICE_ARECORD_RATE = "16000"
VOICE_ARECORD_CHANNELS = "1"
VOICE_AMBIENT_ADJUST_SECONDS = 0.6
VOICE_MIN_FRONT_DISTANCE_CM = 20

VOICE_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_samples")
VOICE_SAMPLE_DEFAULT_DURATION = 5
VOICE_SAMPLE_MAX_DURATION = 30
VOICE_WAKE_OVERLAY_SECONDS = 2.5
# Software boost applied before speech recognition. Useful when cheap USB mics
# have no ALSA Capture/Mic gain control exposed to amixer.
VOICE_RECOGNITION_GAIN = 3.0
# V13.12 also normalises every recognition recording towards this loudness.
# This helps when the mic level is visible but Google hears muffled/quiet speech.
VOICE_TARGET_DB = -18.0
VOICE_MAX_AUTO_GAIN = 12.0
VOICE_SOUND_LEVEL_THRESHOLD = 8
VOICE_SOUND_OVERLAY_SECONDS = 1.5
VOICE_DEBUG_KEEP_FILES = True
VOICE_LAST_RAW_FILE = "janet_last_voice_raw.wav"
VOICE_LAST_BOOSTED_FILE = "janet_last_voice_boosted.wav"

# ----------------------------
# Speech / speaker output
# ----------------------------
# Voice input is handled by arecord. Speech output is separate and uses ALSA
# playback via aplay, so Janet can test USB speakers without PyAudio.
SPEECH_ENABLED = True
SPEECH_APLAY_DEVICE = "plughw:CARD=Y02,DEV=0"
SPEECH_APLAY_RATE = "16000"
SPEECH_APLAY_CHANNELS = "1"
# Your working USB speaker/microphone shows up as: card 3: Y02 [BY Y02].
# Use the stable ALSA card name rather than a card number, because card numbers
# can change when USB devices are replugged.
SPEECH_WORKING_USB_DEVICE = "plughw:CARD=Y02,DEV=0"
SPEECH_WORKING_USB_CARD = "Y02"
SPEECH_AUTO_SET_VOLUME = True
SPEECH_VOLUME_PERCENT = 85
SPEECH_VOLUME_CONTROLS = ("Master", "Speaker", "PCM", "Headphone", "Headphones", "Playback")
SPEECH_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_samples")
SPEECH_TEST_FILE = "janet_speaker_test.wav"
SPEECH_TTS_FILE = "janet_speech_test.wav"
SPEECH_TEST_PHRASE = "Hello, I am Janet. My speaker is working."
# V13.29: now that espeak-ng is installed, prefer the real command-line TTS
# engine and generate a WAV first, then play that WAV through the known-good
# Y02 ALSA device. These settings make Janet sound more like a voice and less
# like the old beep/boop fallback.
SPEECH_PREFERRED_TTS_COMMANDS = ("espeak-ng", "espeak")
SPEECH_TTS_VOICE = "en-gb"
SPEECH_TTS_SPEED = "145"
SPEECH_TTS_AMPLITUDE = "170"
# If espeak/espeak-ng is not available, Janet will generate a small built-in
# robotic voice WAV instead of using the old single beep fallback for greetings.
SPEECH_ROBOT_VOICE_FALLBACK = True
SPEECH_ROBOT_VOICE_RATE = 16000
SPEECH_ROBOT_VOICE_FILE = "janet_robot_voice.wav"
# Prefer the USB speaker/mic dongle over Raspberry Pi HDMI playback. The Pi
# often lists HDMI as vc4hdmi0, which can fail with ALSA Unknown error 524 or
# play to the wrong output.
SPEECH_PREFER_KEYWORDS = ("y02", "by y02", "usb", "device", "speaker", "headset", "headphone", "audio", "snd", "uac")
SPEECH_AVOID_KEYWORDS = ("vc4hdmi", "hdmi", "bcm2835", "vc4", "hdmi0", "hdmi1")

# ----------------------------
# Screenshots / status reports / front camera worker
# ----------------------------
# V13.17 starts the OAK-D detection pipeline at application startup, not only
# when a browser opens /video_front. The web stream and screenshot endpoint both
# read from the latest annotated frame produced by this background worker.
FRONT_CAMERA_AUTO_START = True
FRONT_CAMERA_RESTART_DELAY = 2.0
FRONT_CAMERA_PLACEHOLDER_WIDTH = 640
FRONT_CAMERA_PLACEHOLDER_HEIGHT = 480
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
SCREENSHOT_PREFIX = "janet_front"


# Quieter Flask request logging; /readings is requested often by the dashboard.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

try:
    cv2.setLogLevel(0)
except Exception:
    pass

class MotorControllerI2C:
    def __init__(self, bus_num=MOTOR_I2C_BUS, addr=MOTOR_I2C_ADDR, default_duration=MOTOR_DEFAULT_DURATION):
        if not I2C_AVAILABLE:
            raise RuntimeError("smbus/smbus2 is not installed")
        self.bus = smbus.SMBus(bus_num)
        self.addr = addr
        self.default_duration = default_duration
        self.lock = threading.Lock()

    def send_command(self, command, duration=None):
        if duration is None:
            duration = self.default_duration
        duration_ms = max(0, min(int(duration * 1000), 65535))
        payload = [duration_ms & 0xFF, (duration_ms >> 8) & 0xFF]
        with self.lock:
            self.bus.write_i2c_block_data(self.addr, command, payload)

    def forward(self, duration=None): self.send_command(0x01, duration)
    def backward(self, duration=None): self.send_command(0x02, duration)
    def left(self, duration=None): self.send_command(0x03, duration)
    def right(self, duration=None): self.send_command(0x04, duration)
    def stop(self, duration=None): self.send_command(0x00, duration)

# ----------------------------
# Sonar GPIO pin configuration
# ----------------------------
TRIG_PINS = {"front": 13, "back": 16, "left": 29, "right": 33}
ECHO_PINS = {"front": 15, "back": 18, "left": 31, "right": 35}

# ----------------------------
# Flask app and shared state
# ----------------------------
app = Flask(__name__)

latest_readings = {
    "fps": 0,
    "detections": [],
    "sonar": {"front": -1, "back": -1, "left": -1, "right": -1},
    "sonar_available": GPIO_AVAILABLE,
    "status": "starting",
    "detection_model": {
        "name": DETECTION_MODEL_NAME,
        "family": DETECTION_MODEL_FAMILY,
        "purpose": DETECTION_MODEL_PURPOSE,
        "runtime": DETECTION_RUNTIME,
        "confidence": DETECTION_CONFIDENCE,
        "labels_count": 0,
        "labels_preview": [],
        "last_scan_message": "Not scanned yet",
        "scan_results": [],
    },
    "face": {
        "available": False,
        "enabled": FACE_RECOGNITION_ENABLED,
        "status": "starting",
        "known_faces": [],
        "known_count": 0,
        "last_seen": [],
        "threshold": FACE_RECOGNITION_THRESHOLD,
        "last_message": "Face system starting",
        "cascade": "haarcascade_frontalface_default.xml",
        "storage": FACE_DB_FILE,
        "greeting_enabled": FACE_GREETING_ENABLED,
        "greeting_cooldown_seconds": FACE_GREETING_COOLDOWN_SECONDS,
        "greeting_status": "ready",
        "last_greeting": "",
        "last_greeting_person": "",
        "last_greeting_time": None,
        "speech_greeting_enabled": FACE_SPEECH_GREETING_ENABLED,
        "speech_greeting_cooldown_seconds": FACE_SPEECH_GREETING_COOLDOWN_SECONDS,
        "speech_greeting_status": "ready",
        "last_speech_greeting": "",
        "last_speech_greeting_person": "",
        "last_speech_greeting_time": None,
    },
    "voice": {
        "available": SPEECH_AVAILABLE,
        "enabled": VOICE_ENABLED,
        "status": "starting",
        "last_heard": "",
        "last_action": "",
        "last_error": "",
        "wake_word_active": False,
        "sample_status": "idle",
        "sample_file": "",
        "sample_message": "",
        "mic_level": 0,
        "mic_level_db": None,
        "voice_device": VOICE_ARECORD_DEVICE,
        "last_test_text": "",
        "last_test_message": "",
        "arecord_devices": "",
        "recognition_gain": VOICE_RECOGNITION_GAIN,
        "target_db": VOICE_TARGET_DB,
        "sound_active": False,
        "last_sound_message": "",
        "last_raw_file": "",
        "last_boosted_file": "",
    },
    "speech": {
        "available": True,
        "enabled": SPEECH_ENABLED,
        "status": "starting",
        "speaker_device": SPEECH_APLAY_DEVICE,
        "last_message": "Speech output starting",
        "last_error": "",
        "last_phrase": "",
        "last_file": "",
        "aplay_devices": "",
        "espeak_available": bool(shutil.which("espeak-ng") or shutil.which("espeak")),
        "tts_engine": "espeak-ng preferred",
        "object_greeting_status": "ready",
        "last_object_greeting": "",
        "last_object_greeting_label": "",
        "last_object_greeting_time": None,
        "centering_status": "ready",
        "last_centering_action": "",
    },
    "object": {
        "available": OBJECT_MEMORY_ENABLED,
        "enabled": OBJECT_MEMORY_ENABLED,
        "status": "starting",
        "known_objects": [],
        "known_count": 0,
        "samples_count": 0,
        "last_seen": [],
        "last_message": "Object memory starting",
        "storage": OBJECT_DB_FILE,
        "image_dir": OBJECT_IMAGE_DIR,
        "acknowledge_cooldown_seconds": OBJECT_SPEECH_GREETING_COOLDOWN_SECONDS,
        "scan_status": "ready",
        "last_scan_return": "",
    },
    "routines": {
        "available": True,
        "active": False,
        "current_id": "",
        "current_name": "",
        "status": "ready",
        "last_message": "Motor routines ready",
        "last_completed": "",
        "elapsed_seconds": 0.0,
        "target_seconds": MOTOR_DANCE_DURATION_SECONDS,
        "step_duration": MOTOR_DEFAULT_DURATION,
        "steps_done": 0,
        "routines": [],
    },
}

lock = threading.Lock()
shutdown_event = threading.Event()
motors = None
motor_settings = {
    "duration": MOTOR_DEFAULT_DURATION,
    "acceleration": MOTOR_DEFAULT_ACCELERATION,
    "preset": "normal",
}
motor_settings_lock = threading.Lock()
motor_run_lock = threading.Lock()
motor_stop_event = threading.Event()
voice_command_queue = queue.Queue(maxsize=5)
voice_status_lock = threading.Lock()
voice_recording_event = threading.Event()
voice_capture_lock = threading.Lock()
voice_process_lock = threading.Lock()
voice_live_capture_proc = None
voice_wake_overlay_until = 0.0
voice_sound_overlay_until = 0.0
voice_config_lock = threading.Lock()
voice_config = {
    "device": VOICE_ARECORD_DEVICE,
    "rate": VOICE_ARECORD_RATE,
    "channels": VOICE_ARECORD_CHANNELS,
}

speech_status_lock = threading.Lock()
speech_playback_lock = threading.Lock()
speech_config_lock = threading.Lock()
speech_config = {
    "device": SPEECH_APLAY_DEVICE,
    "rate": SPEECH_APLAY_RATE,
    "channels": SPEECH_APLAY_CHANNELS,
}

face_lock = threading.Lock()
face_last_frame = None
face_detector = None
face_db = {"names": [], "templates": np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32), "images": []}
face_settings = {
    "enabled": FACE_RECOGNITION_ENABLED,
    "threshold": FACE_RECOGNITION_THRESHOLD,
}
last_face_results = []
face_greeting_last_seen = {}
face_greeting_pending = set()
face_greeting_lock = threading.Lock()
face_greeting_queue = queue.Queue(maxsize=10)
face_speech_greeting_last_seen = {}
face_speech_greeting_pending = set()
face_speech_greeting_lock = threading.Lock()
face_speech_greeting_queue = queue.Queue(maxsize=10)
object_speech_greeting_last_seen = {}
object_speech_greeting_pending = set()
object_speech_greeting_lock = threading.Lock()
object_speech_greeting_queue = queue.Queue(maxsize=20)
object_lock = threading.Lock()
object_db = {"samples": []}
object_scan_lock = threading.Lock()
object_scan_state = {
    "active": False,
    "net_turn_seconds": 0.0,
    "last_new_object_time": 0.0,
    "last_return_time": 0.0,
    "seen_keys": {},
    "current_target_key": "",
    "current_target_label": "",
    "target_until": 0.0,
}
target_centering_lock = threading.Lock()
target_centering_last_turn = {}

routine_lock = threading.Lock()
routine_stop_event = threading.Event()
routine_thread = None
routine_state = {
    "active": False,
    "current_id": "",
    "current_name": "",
    "started_at": None,
    "steps_done": 0,
}

front_camera_thread = None
front_camera_start_lock = threading.Lock()
front_camera_started = threading.Event()
front_camera_ready = threading.Event()
front_camera_error = ""
front_stream_condition = threading.Condition()
front_latest_jpeg = None
front_latest_frame_id = 0
front_latest_frame_time = 0.0
front_latest_annotated_frame = None



def update_face_status(status=None, known_faces=None, known_count=None, last_seen=None, threshold=None, last_message=None, available=None, enabled=None, greeting_status=None, last_greeting=None, last_greeting_person=None, last_greeting_time=None, speech_greeting_status=None, last_speech_greeting=None, last_speech_greeting_person=None, last_speech_greeting_time=None):
    with lock:
        face = latest_readings.setdefault("face", {})
        if available is not None:
            face["available"] = bool(available)
        if enabled is not None:
            face["enabled"] = bool(enabled)
        if status is not None:
            face["status"] = status
        if known_faces is not None:
            face["known_faces"] = known_faces
        if known_count is not None:
            face["known_count"] = int(known_count)
        if last_seen is not None:
            face["last_seen"] = last_seen
        if threshold is not None:
            face["threshold"] = float(threshold)
        if last_message is not None:
            face["last_message"] = last_message
        if greeting_status is not None:
            face["greeting_status"] = greeting_status
        if last_greeting is not None:
            face["last_greeting"] = last_greeting
        if last_greeting_person is not None:
            face["last_greeting_person"] = last_greeting_person
        if last_greeting_time is not None:
            face["last_greeting_time"] = last_greeting_time
        if speech_greeting_status is not None:
            face["speech_greeting_status"] = speech_greeting_status
        if last_speech_greeting is not None:
            face["last_speech_greeting"] = last_speech_greeting
        if last_speech_greeting_person is not None:
            face["last_speech_greeting_person"] = last_speech_greeting_person
        if last_speech_greeting_time is not None:
            face["last_speech_greeting_time"] = last_speech_greeting_time
        face["greeting_enabled"] = FACE_GREETING_ENABLED
        face["greeting_cooldown_seconds"] = FACE_GREETING_COOLDOWN_SECONDS
        face["speech_greeting_enabled"] = FACE_SPEECH_GREETING_ENABLED
        face["speech_greeting_cooldown_seconds"] = FACE_SPEECH_GREETING_COOLDOWN_SECONDS
        face["storage"] = FACE_DB_FILE


def init_face_system():
    global face_detector
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    except Exception:
        cascade_path = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        update_face_status(status="cascade unavailable", available=False, last_message="OpenCV Haar cascade could not be loaded")
        return False
    face_detector = detector
    load_known_faces()
    update_face_status(status="ready", available=True, enabled=face_settings["enabled"], threshold=face_settings["threshold"], last_message="Face recognition ready")
    return True


def face_template_from_roi(frame, box):
    x, y, w, h = [int(v) for v in box]
    h_img, w_img = frame.shape[:2]
    pad = int(max(w, h) * 0.18)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return gray.astype(np.float32).reshape(-1)


def detect_faces_in_frame(frame):
    if face_detector is None:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=FACE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [tuple(map(int, f)) for f in faces]
    except Exception as e:
        update_face_status(status="face detection error", last_message=str(e))
        return []


def safe_face_file_part(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "face")).strip("._")
    return value or "face"


def ensure_face_image_list(names, templates=None, images=None):
    """Keep one stored thumbnail filename per face sample.

    Older Janet face DBs only stored OpenCV templates, so this creates a
    small legacy grayscale thumbnail for those old samples. New samples save
    a colour crop from the camera frame.
    """
    os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
    names = list(names or [])
    images = list(images or [])
    while len(images) < len(names):
        images.append("")
    if len(images) > len(names):
        images = images[:len(names)]

    if templates is None:
        templates = np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32)

    for idx, name in enumerate(names):
        filename = images[idx]
        if filename and os.path.exists(os.path.join(FACE_IMAGE_DIR, os.path.basename(filename))):
            images[idx] = os.path.basename(filename)
            continue
        try:
            if templates is not None and idx < len(templates):
                img = templates[idx].reshape(FACE_TEMPLATE_SIZE).astype(np.float32)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_NEAREST)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                filename = f"legacy_{idx:04d}_{safe_face_file_part(name)}.jpg"
                cv2.imwrite(os.path.join(FACE_IMAGE_DIR, filename), img)
                images[idx] = filename
        except Exception:
            images[idx] = ""
    return images


def crop_face_for_storage(frame, box):
    x, y, w, h = [int(v) for v in box]
    h_img, w_img = frame.shape[:2]
    pad = int(max(w, h) * 0.25)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return crop


def save_face_capture_image(frame, box, name, sample_index):
    os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
    crop = crop_face_for_storage(frame, box)
    if crop is None:
        return ""
    crop = cv2.resize(crop, (140, 140), interpolation=cv2.INTER_AREA)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_face_file_part(name)}_{sample_index:04d}.jpg"
    path = os.path.join(FACE_IMAGE_DIR, filename)
    try:
        if cv2.imwrite(path, crop):
            return filename
    except Exception as e:
        print(f"Face thumbnail save failed: {e}")
    return ""


def known_face_summary():
    names = list(face_db.get("names", []))
    templates = face_db.get("templates", np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32))
    images = ensure_face_image_list(names, templates, face_db.get("images", []))
    face_db["images"] = images

    grouped = {}
    for idx, name in enumerate(names):
        grouped.setdefault(name, {"name": name, "samples": 0, "images": []})
        grouped[name]["samples"] += 1
        filename = images[idx] if idx < len(images) else ""
        if filename:
            safe = os.path.basename(filename)
            grouped[name]["images"].append({
                "sample": idx + 1,
                "sample_index": idx,
                "filename": safe,
                "url": f"/face_image/{safe}",
                "thumb_url": f"/face_image/{safe}",
            })
    return [grouped[name] for name in sorted(grouped)]


def load_known_faces():
    global face_db
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
    if not os.path.exists(FACE_DB_FILE):
        face_db = {"names": [], "templates": np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32), "images": []}
        update_face_status(known_faces=[], known_count=0, last_message="No known faces yet")
        return face_db
    try:
        data = np.load(FACE_DB_FILE, allow_pickle=True)
        names = [str(x) for x in data.get("names", np.array([], dtype=object)).tolist()]
        templates = data.get("templates", np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32)).astype(np.float32)
        if templates.ndim == 1 and templates.size:
            templates = templates.reshape(1, -1)
        images = [str(x) for x in data.get("images", np.array([], dtype=object)).tolist()]
        images = ensure_face_image_list(names, templates, images)
        face_db = {"names": names, "templates": templates, "images": images}
        # Upgrade older DBs by adding the image list without changing templates.
        try:
            np.savez_compressed(FACE_DB_FILE, names=np.array(names, dtype=object), templates=templates, images=np.array(images, dtype=object))
        except Exception:
            pass
        update_face_status(known_faces=known_face_summary(), known_count=len(names), last_message=f"Loaded {len(names)} known face sample(s)")
    except Exception as e:
        face_db = {"names": [], "templates": np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32), "images": []}
        update_face_status(status="face db load error", known_faces=[], known_count=0, last_message=str(e))
    return face_db


def save_known_faces():
    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(FACE_IMAGE_DIR, exist_ok=True)
    raw_names = list(face_db.get("names", []))
    templates = face_db.get("templates", np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32)).astype(np.float32)
    images = ensure_face_image_list(raw_names, templates, face_db.get("images", []))
    face_db["images"] = images
    names = np.array(raw_names, dtype=object)
    np.savez_compressed(FACE_DB_FILE, names=names, templates=templates, images=np.array(images, dtype=object))
    update_face_status(known_faces=known_face_summary(), known_count=len(names))


def recognise_face_template(template):
    names = face_db.get("names", [])
    templates = face_db.get("templates")
    if template is None or templates is None or len(names) == 0 or templates.size == 0:
        return "Unknown", None
    diffs = np.mean(np.abs(templates - template.reshape(1, -1)), axis=1)
    best_idx = int(np.argmin(diffs))
    best_score = float(diffs[best_idx])
    threshold = float(face_settings.get("threshold", FACE_RECOGNITION_THRESHOLD))
    if best_score <= threshold:
        return names[best_idx], round(best_score, 1)
    return "Unknown", round(best_score, 1)


def _normalise_face_greeting_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _format_seconds_ago(seconds):
    try:
        seconds = int(seconds)
    except Exception:
        return "never"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    return f"{minutes // 60}h {minutes % 60}m ago"


def trigger_face_greeting_if_needed(name):
    """Queue a small physical hello for selected known faces.

    Cooldown is based on when Janet last recognised each person. That means
    she greets Nico/Paul when they come back after being unseen for 10 minutes,
    but she will not keep repeating the routine while they remain in view.
    """
    if not FACE_GREETING_ENABLED:
        return
    key = _normalise_face_greeting_name(name)
    if not key or key == "unknown" or key not in FACE_GREETING_ROUTINES:
        return

    now = time.time()
    routine = FACE_GREETING_ROUTINES[key]
    display = routine.get("display", name)

    with face_greeting_lock:
        previous_seen = face_greeting_last_seen.get(key)
        face_greeting_last_seen[key] = now
        if previous_seen is not None and (now - previous_seen) < FACE_GREETING_COOLDOWN_SECONDS:
            remaining = int(FACE_GREETING_COOLDOWN_SECONDS - (now - previous_seen))
            update_face_status(greeting_status=f"{display} recognised; greeting cooldown {remaining}s")
            return
        if key in face_greeting_pending:
            update_face_status(greeting_status=f"{display} greeting already queued")
            return
        face_greeting_pending.add(key)

    try:
        face_greeting_queue.put_nowait((key, display, now))
        msg = f"Queued {display} greeting: {routine.get('description', '')}"
        print(f"[FACE GREETING] {msg}")
        update_face_status(greeting_status=msg)
    except queue.Full:
        with face_greeting_lock:
            face_greeting_pending.discard(key)
        update_face_status(greeting_status="Face greeting queue full")


def face_greeting_worker():
    update_face_status(greeting_status="Face greeting worker ready")
    while not shutdown_event.is_set():
        try:
            key, display, queued_at = face_greeting_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        routine = FACE_GREETING_ROUTINES.get(key)
        try:
            if not routine:
                continue
            if not motors:
                update_face_status(greeting_status=f"Wanted to greet {display}, but motors are unavailable")
                continue

            role = routine.get("role", "known face")
            description = routine.get("description", "greeting routine")
            start_msg = f"Greeting {display} ({role}): {description}"
            print(f"[FACE GREETING] {start_msg}")
            update_face_status(greeting_status=start_msg, last_greeting_person=display)

            for direction, duration in routine.get("steps", []):
                if shutdown_event.is_set():
                    break
                ok, message = execute_move(direction, duration=duration, acceleration=0.0)
                if not ok:
                    update_face_status(greeting_status=f"{display} greeting failed: {message}")
                    break
                time.sleep(float(duration) + FACE_GREETING_STEP_GAP)

            # Safety stop after the short routine.
            try:
                execute_move("stop")
            except Exception:
                pass
            last_msg = f"Last greeting: {display} — {description}"
            update_face_status(
                greeting_status="Face greeting ready",
                last_greeting=last_msg,
                last_greeting_person=display,
                last_greeting_time=time.time(),
            )
        finally:
            with face_greeting_lock:
                face_greeting_pending.discard(key)


def trigger_face_speech_greeting_if_needed(name):
    """Queue Janet to say 'Hi Name' when a known face arrives/returns."""
    if not FACE_SPEECH_GREETING_ENABLED or not SPEECH_ENABLED:
        return
    key = _normalise_face_greeting_name(name)
    if not key or key == "unknown":
        return

    display = str(name or "").strip()
    if not display:
        return

    now = time.time()
    with face_speech_greeting_lock:
        previous_seen = face_speech_greeting_last_seen.get(key)
        face_speech_greeting_last_seen[key] = now
        if previous_seen is not None and (now - previous_seen) < FACE_SPEECH_GREETING_COOLDOWN_SECONDS:
            # Do not update the UI every frame with a cooldown countdown; just avoid spam.
            return
        if key in face_speech_greeting_pending:
            update_face_status(speech_greeting_status=f"Speech hello for {display} already queued")
            return
        face_speech_greeting_pending.add(key)

    try:
        phrase = safe_speech_text(FACE_SPEECH_GREETING_TEMPLATE.format(name=display))
        face_speech_greeting_queue.put_nowait((key, display, phrase, now))
        update_face_status(speech_greeting_status=f"Queued spoken hello: {phrase}")
        print(f"[FACE SPEECH] Queued: {phrase}")
    except queue.Full:
        with face_speech_greeting_lock:
            face_speech_greeting_pending.discard(key)
        update_face_status(speech_greeting_status="Face speech greeting queue full")
    except Exception as e:
        with face_speech_greeting_lock:
            face_speech_greeting_pending.discard(key)
        update_face_status(speech_greeting_status=f"Face speech greeting error: {e}")


def face_speech_greeting_worker():
    update_face_status(speech_greeting_status="Face speech greeting worker ready")
    while not shutdown_event.is_set():
        try:
            key, display, phrase, queued_at = face_speech_greeting_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            print(f"[FACE SPEECH] Speaking to {display}: {phrase}")
            update_face_status(
                speech_greeting_status=f"Saying hello to {display}",
                last_speech_greeting_person=display,
            )
            speech_say_worker(phrase, allow_beep_fallback=False)
            update_face_status(
                speech_greeting_status="Face speech greeting ready",
                last_speech_greeting=phrase,
                last_speech_greeting_person=display,
                last_speech_greeting_time=time.time(),
            )
        finally:
            with face_speech_greeting_lock:
                face_speech_greeting_pending.discard(key)


def _normalise_object_label(label):
    return re.sub(r"[^a-z0-9]+", "", str(label or "").lower())


def pretty_object_label(label):
    label = str(label or "object").strip()
    replacements = {
        "tvmonitor": "TV monitor",
        "pottedplant": "potted plant",
        "diningtable": "dining table",
        "cell phone": "phone",
        "mobile phone": "phone",
        "motorbike": "motorbike",
        "aeroplane": "aeroplane",
    }
    label = label.replace("_", " ").replace("-", " ")
    label = replacements.get(label.lower(), label)
    label = re.sub(r"[^A-Za-z0-9 ]+", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label or "object"



def update_object_status(status=None, known_objects=None, known_count=None, samples_count=None, last_seen=None, last_message=None, available=None, enabled=None, scan_status=None, last_scan_return=None):
    with lock:
        obj = latest_readings.setdefault("object", {})
        if available is not None:
            obj["available"] = bool(available)
        if enabled is not None:
            obj["enabled"] = bool(enabled)
        if status is not None:
            obj["status"] = status
        if known_objects is not None:
            obj["known_objects"] = known_objects
        if known_count is not None:
            obj["known_count"] = int(known_count)
        if samples_count is not None:
            obj["samples_count"] = int(samples_count)
        if last_seen is not None:
            obj["last_seen"] = last_seen
        if last_message is not None:
            obj["last_message"] = last_message
        if scan_status is not None:
            obj["scan_status"] = scan_status
        if last_scan_return is not None:
            obj["last_scan_return"] = last_scan_return
        obj["storage"] = OBJECT_DB_FILE
        obj["image_dir"] = OBJECT_IMAGE_DIR
        obj["acknowledge_cooldown_seconds"] = OBJECT_SPEECH_GREETING_COOLDOWN_SECONDS


def safe_object_file_part(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "object")).strip("._")
    return value or "object"


def load_known_objects():
    global object_db
    os.makedirs(OBJECT_DIR, exist_ok=True)
    os.makedirs(OBJECT_IMAGE_DIR, exist_ok=True)
    if not os.path.exists(OBJECT_DB_FILE):
        object_db = {"samples": []}
        update_object_status(status="ready", known_objects=[], known_count=0, samples_count=0, last_message="No object memories yet")
        return object_db
    try:
        with open(OBJECT_DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", []) if isinstance(data, dict) else []
        cleaned = []
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            label = pretty_object_label(sample.get("label") or sample.get("name") or "object")
            filename = os.path.basename(str(sample.get("filename") or ""))
            if not filename:
                continue
            cleaned.append({
                "id": str(sample.get("id") or f"obj_{i:05d}"),
                "label": label,
                "key": _normalise_object_label(label),
                "filename": filename,
                "timestamp": float(sample.get("timestamp") or 0),
                "confidence": int(sample.get("confidence") or 0),
            })
        object_db = {"samples": cleaned}
        save_known_objects(update_status=False)
        update_object_status(status="ready", known_objects=known_object_summary(), known_count=len({s.get('key') for s in cleaned if s.get('key')}), samples_count=len(cleaned), last_message=f"Loaded {len(cleaned)} object memory sample(s)")
    except Exception as e:
        object_db = {"samples": []}
        update_object_status(status="object db load error", known_objects=[], known_count=0, samples_count=0, last_message=str(e))
    return object_db


def save_known_objects(update_status=True):
    os.makedirs(OBJECT_DIR, exist_ok=True)
    os.makedirs(OBJECT_IMAGE_DIR, exist_ok=True)
    samples = list(object_db.get("samples", []))
    with open(OBJECT_DB_FILE, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, indent=2)
    if update_status:
        keys = {s.get("key") for s in samples if s.get("key")}
        update_object_status(known_objects=known_object_summary(), known_count=len(keys), samples_count=len(samples))


def known_object_summary():
    samples = list(object_db.get("samples", []))
    grouped = {}
    for idx, sample in enumerate(samples):
        label = pretty_object_label(sample.get("label", "object"))
        key = sample.get("key") or _normalise_object_label(label)
        grouped.setdefault(key, {"label": label, "key": key, "samples": 0, "images": []})
        grouped[key]["samples"] += 1
        filename = os.path.basename(str(sample.get("filename") or ""))
        if filename:
            grouped[key]["images"].append({
                "sample": idx + 1,
                "sample_index": idx,
                "id": sample.get("id", f"obj_{idx:05d}"),
                "filename": filename,
                "url": f"/object_image/{filename}",
                "thumb_url": f"/object_image/{filename}",
                "confidence": sample.get("confidence", 0),
                "timestamp": sample.get("timestamp", 0),
            })
    return [grouped[k] for k in sorted(grouped, key=lambda x: grouped[x]["label"].lower())]


def crop_object_for_storage(frame, det):
    if frame is None or det is None:
        return None
    try:
        h_img, w_img = frame.shape[:2]
        xmin = float(det.get("xmin", 0))
        ymin = float(det.get("ymin", 0))
        xmax = float(det.get("xmax", 0))
        ymax = float(det.get("ymax", 0))
        x1 = int(max(0, min(1, xmin)) * w_img)
        y1 = int(max(0, min(1, ymin)) * h_img)
        x2 = int(max(0, min(1, xmax)) * w_img)
        y2 = int(max(0, min(1, ymax)) * h_img)
        pad = int(max(8, max(x2 - x1, y2 - y1) * 0.12))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        return crop
    except Exception:
        return None


def save_object_capture_image(frame, det, label):
    os.makedirs(OBJECT_IMAGE_DIR, exist_ok=True)
    crop = crop_object_for_storage(frame, det)
    if crop is None:
        return ""
    try:
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_object_file_part(label)}.jpg"
    path = os.path.join(OBJECT_IMAGE_DIR, filename)
    try:
        if cv2.imwrite(path, crop):
            return filename
    except Exception as e:
        print(f"Object thumbnail save failed: {e}")
    return ""


def remember_object_sample(label, det=None, frame=None):
    if not OBJECT_MEMORY_ENABLED:
        return False, "Object memory disabled"
    label = pretty_object_label(label)
    key = _normalise_object_label(label)
    if not key or key in OBJECT_SPEECH_SKIP_LABELS:
        return False, "Skipped object label"
    filename = save_object_capture_image(frame, det, label)
    if not filename:
        return False, "Could not save object image"
    try:
        confidence = int(float(det.get("confidence", 0))) if det else 0
    except Exception:
        confidence = 0
    with object_lock:
        samples = list(object_db.get("samples", []))
        # Keep the newest OBJECT_MAX_SAMPLES_PER_LABEL samples per object label.
        same = [i for i, sample in enumerate(samples) if sample.get("key") == key]
        while len(same) >= OBJECT_MAX_SAMPLES_PER_LABEL:
            remove_idx = same.pop(0)
            old = samples.pop(remove_idx)
            try:
                os.remove(os.path.join(OBJECT_IMAGE_DIR, os.path.basename(old.get("filename", ""))))
            except Exception:
                pass
            same = [i for i, sample in enumerate(samples) if sample.get("key") == key]
        sample = {
            "id": f"{int(time.time() * 1000)}_{safe_object_file_part(label)}",
            "label": label,
            "key": key,
            "filename": filename,
            "timestamp": time.time(),
            "confidence": confidence,
        }
        samples.append(sample)
        object_db["samples"] = samples
        save_known_objects(update_status=True)
    msg = f"Saved object sample for {label}"
    update_object_status(status="ready", last_message=msg)
    return True, msg


def remove_known_object_sample(sample_index=None, sample_id=None):
    with object_lock:
        samples = list(object_db.get("samples", []))
        idx = None
        if sample_id:
            for i, sample in enumerate(samples):
                if str(sample.get("id")) == str(sample_id):
                    idx = i
                    break
        if idx is None:
            try:
                idx = int(sample_index)
            except (TypeError, ValueError):
                idx = None
        if idx is None or idx < 0 or idx >= len(samples):
            return False, "Object sample not found"
        sample = samples.pop(idx)
        filename = os.path.basename(str(sample.get("filename") or ""))
        if filename:
            try:
                os.remove(os.path.join(OBJECT_IMAGE_DIR, filename))
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Could not remove object image {filename}: {e}")
        object_db["samples"] = samples
        save_known_objects(update_status=True)
    label = pretty_object_label(sample.get("label", "object"))
    msg = f"Removed one {label} object sample"
    update_object_status(status="ready", last_message=msg)
    return True, msg


def remove_known_object(label):
    label = pretty_object_label(label)
    key = _normalise_object_label(label)
    if not key:
        return False, "No object label provided"
    with object_lock:
        samples = list(object_db.get("samples", []))
        keep = []
        removed = []
        for sample in samples:
            if sample.get("key") == key:
                removed.append(sample)
            else:
                keep.append(sample)
        if not removed:
            return False, f"No known object named {label}"
        for sample in removed:
            filename = os.path.basename(str(sample.get("filename") or ""))
            if filename:
                try:
                    os.remove(os.path.join(OBJECT_IMAGE_DIR, filename))
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Could not remove object image {filename}: {e}")
        object_db["samples"] = keep
        save_known_objects(update_status=True)
    msg = f"Removed {len(removed)} sample(s) for {label}"
    update_object_status(status="ready", last_message=msg)
    return True, msg


def record_object_scan_new_acknowledgement(key, label=""):
    now = time.time()
    with object_scan_lock:
        object_scan_state["active"] = True
        object_scan_state["last_new_object_time"] = now
        object_scan_state["current_target_key"] = key
        object_scan_state["current_target_label"] = label or key
        object_scan_state["target_until"] = now + OBJECT_CENTERING_TARGET_SECONDS
        object_scan_state.setdefault("seen_keys", {})[key] = now
    update_object_status(scan_status=f"Object scan active: facing {label or key}")


def record_object_scan_turn(direction, duration):
    try:
        duration = float(duration)
    except Exception:
        duration = TARGET_CENTERING_TURN_DURATION
    with object_scan_lock:
        if direction == "right":
            object_scan_state["net_turn_seconds"] += duration
        elif direction == "left":
            object_scan_state["net_turn_seconds"] -= duration
        object_scan_state["active"] = True


def maybe_return_object_scan_home():
    if not OBJECT_CENTERING_ENABLED or not motors:
        return
    now = time.time()
    with object_scan_lock:
        if not object_scan_state.get("active"):
            return
        last_new = float(object_scan_state.get("last_new_object_time") or 0)
        if last_new <= 0 or now - last_new < OBJECT_ROOM_SCAN_IDLE_RETURN_SECONDS:
            return
        net = float(object_scan_state.get("net_turn_seconds") or 0.0)
        object_scan_state["active"] = False
        object_scan_state["net_turn_seconds"] = 0.0
        object_scan_state["last_return_time"] = now
        object_scan_state["seen_keys"] = {}
        object_scan_state["current_target_key"] = ""
        object_scan_state["current_target_label"] = ""
        object_scan_state["target_until"] = 0.0
    if abs(net) < 0.03:
        msg = "Object scan complete: Janet is already at her start angle"
        update_object_status(scan_status="room scan complete", last_scan_return=msg)
        update_speech_status(centering_status=msg, last_centering_action=msg)
        return
    direction = "left" if net > 0 else "right"
    duration = min(abs(net), OBJECT_ROOM_SCAN_MAX_RETURN_SECONDS)
    ok, message = execute_move_async(direction, duration=duration, acceleration=0.0)
    if ok:
        msg = f"Object scan complete: returning {direction} for {duration:.2f}s toward start position"
    else:
        msg = f"Object scan wanted to return {direction}, but {message}"
    update_object_status(scan_status="room scan complete", last_scan_return=msg)
    update_speech_status(centering_status=msg, last_centering_action=msg)
    print(f"[OBJECT SCAN] {msg}")


def trigger_object_speech_greeting_if_needed(label, det=None, frame=None):
    """Queue Janet to say 'Hi object' for recognised objects, with a 30-minute per-object cooldown.

    Returns True only when this is a new acknowledgement. Janet saves a cropped
    object thumbnail at the same time, then object centering can use that True
    result to turn toward the object once rather than constantly chasing YOLO jitter.
    """
    if not OBJECT_SPEECH_GREETING_ENABLED or not SPEECH_ENABLED:
        return False
    pretty = pretty_object_label(label)
    key = _normalise_object_label(pretty)
    if not key or key in OBJECT_SPEECH_SKIP_LABELS:
        return False

    now = time.time()
    with object_speech_greeting_lock:
        previous_ack = object_speech_greeting_last_seen.get(key)
        if previous_ack is not None and (now - previous_ack) < OBJECT_SPEECH_GREETING_COOLDOWN_SECONDS:
            return False
        if key in object_speech_greeting_pending:
            update_speech_status(object_greeting_status=f"Object hello for {pretty} already queued")
            return False
        object_speech_greeting_pending.add(key)
        # This timestamp is an acknowledgement time, not every-frame last seen.
        object_speech_greeting_last_seen[key] = now

    try:
        remembered, remember_msg = remember_object_sample(pretty, det=det, frame=frame)
        record_object_scan_new_acknowledgement(key, pretty)
        phrase = safe_speech_text(OBJECT_SPEECH_GREETING_TEMPLATE.format(name=pretty))
        object_speech_greeting_queue.put_nowait((key, pretty, phrase, now))
        extra = f"; {remember_msg}" if remembered else ""
        update_speech_status(object_greeting_status=f"Queued object hello: {phrase}{extra}")
        print(f"[OBJECT SPEECH] Queued: {phrase}{extra}")
        return True
    except queue.Full:
        with object_speech_greeting_lock:
            object_speech_greeting_pending.discard(key)
            # Allow a retry if the queue was full.
            if object_speech_greeting_last_seen.get(key) == now:
                object_speech_greeting_last_seen.pop(key, None)
        update_speech_status(object_greeting_status="Object speech greeting queue full")
        return False
    except Exception as e:
        with object_speech_greeting_lock:
            object_speech_greeting_pending.discard(key)
            if object_speech_greeting_last_seen.get(key) == now:
                object_speech_greeting_last_seen.pop(key, None)
        update_speech_status(object_greeting_status=f"Object speech greeting error: {e}")
        return False

def object_speech_greeting_worker():
    update_speech_status(object_greeting_status="Object speech greeting worker ready")
    while not shutdown_event.is_set():
        try:
            key, label, phrase, queued_at = object_speech_greeting_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            print(f"[OBJECT SPEECH] Speaking to {label}: {phrase}")
            update_speech_status(object_greeting_status=f"Saying hello to {label}", last_object_greeting_label=label)
            speech_say_worker(phrase, allow_beep_fallback=False)
            update_speech_status(
                object_greeting_status="Object speech greeting ready",
                last_object_greeting=phrase,
                last_object_greeting_label=label,
                last_object_greeting_time=time.time(),
            )
        finally:
            with object_speech_greeting_lock:
                object_speech_greeting_pending.discard(key)


def maybe_center_on_target(kind, label, center_x, frame_width):
    """Turn Janet left/right only so a recognised face/object is closer to camera centre."""
    if not TARGET_CENTERING_ENABLED or not motors:
        return
    try:
        frame_width = float(frame_width)
        center_x = float(center_x)
    except (TypeError, ValueError):
        return
    if frame_width <= 1:
        return

    centre_norm = center_x / frame_width
    offset = centre_norm - 0.5
    if abs(offset) <= TARGET_CENTERING_DEADBAND:
        return

    # Do not fight manual dance routines.
    try:
        with routine_lock:
            if routine_state.get("active"):
                return
    except Exception:
        pass

    direction = "right" if offset > 0 else "left"
    now = time.time()
    # Global cooldown for face/object tracking prevents twitching and command floods.
    cooldown_key = f"{kind}:global"
    with target_centering_lock:
        last = target_centering_last_turn.get(cooldown_key, 0)
        if now - last < TARGET_CENTERING_COOLDOWN_SECONDS:
            return
        target_centering_last_turn[cooldown_key] = now

    ok, message = execute_move_async(direction, duration=TARGET_CENTERING_TURN_DURATION, acceleration=0.0)
    if ok and kind == "object":
        record_object_scan_turn(direction, TARGET_CENTERING_TURN_DURATION)
    label = pretty_object_label(label) if kind == "object" else str(label)
    if ok:
        msg = f"Turning {direction} to face {kind}: {label}"
    else:
        msg = f"Wanted to turn {direction} to face {kind}: {label}, but {message}"
    update_speech_status(centering_status=msg, last_centering_action=msg)
    print(f"[CENTERING] {msg}")


def check_object_greetings_and_centering(detections, frame_width, frame=None):
    """Say hello to newly recognised objects, save their photo, then rotate toward that new object.

    Already acknowledged labels are ignored for 30 minutes. This gives Janet a
    simple object memory and suppresses object-recognition jitter/repeats.
    """
    if not detections:
        maybe_return_object_scan_home()
        update_object_status(last_seen=[])
        return
    usable = []
    newly_acknowledged = []
    best_by_label = {}
    for det in detections:
        label = pretty_object_label(det.get("label", ""))
        key = _normalise_object_label(label)
        if not key or key in OBJECT_SPEECH_SKIP_LABELS:
            continue
        try:
            conf = float(det.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0
        if conf < TARGET_CENTERING_MIN_CONFIDENCE:
            continue
        item = (conf, label, det)
        usable.append(item)
        if key not in best_by_label or conf > best_by_label[key][0]:
            best_by_label[key] = item

    last_seen = []
    for conf, label, det in sorted(best_by_label.values(), reverse=True, key=lambda item: item[0]):
        last_seen.append({
            "label": label,
            "confidence": int(conf),
            "xmin": det.get("xmin"),
            "ymin": det.get("ymin"),
            "xmax": det.get("xmax"),
            "ymax": det.get("ymax"),
        })
        if trigger_object_speech_greeting_if_needed(label, det=det, frame=frame):
            newly_acknowledged.append((conf, label, det))

    update_object_status(status="enabled", last_seen=last_seen, known_objects=known_object_summary(), known_count=len({s.get('key') for s in object_db.get('samples', []) if s.get('key')}), samples_count=len(object_db.get('samples', [])))

    # Only centre on newly acknowledged objects. Previously acknowledged objects
    # stay quiet/ignored for 30 minutes, so detection jitter will not make Janet
    # keep re-recognising and re-turning toward the same label.
    if OBJECT_CENTERING_ENABLED and newly_acknowledged:
        newly_acknowledged.sort(reverse=True, key=lambda item: item[0])
        conf, label, det = newly_acknowledged[0]
        try:
            centre_norm = (float(det.get("xmin", 0)) + float(det.get("xmax", 0))) / 2.0
            maybe_center_on_target("object", label, centre_norm * float(frame_width), frame_width)
        except Exception:
            pass
    elif OBJECT_CENTERING_ENABLED:
        # Keep turning briefly toward the object we just acknowledged until it is
        # near centre, but do not speech-acknowledge it again.
        with object_scan_lock:
            target_key = object_scan_state.get("current_target_key", "")
            target_until = float(object_scan_state.get("target_until") or 0.0)
        if target_key and time.time() < target_until and target_key in best_by_label:
            conf, label, det = best_by_label[target_key]
            try:
                centre_norm = (float(det.get("xmin", 0)) + float(det.get("xmax", 0))) / 2.0
                maybe_center_on_target("object", label, centre_norm * float(frame_width), frame_width)
            except Exception:
                pass

    maybe_return_object_scan_home()

def check_face_greetings(results):
    seen_names = set()
    for item in results or []:
        name = str(item.get("name", "")).strip()
        if not name or name.lower() == "unknown":
            continue
        key = _normalise_face_greeting_name(name)
        if key in seen_names:
            continue
        seen_names.add(key)
        trigger_face_speech_greeting_if_needed(name)
        # Old Nico/Paul forward/back motor greetings are disabled in V13.28.
        # Turning to face the recognised person is handled by target centering.
        if FACE_GREETING_ENABLED:
            trigger_face_greeting_if_needed(name)


def process_faces_for_frame(frame):
    if face_detector is None or not face_settings.get("enabled", True):
        update_face_status(enabled=False, last_seen=[])
        return []
    faces = detect_faces_in_frame(frame)
    results = []
    frame_w = frame.shape[1] if frame is not None and len(frame.shape) >= 2 else 0
    for (x, y, w, h) in faces:
        template = face_template_from_roi(frame, (x, y, w, h))
        name, score = recognise_face_template(template)
        result = {
            "name": name,
            "score": score,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
        }
        results.append(result)
        if name and str(name).lower() != "unknown":
            maybe_center_on_target("face", name, x + (w / 2.0), frame_w)
    check_face_greetings(results)
    update_face_status(status="enabled", enabled=True, last_seen=results, threshold=face_settings.get("threshold", FACE_RECOGNITION_THRESHOLD))
    return results


def add_known_face_from_latest_frame(name):
    global face_db
    name = re.sub(r"[^A-Za-z0-9 _.-]+", "", str(name or "")).strip()
    if not name:
        return False, "Please enter a name first"
    with face_lock:
        frame = None if face_last_frame is None else face_last_frame.copy()
    if frame is None:
        return False, "No front camera frame available yet"
    faces = detect_faces_in_frame(frame)
    if not faces:
        return False, "No face found in the current front camera frame"
    # Use the largest detected face so the user can stand in front of Janet and press Add.
    face = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    template = face_template_from_roi(frame, face)
    if template is None:
        return False, "Could not create a face template"
    names = list(face_db.get("names", []))
    templates = face_db.get("templates", np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32))
    images = ensure_face_image_list(names, templates, face_db.get("images", []))
    sample_index = len(names) + 1
    image_file = save_face_capture_image(frame, face, name, sample_index)
    names.append(name)
    images.append(image_file)
    if templates.size == 0:
        templates = template.reshape(1, -1).astype(np.float32)
    else:
        templates = np.vstack([templates.astype(np.float32), template.reshape(1, -1).astype(np.float32)])
    face_db = {"names": names, "templates": templates, "images": images}
    save_known_faces()
    msg = f"Added face sample for {name}. Total samples: {len(names)}"
    update_face_status(status="ready", last_message=msg)
    return True, msg


def remove_known_face(name):
    global face_db
    name = str(name or "").strip()
    if not name:
        return False, "No face name provided"
    names = list(face_db.get("names", []))
    templates = face_db.get("templates", np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32))
    images = ensure_face_image_list(names, templates, face_db.get("images", []))
    keep = [i for i, n in enumerate(names) if n != name]
    remove_indexes = [i for i, n in enumerate(names) if n == name]
    removed = len(names) - len(keep)
    if removed <= 0:
        return False, f"No known face named {name}"
    # Delete image files for removed samples.
    for i in remove_indexes:
        if i < len(images) and images[i]:
            try:
                os.remove(os.path.join(FACE_IMAGE_DIR, os.path.basename(images[i])))
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Could not remove face image {images[i]}: {e}")
    new_names = [names[i] for i in keep]
    new_templates = templates[keep] if len(keep) else np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32)
    new_images = [images[i] for i in keep]
    face_db = {"names": new_names, "templates": new_templates.astype(np.float32), "images": new_images}
    save_known_faces()
    msg = f"Removed {removed} sample(s) for {name}"
    update_face_status(status="ready", last_message=msg)
    return True, msg



def remove_known_face_sample(sample_index=None, sample_id=None):
    """Remove one specific face sample instead of all samples for a person."""
    global face_db
    names = list(face_db.get("names", []))
    templates = face_db.get("templates", np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32))
    images = ensure_face_image_list(names, templates, face_db.get("images", []))
    idx = None
    try:
        idx = int(sample_index)
    except (TypeError, ValueError):
        idx = None
    if idx is None or idx < 0 or idx >= len(names):
        return False, "Face sample not found"
    removed_name = names[idx]
    if idx < len(images) and images[idx]:
        try:
            os.remove(os.path.join(FACE_IMAGE_DIR, os.path.basename(images[idx])))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Could not remove face image {images[idx]}: {e}")
    keep = [i for i in range(len(names)) if i != idx]
    new_names = [names[i] for i in keep]
    new_templates = templates[keep] if len(keep) else np.empty((0, FACE_TEMPLATE_SIZE[0] * FACE_TEMPLATE_SIZE[1]), dtype=np.float32)
    new_images = [images[i] for i in keep]
    face_db = {"names": new_names, "templates": new_templates.astype(np.float32), "images": new_images}
    save_known_faces()
    msg = f"Removed one face sample for {removed_name}"
    update_face_status(status="ready", last_message=msg)
    return True, msg


def set_face_settings(enabled=None, threshold=None):
    if enabled is not None:
        face_settings["enabled"] = bool(enabled)
    if threshold is not None:
        face_settings["threshold"] = clamp_float(threshold, 20.0, 120.0, face_settings.get("threshold", FACE_RECOGNITION_THRESHOLD))
    update_face_status(enabled=face_settings["enabled"], threshold=face_settings["threshold"], last_message="Face settings saved")
    return dict(face_settings)


def detect_arecord_capture_device():
    """Return the best ALSA capture device for Janet's USB microphone.

    On this Raspberry Pi, ALSA lists the new mic/speaker as something like:
    plughw:CARD=Y02,DEV=0. Older Janet tests used CARD=Device.
    The plain 'default' device can fail with: capture slave is not defined.
    """
    # Prefer plughw because it allows ALSA to do safe rate/format conversion.
    try:
        result = subprocess.run(["arecord", "-L"], capture_output=True, text=True, timeout=5)
        output = result.stdout or ""
        plughw_lines = []
        for line in output.splitlines():
            dev = line.strip()
            if dev.startswith("plughw:"):
                plughw_lines.append(dev)
        # New working combined speaker/mic appears as CARD=Y02 / BY Y02.
        for dev in plughw_lines:
            if "CARD=Y02" in dev:
                return dev
        # Older USB mic appeared as CARD=Device. Keep this as fallback.
        for dev in plughw_lines:
            if "CARD=Device" in dev:
                return dev
        if plughw_lines:
            return plughw_lines[0]
    except Exception:
        pass

    # Fallback to numeric card/device from arecord -l.
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        output = result.stdout or ""
        match = re.search(r"card\s+(\d+):.*?device\s+(\d+):", output, re.IGNORECASE)
        if match:
            return f"plughw:{match.group(1)},{match.group(2)}"
    except Exception:
        pass

    return "default"


def normalise_voice_device(device):
    device = str(device or "auto").strip()
    if not device or device.lower() in {"auto", "default"}:
        return detect_arecord_capture_device()
    return device


def get_voice_config():
    with voice_config_lock:
        # Auto-select lazily, so the mic can be plugged in before Janet starts.
        if voice_config.get("device", "auto").lower() in {"auto", "default"}:
            voice_config["device"] = normalise_voice_device(voice_config.get("device"))
        return dict(voice_config)


def set_voice_config(device=None, rate=None, channels=None):
    with voice_config_lock:
        if device is not None:
            voice_config["device"] = normalise_voice_device(device)
        if rate is not None:
            voice_config["rate"] = str(rate).strip() or VOICE_ARECORD_RATE
        if channels is not None:
            voice_config["channels"] = str(channels).strip() or VOICE_ARECORD_CHANNELS
        return dict(voice_config)


def _speaker_candidate_score(device, description=""):
    blob = f"{device} {description}".lower()
    # Strongly prefer the speaker/microphone dongle that worked on Janet: BY Y02.
    if "y02" in blob or "by y02" in blob:
        return 250
    if any(word in blob for word in SPEECH_AVOID_KEYWORDS):
        return -100
    score = 0
    if device.startswith("plughw:"):
        score += 40
    if device.startswith("sysdefault:"):
        score += 20
    if device.startswith("hw:"):
        score += 10
    for word in SPEECH_PREFER_KEYWORDS:
        if word in blob:
            score += 12
    return score


def list_aplay_output_candidates():
    """Return ranked ALSA playback candidates, preferring USB over HDMI."""
    candidates = []
    seen = set()

    def add(device, description="", source=""):
        device = str(device or "").strip()
        if not device or device in seen:
            return
        seen.add(device)
        score = _speaker_candidate_score(device, description)
        candidates.append({
            "device": device,
            "description": str(description or "").strip(),
            "source": source,
            "score": score,
            "avoid": score < 0,
        })

    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        output = result.stdout or ""
        for line in output.splitlines():
            m = re.search(r"card\s+(\d+):\s*([^\[]+)\[([^\]]+)\].*device\s+(\d+):\s*([^\[]+)(?:\[([^\]]+)\])?", line, re.IGNORECASE)
            if not m:
                continue
            card_no, card_short, card_long, dev_no, dev_short, dev_long = m.groups()
            card_short = safe_face_file_part(card_short).replace("_", "") or card_no
            desc = " ".join(x for x in [card_short, card_long, dev_short, dev_long] if x)
            add(f"plughw:CARD={card_short},DEV={dev_no}", desc, "aplay -l")
            add(f"plughw:{card_no},{dev_no}", desc, "aplay -l")
    except Exception:
        pass

    try:
        result = subprocess.run(["aplay", "-L"], capture_output=True, text=True, timeout=5)
        output = result.stdout or ""
        lines = output.splitlines()
        for i, raw in enumerate(lines):
            if not raw.strip() or raw.startswith((" ", "\t")):
                continue
            dev = raw.strip()
            desc = lines[i + 1].strip() if i + 1 < len(lines) and lines[i + 1].startswith((" ", "\t")) else ""
            if dev.startswith(("plughw:", "sysdefault:", "hw:")):
                add(dev, desc, "aplay -L")
    except Exception:
        pass

    candidates.sort(key=lambda item: item.get("score", 0), reverse=True)
    return candidates


def detect_aplay_output_device():
    """Return the best ALSA playback device for Janet's speaker.

    Avoid HDMI/vc4hdmi first, because the robot's speaker is normally the USB
    audio dongle. For this robot, the known working card is Y02 / BY Y02.
    """
    candidates = list_aplay_output_candidates()
    for item in candidates:
        dev = item.get("device", "")
        desc = item.get("description", "")
        blob = f"{dev} {desc}".lower()
        if "card=y02" in blob or "y02" in blob:
            return dev
    for item in candidates:
        if not item.get("avoid"):
            return item["device"]
    if candidates:
        return candidates[0]["device"]
    return SPEECH_WORKING_USB_DEVICE

def normalise_speech_device(device):
    device = str(device or "auto").strip()
    if not device or device.lower() in {"auto", "default"}:
        return detect_aplay_output_device()
    return device


def get_speech_config():
    with speech_config_lock:
        if speech_config.get("device", "auto").lower() in {"auto", "default"}:
            speech_config["device"] = normalise_speech_device(speech_config.get("device"))
        return dict(speech_config)


def set_speech_config(device=None, rate=None, channels=None):
    with speech_config_lock:
        if device is not None:
            speech_config["device"] = normalise_speech_device(device)
        if rate is not None:
            speech_config["rate"] = str(rate).strip() or SPEECH_APLAY_RATE
        if channels is not None:
            speech_config["channels"] = str(channels).strip() or SPEECH_APLAY_CHANNELS
        return dict(speech_config)


def update_speech_status(status=None, speaker_device=None, last_message=None, last_error=None, last_phrase=None, last_file=None, aplay_devices=None, volume_status=None, volume_card=None, volume_percent=None, tts_engine=None, object_greeting_status=None, last_object_greeting=None, last_object_greeting_label=None, last_object_greeting_time=None, centering_status=None, last_centering_action=None):
    with speech_status_lock:
        with lock:
            speech = latest_readings.setdefault("speech", {})
            speech["available"] = True
            speech["enabled"] = SPEECH_ENABLED
            speech["speaker_device"] = get_speech_config().get("device", SPEECH_APLAY_DEVICE)
            speech["espeak_available"] = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
            if status is not None:
                speech["status"] = status
            if speaker_device is not None:
                speech["speaker_device"] = speaker_device
            if last_message is not None:
                speech["last_message"] = last_message
            if last_error is not None:
                speech["last_error"] = last_error
            if last_phrase is not None:
                speech["last_phrase"] = last_phrase
            if last_file is not None:
                speech["last_file"] = last_file
            if aplay_devices is not None:
                speech["aplay_devices"] = aplay_devices
            if volume_status is not None:
                speech["volume_status"] = volume_status
            if volume_card is not None:
                speech["volume_card"] = volume_card
            if volume_percent is not None:
                speech["volume_percent"] = volume_percent
            if tts_engine is not None:
                speech["tts_engine"] = tts_engine
            if object_greeting_status is not None:
                speech["object_greeting_status"] = object_greeting_status
            if last_object_greeting is not None:
                speech["last_object_greeting"] = last_object_greeting
            if last_object_greeting_label is not None:
                speech["last_object_greeting_label"] = last_object_greeting_label
            if last_object_greeting_time is not None:
                speech["last_object_greeting_time"] = last_object_greeting_time
            if centering_status is not None:
                speech["centering_status"] = centering_status
            if last_centering_action is not None:
                speech["last_centering_action"] = last_centering_action


def safe_speech_text(text):
    text = str(text or SPEECH_TEST_PHRASE).strip()
    # Keep command-line TTS safe and short.
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:240] or SPEECH_TEST_PHRASE


def speech_card_for_device(device=None):
    """Return an amixer/aplay card id for the selected speaker device."""
    device = str(device or get_speech_config().get("device") or SPEECH_WORKING_USB_DEVICE)
    m = re.search(r"CARD=([^,\s]+)", device)
    if m:
        return m.group(1)
    m = re.search(r"(?:plughw|hw):(\d+)(?:,|$)", device)
    if m:
        return m.group(1)
    if "Y02" in device or "y02" in device.lower():
        return SPEECH_WORKING_USB_CARD
    # Last resort: find the Y02 card by name from aplay -l.
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in (result.stdout or "").splitlines():
            m = re.search(r"card\s+(\d+):\s*Y02\b", line, re.IGNORECASE)
            if m:
                return m.group(1)
    except Exception:
        pass
    return ""


def set_speaker_volume(percent=None, device=None, quiet=False):
    """Unmute and raise the selected speaker card volume.

    This fixes the easy-to-miss case where the USB speaker works but ALSA mixer
    volume is at 0%. It only touches the selected speaker card, normally Y02.
    """
    percent = int(clamp_float(percent, 0, 100, SPEECH_VOLUME_PERCENT))
    device = device or get_speech_config().get("device") or SPEECH_WORKING_USB_DEVICE
    card = speech_card_for_device(device)
    if not card:
        msg = "Could not find ALSA card for speaker volume"
        if not quiet:
            update_speech_status(status="volume error", last_error=msg, volume_status=msg)
        return False, msg

    if not shutil.which("amixer"):
        msg = "amixer not found; install alsa-utils"
        if not quiet:
            update_speech_status(status="volume error", last_error=msg, volume_status=msg, volume_card=card)
        return False, msg

    try:
        result = subprocess.run(["amixer", "-c", str(card), "scontrols"], capture_output=True, text=True, timeout=5)
        controls = re.findall(r"Simple mixer control '([^']+)'", result.stdout or "")
        preferred = [c for c in SPEECH_VOLUME_CONTROLS if c in controls]
        if not preferred:
            # Some USB dongles use unusual names; try the first few playback controls.
            preferred = controls[:3]
        if not preferred:
            msg = f"No mixer controls found on card {card}. Use alsamixer -c {card}."
            update_speech_status(volume_status=msg, volume_card=str(card), volume_percent=percent)
            return False, msg

        changed = []
        errors = []
        for control in preferred:
            cmd = ["amixer", "-c", str(card), "sset", control, f"{percent}%", "unmute"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                changed.append(control)
            else:
                errors.append((res.stderr or res.stdout or control).strip().splitlines()[-1])
        if changed:
            msg = f"Volume set to {percent}% on card {card}: " + ", ".join(changed)
            update_speech_status(volume_status=msg, volume_card=str(card), volume_percent=percent)
            return True, msg
        msg = "Volume controls failed: " + (" | ".join(errors[-3:]) or "unknown error")
        if not quiet:
            update_speech_status(status="volume error", last_error=msg, volume_status=msg, volume_card=str(card), volume_percent=percent)
        return False, msg
    except Exception as e:
        msg = f"Volume helper error: {e}"
        if not quiet:
            update_speech_status(status="volume error", last_error=msg, volume_status=msg, volume_card=str(card), volume_percent=percent)
        return False, msg


def generate_test_tone_wav(path, duration=1.2, rate=16000, frequency=660.0, volume=0.45):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = max(0.2, min(float(duration), 5.0))
    rate = int(rate)
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    # Soft fade avoids clicks.
    tone = np.sin(2 * np.pi * float(frequency) * t)
    fade_len = max(1, int(rate * 0.04))
    fade = np.ones_like(tone)
    fade[:fade_len] = np.linspace(0, 1, fade_len)
    fade[-fade_len:] = np.linspace(1, 0, fade_len)
    samples = np.clip(tone * fade * float(volume) * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


def play_wav_with_aplay(path, cfg=None):
    cfg = cfg or get_speech_config()
    if SPEECH_AUTO_SET_VOLUME:
        set_speaker_volume(SPEECH_VOLUME_PERCENT, device=cfg.get("device"), quiet=True)
    cmd = ["aplay", "-D", cfg["device"], path]
    with speech_playback_lock:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return result.returncode, (result.stdout or ""), (result.stderr or "")


def speech_test_beep_worker(duration=1.2):
    os.makedirs(SPEECH_SAMPLE_DIR, exist_ok=True)
    cfg = get_speech_config()
    path = os.path.join(SPEECH_SAMPLE_DIR, SPEECH_TEST_FILE)
    update_speech_status(status="playing beep", speaker_device=cfg["device"], last_message=f"Playing test beep via {cfg['device']}", last_error="")
    try:
        generate_test_tone_wav(path, duration=duration, rate=int(cfg.get("rate", SPEECH_APLAY_RATE)))
        rc, stdout, stderr = play_wav_with_aplay(path, cfg)
        if rc == 0:
            update_speech_status(status="ready", last_message=f"Speaker test beep played via {cfg['device']}", last_file=path, last_error="")
        else:
            err = (stderr or stdout or "aplay failed").strip()
            update_speech_status(status="speaker test failed", last_message="Speaker test failed", last_error=err, last_file=path)
    except FileNotFoundError:
        update_speech_status(status="aplay missing", last_error="aplay not found. Install ALSA tools: sudo apt install alsa-utils")
    except Exception as e:
        update_speech_status(status="speaker test error", last_error=str(e), last_file=path)



def speech_test_all_speakers_worker():
    """Try a short beep on every non-HDMI candidate until one works."""
    os.makedirs(SPEECH_SAMPLE_DIR, exist_ok=True)
    path = os.path.join(SPEECH_SAMPLE_DIR, SPEECH_TEST_FILE)
    try:
        generate_test_tone_wav(path, duration=0.65, rate=int(SPEECH_APLAY_RATE), frequency=740.0)
    except Exception as e:
        update_speech_status(status="speaker scan error", last_error=f"Could not generate test tone: {e}")
        return

    candidates = list_aplay_output_candidates()
    if not candidates:
        update_speech_status(status="speaker scan failed", last_error="No ALSA playback candidates found from aplay -l/-L")
        return

    errors = []
    tested = 0
    for item in candidates:
        device = item.get("device")
        if item.get("avoid"):
            continue
        tested += 1
        update_speech_status(status="testing speakers", speaker_device=device, last_message=f"Trying speaker candidate {tested}: {device}", last_error="")
        cfg = {"device": device, "rate": SPEECH_APLAY_RATE, "channels": SPEECH_APLAY_CHANNELS}
        try:
            rc, stdout, stderr = play_wav_with_aplay(path, cfg)
        except Exception as e:
            rc, stdout, stderr = 1, "", str(e)
        if rc == 0:
            set_speech_config(device=device)
            update_speech_status(status="ready", speaker_device=device, last_message=f"Working speaker selected: {device}", last_error="", last_file=path)
            return
        err = (stderr or stdout or "aplay failed").strip().splitlines()[-1]
        errors.append(f"{device}: {err}")
        time.sleep(0.15)

    for item in candidates:
        if not item.get("avoid"):
            continue
        device = item.get("device")
        update_speech_status(status="testing HDMI fallback", speaker_device=device, last_message=f"Trying fallback/HDMI candidate: {device}", last_error="")
        cfg = {"device": device, "rate": SPEECH_APLAY_RATE, "channels": SPEECH_APLAY_CHANNELS}
        try:
            rc, stdout, stderr = play_wav_with_aplay(path, cfg)
        except Exception as e:
            rc, stdout, stderr = 1, "", str(e)
        if rc == 0:
            set_speech_config(device=device)
            update_speech_status(status="ready", speaker_device=device, last_message=f"Fallback speaker selected: {device}", last_error="", last_file=path)
            return
        err = (stderr or stdout or "aplay failed").strip().splitlines()[-1]
        errors.append(f"{device}: {err}")

    update_speech_status(
        status="speaker scan failed",
        last_message="No working speaker found yet",
        last_error=" | ".join(errors[-8:]) or "All speaker candidates failed",
        last_file=path,
    )


def _apply_short_fade(samples, rate, fade_ms=12):
    if samples.size == 0:
        return samples
    fade_len = max(1, min(len(samples) // 3, int(rate * fade_ms / 1000.0)))
    env = np.ones(len(samples), dtype=np.float32)
    env[:fade_len] = np.linspace(0.0, 1.0, fade_len)
    env[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    return samples * env


def _robot_vowel(vowel, duration=0.14, rate=SPEECH_ROBOT_VOICE_RATE, f0=135.0):
    # Very small built-in robotic/formant-ish voice. This is not natural TTS,
    # but it is understandable enough for short phrases like "Hi Nico" when no
    # espeak command/library is installed.
    formants = {
        "a": (730, 1090, 2440),
        "e": (530, 1840, 2480),
        "i": (300, 2200, 3000),
        "o": (570, 840, 2410),
        "u": (330, 900, 2200),
        "y": (300, 2100, 2800),
    }
    f1, f2, f3 = formants.get(vowel, formants["e"])
    n = max(1, int(rate * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    buzz = 0.45 * np.sin(2 * np.pi * f0 * t)
    buzz += 0.16 * np.sin(2 * np.pi * f0 * 2 * t)
    buzz += 0.10 * np.sin(2 * np.pi * f0 * 3 * t)
    form = 0.28 * np.sin(2 * np.pi * f1 * t) + 0.16 * np.sin(2 * np.pi * f2 * t) + 0.07 * np.sin(2 * np.pi * f3 * t)
    return _apply_short_fade(buzz + form, rate)


def _robot_noise(duration=0.055, rate=SPEECH_ROBOT_VOICE_RATE, amp=0.16):
    n = max(1, int(rate * duration))
    samples = np.random.default_rng().normal(0, amp, n).astype(np.float32)
    return _apply_short_fade(samples, rate)


def _robot_hum(duration=0.07, rate=SPEECH_ROBOT_VOICE_RATE, f=130.0, amp=0.18):
    n = max(1, int(rate * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    samples = amp * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * (f * 2) * t))
    return _apply_short_fade(samples.astype(np.float32), rate)


def _robot_gap(duration=0.035, rate=SPEECH_ROBOT_VOICE_RATE):
    return np.zeros(max(1, int(rate * duration)), dtype=np.float32)


def _robot_word_to_audio(word, rate=SPEECH_ROBOT_VOICE_RATE):
    word = re.sub(r"[^a-z0-9]+", "", str(word or "").lower())
    if not word:
        return [_robot_gap(0.08, rate)]

    # A few useful words/names get hand-tuned patterns.
    special = {
        "hi": ["h", "ai"],
        "hello": ["h", "e", "l", "o"],
        "nico": ["n", "i", "k", "o"],
        "paul": ["p", "o", "l"],
        "person": ["p", "er", "s", "o", "n"],
        "bottle": ["b", "o", "t", "l"],
        "chair": ["ch", "e", "r"],
        "phone": ["f", "o", "n"],
        "cup": ["k", "u", "p"],
        "dog": ["d", "o", "g"],
        "cat": ["k", "a", "t"],
        "car": ["k", "a", "r"],
        "book": ["b", "u", "k"],
        "clock": ["k", "l", "o", "k"],
        "keyboard": ["k", "i", "b", "o", "r", "d"],
        "laptop": ["l", "a", "p", "t", "o", "p"],
        "remote": ["r", "e", "m", "o", "t"],
        "mouse": ["m", "ou", "s"],
        "tv": ["t", "i", "v", "i"],
    }
    units = special.get(word)
    if units is None:
        units = []
        i = 0
        while i < len(word):
            two = word[i:i+2]
            if two in {"ch", "sh", "th", "oo", "ou", "ee", "ai", "ay", "er"}:
                units.append(two)
                i += 2
            else:
                units.append(word[i])
                i += 1

    parts = []
    for u in units:
        if u in {"a", "e", "i", "o", "u", "y"}:
            parts.append(_robot_vowel(u, 0.13, rate))
        elif u in {"ai", "ay"}:
            parts.append(_robot_vowel("a", 0.12, rate))
            parts.append(_robot_vowel("i", 0.12, rate, f0=145.0))
        elif u in {"ee"}:
            parts.append(_robot_vowel("i", 0.18, rate))
        elif u in {"oo", "ou"}:
            parts.append(_robot_vowel("u", 0.16, rate))
        elif u in {"er"}:
            parts.append(_robot_vowel("e", 0.10, rate, f0=125.0))
            parts.append(_robot_hum(0.06, rate, f=115.0))
        elif u in {"s", "f", "h", "sh", "th"}:
            parts.append(_robot_noise(0.055 if u != "h" else 0.04, rate, amp=0.12 if u == "h" else 0.18))
        elif u in {"p", "t", "k", "c", "q"}:
            parts.append(_robot_gap(0.018, rate))
            parts.append(_robot_noise(0.03, rate, amp=0.22))
        elif u in {"b", "d", "g", "v", "z", "m", "n", "l", "r", "w"}:
            freq = {"m": 105, "n": 125, "l": 150, "r": 120, "w": 115, "b": 120, "d": 140, "g": 120, "v": 135, "z": 145}.get(u, 125)
            parts.append(_robot_hum(0.055, rate, f=freq))
        else:
            parts.append(_robot_vowel("e", 0.08, rate))
        parts.append(_robot_gap(0.012, rate))
    return parts


def generate_robot_speech_wav(text, path, rate=SPEECH_ROBOT_VOICE_RATE):
    text = safe_speech_text(text)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not words:
        words = ["hi"]
    parts = []
    for word in words:
        parts.extend(_robot_word_to_audio(word, rate))
        parts.append(_robot_gap(0.12, rate))
    samples = np.concatenate(parts) if parts else np.zeros(int(rate * 0.2), dtype=np.float32)
    # Add a tiny robot shimmer so it sounds intentional rather than like a pure tone.
    t = np.linspace(0, len(samples) / rate, len(samples), endpoint=False)
    samples = samples + 0.025 * np.sin(2 * np.pi * 34 * t).astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 1.0
    if peak > 0:
        samples = samples / peak * 0.55
    samples_i16 = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(samples_i16.tobytes())
    return path


def try_py_espeakng_to_wav(text, wav_path):
    """Try py-espeak-ng's synth_wav API if the user installed it with pip."""
    try:
        from espeakng import ESpeakNG
        try:
            esng = ESpeakNG(voice="en-gb")
        except TypeError:
            esng = ESpeakNG()
            try:
                esng.voice = "en-gb"
            except Exception:
                pass
        if not hasattr(esng, "synth_wav"):
            return False, "py-espeak-ng installed but synth_wav is unavailable"
        wav_bytes = esng.synth_wav(text)
        if not wav_bytes:
            return False, "py-espeak-ng returned no WAV data"
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        return True, "py-espeak-ng synth_wav"
    except Exception as e:
        return False, str(e)


def speech_say_worker(text, allow_beep_fallback=True):
    text = safe_speech_text(text)
    os.makedirs(SPEECH_SAMPLE_DIR, exist_ok=True)
    cfg = get_speech_config()
    wav_path = os.path.join(SPEECH_SAMPLE_DIR, SPEECH_TTS_FILE)
    robot_path = os.path.join(SPEECH_SAMPLE_DIR, SPEECH_ROBOT_VOICE_FILE)
    update_speech_status(status="speaking", speaker_device=cfg["device"], last_phrase=text, last_message=f"Trying to speak via {cfg['device']}", last_error="")
    try:
        tts_cmd = None
        for candidate in SPEECH_PREFERRED_TTS_COMMANDS:
            found = shutil.which(candidate)
            if found:
                tts_cmd = found
                break
        if tts_cmd:
            # Generate a WAV first, then play it through the selected ALSA device.
            # This avoids espeak choosing HDMI/default audio by mistake.
            tts_args = [tts_cmd]
            engine_name = os.path.basename(tts_cmd)
            if engine_name in {"espeak", "espeak-ng"}:
                tts_args += ["-v", SPEECH_TTS_VOICE, "-s", SPEECH_TTS_SPEED, "-a", SPEECH_TTS_AMPLITUDE]
            tts_args += ["-w", wav_path, text]
            gen = subprocess.run(tts_args, capture_output=True, text=True, timeout=20)
            if gen.returncode != 0:
                err = (gen.stderr or gen.stdout or "TTS generation failed").strip()
                update_speech_status(status="speech generation failed", last_error=err, last_file=wav_path, tts_engine=os.path.basename(tts_cmd))
                return
            rc, stdout, stderr = play_wav_with_aplay(wav_path, cfg)
            if rc == 0:
                update_speech_status(status="ready", last_message=f"Spoke phrase via {cfg['device']}", last_file=wav_path, last_error="", tts_engine=os.path.basename(tts_cmd))
            else:
                err = (stderr or stdout or "aplay failed").strip()
                update_speech_status(status="speech playback failed", last_error=err, last_file=wav_path, tts_engine=os.path.basename(tts_cmd))
            return

        # Try the user's pip-installed py-espeak-ng wrapper if available. It can
        # synthesize a WAV, which we then route through Janet's selected Y02 ALSA device.
        ok, engine_msg = try_py_espeakng_to_wav(text, wav_path)
        if ok:
            rc, stdout, stderr = play_wav_with_aplay(wav_path, cfg)
            if rc == 0:
                update_speech_status(status="ready", last_message=f"Spoke phrase using {engine_msg} via {cfg['device']}", last_file=wav_path, last_error="", tts_engine="py-espeak-ng")
            else:
                err = (stderr or stdout or "aplay failed").strip()
                update_speech_status(status="speech playback failed", last_error=err, last_file=wav_path, tts_engine="py-espeak-ng")
            return

        # Built-in fallback: generate a small robotic voice WAV instead of a plain beep.
        if SPEECH_ROBOT_VOICE_FALLBACK:
            generate_robot_speech_wav(text, robot_path, rate=int(cfg.get("rate", SPEECH_ROBOT_VOICE_RATE)))
            rc, stdout, stderr = play_wav_with_aplay(robot_path, cfg)
            if rc == 0:
                update_speech_status(status="ready", last_message=f"Spoke phrase with built-in robot voice via {cfg['device']}", last_file=robot_path, last_error="", tts_engine="built-in robot voice")
            else:
                err = (stderr or stdout or "aplay failed").strip()
                update_speech_status(status="robot speech playback failed", last_error=err, last_file=robot_path, tts_engine="built-in robot voice")
            return

        # Keep the old beep fallback only for manual test phrases, not automatic greetings.
        if allow_beep_fallback:
            generate_test_tone_wav(wav_path, duration=1.0, rate=int(cfg.get("rate", SPEECH_APLAY_RATE)), frequency=880.0)
            rc, stdout, stderr = play_wav_with_aplay(wav_path, cfg)
            if rc == 0:
                update_speech_status(status="ready", last_message="No TTS engine installed, so Janet played a beep instead.", last_file=wav_path, last_error="", tts_engine="beep fallback")
            else:
                err = (stderr or stdout or "aplay failed").strip()
                update_speech_status(status="speech fallback failed", last_error=err, last_file=wav_path, tts_engine="beep fallback")
        else:
            update_speech_status(status="no speech engine", last_message="No TTS engine available and beep fallback is disabled for greetings", last_error=engine_msg, tts_engine="none")
    except FileNotFoundError:
        update_speech_status(status="aplay missing", last_error="aplay not found. Install ALSA tools: sudo apt install alsa-utils")
    except Exception as e:
        update_speech_status(status="speech error", last_error=str(e), last_file=wav_path)


def wav_audio_stats(path):
    """Return basic volume stats for a WAV file recorded by arecord."""
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
        if not frames:
            return {"rms": 0, "peak": 0, "db": -120.0, "level": 0}
        if sample_width == 2:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        elif sample_width == 1:
            samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) * 256
        else:
            return {"rms": 0, "peak": 0, "db": -120.0, "level": 0}
        if channels and channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if samples.size == 0:
            return {"rms": 0, "peak": 0, "db": -120.0, "level": 0}
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        db = 20 * math.log10(max(rms, 1.0) / 32768.0)
        # 0 = silence/too quiet, 100 = loud. Normal speech is often 20-70 here.
        level = int(max(0, min(100, ((db + 60.0) / 60.0) * 100)))
        return {"rms": round(rms, 1), "peak": round(peak, 1), "db": round(db, 1), "level": level}
    except Exception as e:
        return {"rms": 0, "peak": 0, "db": -120.0, "level": 0, "error": str(e)}


def mark_wake_word_heard(text=""):
    """Show a temporary green JANET overlay on the video stream."""
    global voice_wake_overlay_until
    voice_wake_overlay_until = time.time() + VOICE_WAKE_OVERLAY_SECONDS
    with voice_status_lock:
        with lock:
            voice = latest_readings.setdefault("voice", {})
            voice["wake_word_active"] = True
            if text is not None:
                voice["last_heard"] = text


def mark_sound_heard(stats=None):
    """Show a temporary SOUND overlay when the mic records clear audio energy."""
    global voice_sound_overlay_until
    voice_sound_overlay_until = time.time() + VOICE_SOUND_OVERLAY_SECONDS
    stats = stats or {}
    msg = f"Sound heard: level {stats.get('level', 0)}% / {stats.get('db', -120)} dB"
    with voice_status_lock:
        with lock:
            voice = latest_readings.setdefault("voice", {})
            voice["sound_active"] = True
            voice["last_sound_message"] = msg


def update_voice_status(status=None, last_heard=None, last_action=None, last_error=None, sample_status=None, sample_file=None, sample_message=None, mic_level=None, mic_level_db=None, voice_device=None, last_test_text=None, last_test_message=None, arecord_devices=None, last_sound_message=None, last_raw_file=None, last_boosted_file=None):
    global voice_wake_overlay_until, voice_sound_overlay_until
    with voice_status_lock:
        with lock:
            voice = latest_readings.setdefault("voice", {})
            voice["available"] = SPEECH_AVAILABLE
            voice["enabled"] = VOICE_ENABLED
            voice["wake_word_active"] = time.time() < voice_wake_overlay_until
            voice["sound_active"] = time.time() < voice_sound_overlay_until
            voice["voice_device"] = get_voice_config().get("device", VOICE_ARECORD_DEVICE)
            if status is not None:
                voice["status"] = status
            if last_heard is not None:
                voice["last_heard"] = last_heard
            if last_action is not None:
                voice["last_action"] = last_action
            if last_error is not None:
                voice["last_error"] = last_error
            if sample_status is not None:
                voice["sample_status"] = sample_status
            if sample_file is not None:
                voice["sample_file"] = sample_file
            if sample_message is not None:
                voice["sample_message"] = sample_message
            if mic_level is not None:
                voice["mic_level"] = mic_level
            if mic_level_db is not None:
                voice["mic_level_db"] = mic_level_db
            if voice_device is not None:
                voice["voice_device"] = voice_device
            if last_test_text is not None:
                voice["last_test_text"] = last_test_text
            if last_test_message is not None:
                voice["last_test_message"] = last_test_message
            if arecord_devices is not None:
                voice["arecord_devices"] = arecord_devices
            if last_sound_message is not None:
                voice["last_sound_message"] = last_sound_message
            if last_raw_file is not None:
                voice["last_raw_file"] = last_raw_file
            if last_boosted_file is not None:
                voice["last_boosted_file"] = last_boosted_file


def stop_live_voice_capture():
    """Pause live listening and stop its active arecord process, freeing the USB mic."""
    global voice_live_capture_proc
    voice_recording_event.set()
    with voice_process_lock:
        proc = voice_live_capture_proc
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    time.sleep(0.15)


def run_arecord_capture(output_path, duration, cfg, interruptible=False):
    """Record a WAV file with arecord. When interruptible=True, Voice tab tests can stop it."""
    global voice_live_capture_proc
    cmd = [
        "arecord",
        "-D", cfg["device"],
        "-d", str(int(duration)),
        "-r", cfg["rate"],
        "-c", cfg["channels"],
        "-f", "S16_LE",
        "-t", "wav",
        output_path,
    ]
    with voice_capture_lock:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if interruptible:
            with voice_process_lock:
                voice_live_capture_proc = proc
        try:
            deadline = time.time() + int(duration) + 8
            while proc.poll() is None:
                if interruptible and voice_recording_event.is_set():
                    try:
                        proc.terminate()
                        proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    return -999, "", "live capture interrupted", True
                if time.time() > deadline:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return -998, "", "arecord timed out", False
                time.sleep(0.05)
            stdout, stderr = proc.communicate(timeout=1)
            return proc.returncode, stdout or "", stderr or "", False
        finally:
            if interruptible:
                with voice_process_lock:
                    if voice_live_capture_proc is proc:
                        voice_live_capture_proc = None


def voice_debug_path(filename):
    os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
    return os.path.join(VOICE_SAMPLE_DIR, filename)


def prepare_recognition_wav(wav_path, debug_prefix="janet"):
    """Create an adaptively boosted mono 16-bit WAV for recognition.

    Returns (audio_bytes, samplerate, gain, boosted_path).
    The original file is copied too, so the Voice tab can show what Janet heard.
    """
    try:
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(f"SoundFile is required for Janet voice recognition: {e}")

    data, samplerate = sf.read(wav_path, dtype="int16", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1).astype(np.int16)

    raw_path = voice_debug_path(VOICE_LAST_RAW_FILE)
    boosted_path = voice_debug_path(VOICE_LAST_BOOSTED_FILE)

    try:
        sf.write(raw_path, data, int(samplerate), format="WAV", subtype="PCM_16")
    except Exception:
        raw_path = wav_path

    samples = np.asarray(data, dtype=np.float32)
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    current_db = 20 * math.log10(max(rms, 1.0) / 32768.0)
    auto_gain = 10 ** ((VOICE_TARGET_DB - current_db) / 20.0)
    auto_gain = max(1.0, min(float(VOICE_MAX_AUTO_GAIN), auto_gain))
    total_gain = max(float(VOICE_RECOGNITION_GAIN), auto_gain)

    boosted = samples * total_gain
    # Gentle limiter so quiet speech is lifted but clipping is avoided.
    peak = float(np.max(np.abs(boosted))) if boosted.size else 0.0
    if peak > 32000:
        boosted *= 32000.0 / peak
    boosted = np.clip(boosted, -32768, 32767).astype(np.int16)

    sf.write(boosted_path, boosted, int(samplerate), format="WAV", subtype="PCM_16")
    update_voice_status(last_raw_file=raw_path, last_boosted_file=boosted_path)
    return boosted.tobytes(), int(samplerate), round(total_gain, 2), boosted_path


def make_google_audio_from_wav(recognizer, wav_path):
    """Load WAV for SpeechRecognition with adaptive software gain and FLAC fallback.

    Janet now writes two debug files in voice_samples/:
    - janet_last_voice_raw.wav: exactly what arecord captured
    - janet_last_voice_boosted.wav: what is sent to recognition
    """
    if sr is None:
        raise RuntimeError("SpeechRecognition is not available")

    try:
        import soundfile as sf
        audio_bytes, samplerate, used_gain, boosted_path = prepare_recognition_wav(wav_path)
        audio = sr.AudioData(audio_bytes, int(samplerate), 2)

        if not shutil.which("flac"):
            # SpeechRecognition's Google recognizer asks AudioData for FLAC bytes.
            # Provide those bytes using SoundFile so Debian's command-line flac is not needed.
            data = np.frombuffer(audio_bytes, dtype="<i2")
            bio = io.BytesIO()
            sf.write(bio, data, int(samplerate), format="FLAC", subtype="PCM_16")
            flac_bytes = bio.getvalue()
            audio.get_flac_data = lambda convert_rate=None, convert_width=2: flac_bytes

        return audio

    except Exception as sf_error:
        # Fallback path: no software gain, but still works if SpeechRecognition can
        # read the WAV and a command-line flac encoder is available.
        try:
            with sr.AudioFile(wav_path) as source:
                return recognizer.record(source)
        except Exception as sr_error:
            raise RuntimeError(
                "Could not prepare audio for speech recognition. "
                f"SoundFile path failed: {sf_error}; SpeechRecognition path failed: {sr_error}"
            )

def current_front_distance():
    with lock:
        try:
            return float(latest_readings.get("sonar", {}).get("front", -1))
        except (TypeError, ValueError):
            return -1


def queue_voice_command(direction, heard_text):
    try:
        while voice_command_queue.qsize() > 2:
            voice_command_queue.get_nowait()
        voice_command_queue.put_nowait((direction, heard_text, time.time()))
        update_voice_status(status="command queued", last_heard=heard_text, last_action=f"Queued {direction}", last_error="")
        return True
    except queue.Full:
        update_voice_status(status="queue full", last_heard=heard_text, last_error="Voice command queue full")
        return False

if MOTORS_AVAILABLE:
    try:
        motors = MotorControllerI2C()
        print(f"MotorController initialized on I2C bus {MOTOR_I2C_BUS}, addr {hex(MOTOR_I2C_ADDR)}")
    except Exception as e:
        print(f"Motor init failed: {e}")
        motors = None
else:
    print("Motors unavailable: install python3-smbus or smbus2 on the Raspberry Pi")

def clamp_float(value, minimum, maximum, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = fallback
    return max(minimum, min(value, maximum))


def get_motor_settings():
    with motor_settings_lock:
        return dict(motor_settings)


def set_motor_settings(duration=None, acceleration=None, preset=None):
    with motor_settings_lock:
        if duration is not None:
            motor_settings["duration"] = clamp_float(duration, MOTOR_MIN_DURATION, MOTOR_MAX_DURATION, motor_settings["duration"])
        if acceleration is not None:
            motor_settings["acceleration"] = clamp_float(acceleration, MOTOR_MIN_ACCELERATION, MOTOR_MAX_ACCELERATION, motor_settings["acceleration"])
        if preset is not None:
            motor_settings["preset"] = preset
        return dict(motor_settings)


def motor_function_for(direction):
    if REVERSE_MOTOR_LOGIC:
        mapping = {
            "forward": motors.backward,
            "backward": motors.forward,
            "left": motors.right,
            "right": motors.left,
            "stop": motors.stop,
        }
    else:
        mapping = {
            "forward": motors.forward,
            "backward": motors.backward,
            "left": motors.left,
            "right": motors.right,
            "stop": motors.stop,
        }
    return mapping.get(direction)


def execute_move(direction, duration=None, acceleration=None):
    if direction not in {"forward", "backward", "left", "right", "stop"}:
        return False, "Invalid direction"
    if not motors:
        return False, "Motors not available"

    settings = get_motor_settings()
    duration = clamp_float(duration, MOTOR_MIN_DURATION, MOTOR_MAX_DURATION, settings["duration"])
    acceleration = clamp_float(acceleration, MOTOR_MIN_ACCELERATION, MOTOR_MAX_ACCELERATION, settings["acceleration"])
    action = motor_function_for(direction)

    try:
        if direction == "stop":
            motor_stop_event.set()
            action()
            return True, "Stopped"

        # Cancel any previous ramp still running, then execute this command.
        motor_stop_event.set()
        time.sleep(0.005)
        motor_stop_event.clear()

        with motor_run_lock:
            if acceleration <= 0.01:
                action(duration)
            else:
                # The Arduino I2C protocol only accepts direction + duration, not speed.
                # This simulates soft acceleration by pulsing short movement chunks
                # with shrinking gaps. Higher acceleration = gentler/slower ramp.
                steps = max(2, min(8, int(2 + acceleration * 6)))
                chunk = max(0.03, duration / steps)
                for step in range(steps):
                    if motor_stop_event.is_set():
                        break
                    action(chunk)
                    pause = 0.08 * acceleration * (1 - ((step + 1) / steps))
                    if pause > 0:
                        time.sleep(pause)
        return True, f"Moving {direction} for {duration:.2f}s, acceleration {acceleration:.2f}"
    except Exception as e:
        print(f"Motor command failed: {e}")
        return False, str(e)


def execute_move_async(direction, duration=None, acceleration=None):
    """Start a motor command in the background so the web button responds immediately."""
    if direction not in {"forward", "backward", "left", "right", "stop"}:
        return False, "Invalid direction"
    if not motors:
        return False, "Motors not available"

    if direction == "stop":
        return execute_move(direction, duration=duration, acceleration=acceleration)

    worker = threading.Thread(
        target=execute_move,
        args=(direction,),
        kwargs={"duration": duration, "acceleration": acceleration},
        daemon=True,
    )
    worker.start()
    return True, f"Queued {direction}"


def routine_definitions_payload():
    return [
        {
            "id": rid,
            "name": data.get("name", rid),
            "emoji": data.get("emoji", "🤖"),
            "description": data.get("description", ""),
            "duration_seconds": MOTOR_DANCE_DURATION_SECONDS,
            "return_home": MOTOR_DANCE_RETURN_HOME,
            "pattern": list(data.get("pattern", [])),
        }
        for rid, data in MOTOR_DANCE_ROUTINES.items()
    ]


def update_routine_status(active=None, current_id=None, current_name=None, status=None, last_message=None, last_completed=None, elapsed_seconds=None, steps_done=None, step_duration=None):
    with lock:
        routines = latest_readings.setdefault("routines", {})
        if active is not None:
            routines["active"] = bool(active)
        if current_id is not None:
            routines["current_id"] = current_id
        if current_name is not None:
            routines["current_name"] = current_name
        if status is not None:
            routines["status"] = status
        if last_message is not None:
            routines["last_message"] = last_message
        if last_completed is not None:
            routines["last_completed"] = last_completed
        if elapsed_seconds is not None:
            routines["elapsed_seconds"] = round(float(elapsed_seconds), 1)
        if steps_done is not None:
            routines["steps_done"] = int(steps_done)
        if step_duration is not None:
            routines["step_duration"] = round(float(step_duration), 2)
        routines["target_seconds"] = MOTOR_DANCE_DURATION_SECONDS
        routines["routines"] = routine_definitions_payload()


def get_routine_status_payload():
    with routine_lock:
        started_at = routine_state.get("started_at")
        if routine_state.get("active") and started_at:
            elapsed = time.time() - started_at
        else:
            with lock:
                elapsed = latest_readings.get("routines", {}).get("elapsed_seconds", 0.0)
    update_routine_status(elapsed_seconds=elapsed)
    with lock:
        return dict(latest_readings.get("routines", {}))


def stop_motor_routine():
    routine_stop_event.set()
    try:
        execute_move("stop")
    except Exception:
        pass
    update_routine_status(status="stopping", last_message="Stopping routine...")


def routine_is_balanced(pattern):
    """Check whether a pattern has equal opposite directions.

    This is not real odometry, but it keeps the commanded movements balanced:
    forward count == backward count, left count == right count.
    """
    return (
        pattern.count("forward") == pattern.count("backward")
        and pattern.count("left") == pattern.count("right")
    )


def run_motor_dance_routine(routine_id):
    routine = MOTOR_DANCE_ROUTINES.get(routine_id)
    if not routine:
        update_routine_status(active=False, status="error", last_message=f"Unknown routine: {routine_id}")
        return

    name = routine.get("name", routine_id)
    pattern = [d for d in routine.get("pattern", []) if d in {"forward", "backward", "left", "right"}]
    if not pattern:
        update_routine_status(active=False, status="error", last_message=f"Routine {name} has no valid steps")
        return
    if MOTOR_DANCE_RETURN_HOME and not routine_is_balanced(pattern):
        update_routine_status(active=False, status="error", last_message=f"Routine {name} is not return-home balanced")
        return

    settings = get_motor_settings()
    step_duration = clamp_float(settings.get("duration"), MOTOR_MIN_DURATION, MOTOR_MAX_DURATION, MOTOR_DEFAULT_DURATION)
    acceleration = clamp_float(settings.get("acceleration"), MOTOR_MIN_ACCELERATION, MOTOR_MAX_ACCELERATION, MOTOR_DEFAULT_ACCELERATION)
    target_seconds = float(MOTOR_DANCE_DURATION_SECONDS)

    # A complete pattern cycle must fit inside 20 seconds, otherwise Janet could
    # stop part-way through a cycle and not return to her start point. If the
    # manual motor step has been set very long, shrink only the routine step so
    # at least one complete balanced cycle can run safely within 20 seconds.
    cycle_seconds = len(pattern) * (step_duration + MOTOR_DANCE_STEP_GAP)
    if cycle_seconds > target_seconds:
        step_duration = max(MOTOR_MIN_DURATION, (target_seconds / len(pattern)) - MOTOR_DANCE_STEP_GAP)
        cycle_seconds = len(pattern) * (step_duration + MOTOR_DANCE_STEP_GAP)

    with routine_lock:
        routine_state.update({"active": True, "current_id": routine_id, "current_name": name, "started_at": time.time(), "steps_done": 0})
    update_routine_status(
        active=True,
        current_id=routine_id,
        current_name=name,
        status="running",
        last_message=f"Running {name} for {target_seconds:.0f}s — return-home balanced",
        elapsed_seconds=0,
        steps_done=0,
        step_duration=step_duration,
    )

    print(f"[ROUTINE] Starting {name}: {routine.get('description', '')}")
    start = time.time()
    end_time = start + target_seconds
    steps_done = 0
    cycles_done = 0
    last_dir = None
    same_dir_count = 0
    stopped = False

    try:
        # Only execute full balanced cycles. If there is spare time at the end,
        # Janet waits/stops rather than starting an incomplete cycle.
        while not shutdown_event.is_set() and not routine_stop_event.is_set():
            remaining = end_time - time.time()
            if remaining < cycle_seconds:
                break

            for direction in pattern:
                if shutdown_event.is_set() or routine_stop_event.is_set():
                    break

                if direction == last_dir:
                    same_dir_count += 1
                else:
                    same_dir_count = 1
                    last_dir = direction
                if same_dir_count > MOTOR_DANCE_MAX_CONSECUTIVE_STEPS:
                    # This should not trigger for Janet's built-in routines, but it
                    # protects future edited patterns.
                    update_routine_status(status="error", last_message=f"{name} stopped: too many consecutive {direction} steps", steps_done=steps_done, elapsed_seconds=time.time() - start)
                    routine_stop_event.set()
                    break

                ok, message = execute_move(direction, duration=step_duration, acceleration=acceleration)
                steps_done += 1
                with routine_lock:
                    routine_state["steps_done"] = steps_done
                if not ok:
                    update_routine_status(status="error", last_message=f"{name} failed on {direction}: {message}", steps_done=steps_done, elapsed_seconds=time.time() - start)
                    routine_stop_event.set()
                    break
                update_routine_status(status="running", last_message=f"{name}: cycle {cycles_done + 1}, step {steps_done} {direction}", steps_done=steps_done, elapsed_seconds=time.time() - start)
                time.sleep(max(0.02, step_duration + MOTOR_DANCE_STEP_GAP))

            if routine_stop_event.is_set():
                break
            cycles_done += 1

        # If we completed normally, hold still until the 20-second routine window
        # finishes. This keeps each dance 20 seconds long without risking an
        # unbalanced final half-cycle.
        if not routine_stop_event.is_set():
            try:
                execute_move("stop")
            except Exception:
                pass
            while not shutdown_event.is_set() and time.time() < end_time:
                remaining = end_time - time.time()
                update_routine_status(status="return-home hold", last_message=f"{name}: back at start, holding for {max(0, remaining):.1f}s", steps_done=steps_done, elapsed_seconds=time.time() - start)
                time.sleep(min(0.25, max(0.02, remaining)))

    finally:
        try:
            execute_move("stop")
        except Exception:
            pass
        stopped = routine_stop_event.is_set()
        routine_stop_event.clear()
        elapsed = min(target_seconds, time.time() - start) if not stopped else max(0.0, time.time() - start)
        with routine_lock:
            routine_state.update({"active": False, "current_id": "", "current_name": "", "started_at": None, "steps_done": steps_done})
        if stopped:
            msg = f"Stopped {name} after {elapsed:.1f}s and {steps_done} step(s)"
            status = "stopped"
        else:
            msg = f"Completed {name}: 20s return-home dance, {cycles_done} balanced cycle(s), {steps_done} step(s)"
            status = "ready"
        print(f"[ROUTINE] {msg}")
        update_routine_status(active=False, current_id="", current_name="", status=status, last_message=msg, last_completed=name, elapsed_seconds=elapsed, steps_done=steps_done)

def start_motor_routine(routine_id):
    global routine_thread
    if routine_id == "random":
        routine_id = random.choice(list(MOTOR_DANCE_ROUTINES.keys()))
    if routine_id not in MOTOR_DANCE_ROUTINES:
        return False, f"Unknown routine: {routine_id}"
    if not motors:
        return False, "Motors not available"
    name = MOTOR_DANCE_ROUTINES[routine_id].get("name", routine_id)
    with routine_lock:
        if routine_state.get("active"):
            return False, f"A routine is already running: {routine_state.get('current_name', 'routine')}"
        routine_state.update({"active": True, "current_id": routine_id, "current_name": name, "started_at": time.time(), "steps_done": 0})
        routine_stop_event.clear()
        routine_thread = threading.Thread(target=run_motor_dance_routine, args=(routine_id,), daemon=True)
        routine_thread.start()
    update_routine_status(active=True, status="queued", current_id=routine_id, current_name=name, last_message=f"Queued {name}", elapsed_seconds=0, steps_done=0)
    return True, f"Started {name}"


def parse_voice_command(text):
    text = (text or "").lower().strip()
    if VOICE_WAKE_WORD not in text:
        return None

    # Only look after the wake word when possible: "janet move forward".
    after_wake = text.split(VOICE_WAKE_WORD, 1)[-1].strip()
    command_text = after_wake or text

    if any(phrase in command_text for phrase in VOICE_FORWARD_PHRASES):
        return "forward"
    if any(phrase in command_text for phrase in VOICE_BACKWARD_PHRASES):
        return "backward"
    if any(phrase in command_text for phrase in VOICE_LEFT_PHRASES):
        return "left"
    if any(phrase in command_text for phrase in VOICE_RIGHT_PHRASES):
        return "right"
    if any(phrase in command_text for phrase in VOICE_STOP_PHRASES):
        return "stop"
    return None


def voice_motor_worker():
    update_voice_status(status="voice motor worker ready")
    while not shutdown_event.is_set():
        try:
            direction, heard_text, queued_at = voice_command_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if direction == "forward":
            front = current_front_distance()
            if 0 < front < VOICE_MIN_FRONT_DISTANCE_CM:
                message = f"Forward blocked: obstacle {front:.1f}cm ahead"
                print(f"[VOICE] {message}")
                update_voice_status(status="blocked", last_heard=heard_text, last_action=message, last_error="")
                continue

        ok, message = execute_move(direction)
        print(f"[VOICE] heard='{heard_text}' direction={direction} result={message}")
        update_voice_status(
            status="listening" if ok else "motor error",
            last_heard=heard_text,
            last_action=message,
            last_error="" if ok else message,
        )


def voice_listener_loop():
    """
    Voice control without PyAudio.

    PyAudio is awkward on Raspberry Pi/Python 3.13 because it needs the
    PortAudio development headers. Janet already records samples with arecord,
    so this loop captures short WAV chunks with arecord and feeds them to
    speech_recognition.AudioFile. That keeps the camera/web threads isolated
    and avoids the PyAudio dependency completely.
    """
    if not VOICE_ENABLED:
        update_voice_status(status="disabled")
        return
    if not SPEECH_AVAILABLE:
        update_voice_status(status="speech_recognition missing", last_error="Install with: pip install SpeechRecognition")
        print("Voice unavailable: install with: pip install SpeechRecognition")
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    # Check arecord exists before starting the loop.
    try:
        check = subprocess.run(["arecord", "--version"], capture_output=True, text=True, timeout=3)
        if check.returncode != 0:
            raise RuntimeError(check.stderr.strip() or "arecord check failed")
    except Exception as e:
        msg = f"arecord unavailable: {e}. Install with: sudo apt install alsa-utils"
        update_voice_status(status="arecord missing", last_error=msg)
        print(f"Voice unavailable: {msg}")
        return

    update_voice_status(status="listening via arecord", last_error="")
    print("Voice listener ready via arecord. Say: Janet move forward")

    while not shutdown_event.is_set():
        if voice_recording_event.is_set():
            time.sleep(0.2)
            continue

        tmp_path = None
        try:
            os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
            tmp_path = os.path.join(VOICE_SAMPLE_DIR, ".janet_listen_tmp.wav")
            cfg = get_voice_config()
            update_voice_status(status=f"recording {VOICE_PHRASE_TIME_LIMIT}s via {cfg['device']}", voice_device=cfg["device"])
            rc, stdout, stderr, interrupted = run_arecord_capture(
                tmp_path, int(VOICE_PHRASE_TIME_LIMIT), cfg, interruptible=True
            )
            if interrupted:
                # Voice tab test/sample requested the microphone. Do not treat this as an error.
                continue
            if rc != 0:
                err = (stderr or stdout or "arecord failed").strip()
                # If ALSA default is broken, auto-switch to the real USB capture device.
                if "capture slave is not defined" in err or cfg.get("device") in {"default", "auto"}:
                    suggested = detect_arecord_capture_device()
                    if suggested and suggested != cfg.get("device"):
                        set_voice_config(device=suggested)
                        update_voice_status(status="switched mic device", voice_device=suggested, last_error=f"Switched from {cfg.get('device')} to {suggested}")
                        time.sleep(0.2)
                        continue
                update_voice_status(status="mic record error", last_error=err)
                time.sleep(1)
                continue

            stats = wav_audio_stats(tmp_path)
            if stats.get("level", 0) >= VOICE_SOUND_LEVEL_THRESHOLD:
                mark_sound_heard(stats)
            level_msg = f"level {stats.get('level', 0)}% / {stats.get('db', -120)} dB, adaptive gain to {VOICE_TARGET_DB} dB"
            update_voice_status(status=f"recognising ({level_msg})", mic_level=stats.get("level", 0), mic_level_db=stats.get("db", -120.0), last_sound_message=f"Sound heard at {stats.get('level', 0)}% / {stats.get('db', -120)} dB" if stats.get("level", 0) >= VOICE_SOUND_LEVEL_THRESHOLD else "Too quiet / no clear sound")
            audio = make_google_audio_from_wav(recognizer, tmp_path)
            text = recognizer.recognize_google(audio).lower().strip()
            print(f"[VOICE] heard='{text}' {level_msg}")
            if VOICE_WAKE_WORD in text:
                mark_wake_word_heard(text)
            update_voice_status(status="listening via arecord", last_heard=text, last_error="")
            direction = parse_voice_command(text)
            if direction:
                queue_voice_command(direction, text)

        except sr.UnknownValueError:
            update_voice_status(status="listening via arecord", last_action="Sound heard, but words not understood", last_error="Could not understand audio")
        except Exception as e:
            update_voice_status(status="voice error", last_error=str(e))
            print(f"Voice listener error: {e}")
            time.sleep(1)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass



def safe_sample_filename(name):
    name = (name or "janet_voice_sample").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._") or "janet_voice_sample"
    if not name.lower().endswith(".wav"):
        name += ".wav"
    return name


def record_voice_sample_worker(duration, filename):
    duration = int(clamp_float(duration, 1, VOICE_SAMPLE_MAX_DURATION, VOICE_SAMPLE_DEFAULT_DURATION))
    filename = safe_sample_filename(filename)
    os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
    output_path = os.path.join(VOICE_SAMPLE_DIR, filename)

    stop_live_voice_capture()
    update_voice_status(
        status="voice sample recording",
        sample_status="recording",
        sample_file=output_path,
        sample_message=f"Recording {duration}s sample...",
        last_error="",
    )
    try:
        cfg = get_voice_config()
        rc, stdout, stderr, interrupted = run_arecord_capture(output_path, duration, cfg, interruptible=False)
        if rc == 0 and os.path.exists(output_path):
            stats = wav_audio_stats(output_path)
            message = f"Saved sample: {output_path} | level {stats.get('level', 0)}% / {stats.get('db', -120)} dB"
            print(f"[VOICE SAMPLE] {message}")
            update_voice_status(status="listening", sample_status="saved", sample_file=output_path, sample_message=message, last_error="", mic_level=stats.get("level", 0), mic_level_db=stats.get("db", -120.0))
        else:
            err = (stderr or stdout or "arecord failed").strip()
            print(f"[VOICE SAMPLE] Failed: {err}")
            update_voice_status(status="sample failed", sample_status="error", sample_file=output_path, sample_message=err, last_error=err)
    except FileNotFoundError:
        msg = "arecord not found. Install ALSA tools: sudo apt install alsa-utils"
        print(f"[VOICE SAMPLE] {msg}")
        update_voice_status(status="sample failed", sample_status="error", sample_file=output_path, sample_message=msg, last_error=msg)
    except Exception as e:
        msg = str(e)
        print(f"[VOICE SAMPLE] {msg}")
        update_voice_status(status="sample failed", sample_status="error", sample_file=output_path, sample_message=msg, last_error=msg)
    finally:
        voice_recording_event.clear()

# ----------------------------
# Sonar class
# ----------------------------
class SonarArray:
    def __init__(self):
        if not GPIO_AVAILABLE: raise RuntimeError("RPi.GPIO not available")
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        for name in TRIG_PINS:
            GPIO.setup(TRIG_PINS[name], GPIO.OUT)
            GPIO.setup(ECHO_PINS[name], GPIO.IN)
            GPIO.output(TRIG_PINS[name], False)
        time.sleep(0.2)

    def _measure(self, trig, echo):
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)
        timeout_start = time.time()
        pulse_start = pulse_end = None
        while GPIO.input(echo) == 0:
            pulse_start = time.time()
            if time.time() - timeout_start > 0.05: return -1
        timeout_start = time.time()
        while GPIO.input(echo) == 1:
            pulse_end = time.time()
            if time.time() - timeout_start > 0.05: return -1
        if pulse_start is None or pulse_end is None: return -1
        distance_cm = (pulse_end - pulse_start) * 17150 + 1.15
        return round(distance_cm, 2) if 0 < distance_cm < 500 else -1

    def get_readings(self):
        return {name: self._measure(TRIG_PINS[name], ECHO_PINS[name]) for name in TRIG_PINS}

    def cleanup(self):
        if GPIO_AVAILABLE: GPIO.cleanup()

def sonar_loop():
    if not GPIO_AVAILABLE: return
    try:
        sonar = SonarArray()
        while not shutdown_event.is_set():
            with lock: latest_readings["sonar"] = sonar.get_readings()
            time.sleep(0.2)
    except Exception as e:
        print(f"Sonar error: {e}")
    finally:
        if 'sonar' in locals(): sonar.cleanup()

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# ----------------------------
# CAMERAS
# ----------------------------
def generate_front_stream_legacy():
    try:
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build()
            detection = pipeline.create(dai.node.DetectionNetwork).build(
                camera, dai.NNModelDescription(DETECTION_MODEL_NAME)
            )
            detection.setConfidenceThreshold(DETECTION_CONFIDENCE)
            label_map = detection.getClasses()
            with lock:
                latest_readings.setdefault("detection_model", {}).update({
                    "name": DETECTION_MODEL_NAME,
                    "family": DETECTION_MODEL_FAMILY,
                    "purpose": DETECTION_MODEL_PURPOSE,
                    "runtime": DETECTION_RUNTIME,
                    "confidence": DETECTION_CONFIDENCE,
                    "labels_count": len(label_map or []),
                    "labels_preview": list(label_map[:12]) if label_map else [],
                })
            q_rgb = detection.passthrough.createOutputQueue()
            q_det = detection.out.createOutputQueue()
            pipeline.start()
            fps_counter = 0
            fps_timer = time.time()
            current_fps = 0.0
            face_frame_counter = 0
            
            while pipeline.isRunning() and not shutdown_event.is_set():
                in_rgb = q_rgb.get()
                in_det = q_det.get()
                frame = in_rgb.getCvFrame()
                detections = in_det.detections
                
                detection_readings = []
                for det in detections:
                    bbox = frame_norm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                    label = label_map[det.label] if label_map and det.label < len(label_map) else str(det.label)
                    confidence = int(det.confidence * 100)
                    detection_readings.append({"label": label, "confidence": confidence, "xmin": det.xmin, "ymin": det.ymin, "xmax": det.xmax, "ymax": det.ymax})
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {confidence}%", (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                check_object_greetings_and_centering(detection_readings, frame.shape[1], frame)

                fps_counter += 1
                now = time.time()
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_timer = now

                show_wake = now < voice_wake_overlay_until
                show_sound = now < voice_sound_overlay_until
                with lock:
                    latest_readings["detections"] = detection_readings
                    latest_readings["fps"] = round(current_fps, 1)
                    latest_readings.setdefault("voice", {})["wake_word_active"] = show_wake
                    latest_readings.setdefault("voice", {})["sound_active"] = show_sound

                draw_voice_overlay(frame, show_wake, show_sound)

                # Store the latest front frame so the Face tab can add a known face.
                with face_lock:
                    globals()["face_last_frame"] = frame.copy()

                # Lightweight face recognition overlay. It runs every few frames to keep FPS stable.
                face_frame_counter += 1
                if face_frame_counter % max(1, int(FACE_DETECTION_EVERY_N_FRAMES)) == 0:
                    globals()["last_face_results"] = process_faces_for_frame(frame)
                for face in globals().get("last_face_results", []):
                    x, y, w, h = int(face.get("x", 0)), int(face.get("y", 0)), int(face.get("w", 0)), int(face.get("h", 0))
                    name = face.get("name", "Unknown")
                    score = face.get("score")
                    color = FACE_OVERLAY_COLOR_KNOWN if name != "Unknown" else FACE_OVERLAY_COLOR_UNKNOWN
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{name}" if score is None else f"{name} {score}"
                    cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                success, jpeg = cv2.imencode(".jpg", frame)
                if not success: continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
    except Exception as exc:
        print(f"Front camera error: {exc}")
        while not shutdown_event.is_set():
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Front Cam Error", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            success, jpeg = cv2.imencode(".jpg", frame)
            if success: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(1)


def make_front_placeholder_frame(title="Front Camera Starting", subtitle="OAK-D detection pipeline is warming up"):
    frame = np.zeros((FRONT_CAMERA_PLACEHOLDER_HEIGHT, FRONT_CAMERA_PLACEHOLDER_WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, title, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, subtitle, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 190), 1)
    return frame


def set_latest_front_frame(frame):
    """Store the latest fully annotated front frame for streaming and screenshots."""
    global front_latest_jpeg, front_latest_frame_id, front_latest_frame_time, front_latest_annotated_frame
    success, jpeg = cv2.imencode(".jpg", frame)
    if not success:
        return False
    with front_stream_condition:
        front_latest_jpeg = jpeg.tobytes()
        front_latest_frame_id += 1
        front_latest_frame_time = time.time()
        front_latest_annotated_frame = frame.copy()
        front_stream_condition.notify_all()
    return True


def set_front_placeholder(title="Front Camera Starting", subtitle="OAK-D detection pipeline is warming up"):
    set_latest_front_frame(make_front_placeholder_frame(title, subtitle))


def _wrap_overlay_text(text, max_chars=54):
    """Small single/two-line text wrapper for voice overlays."""
    text = " ".join(str(text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) >= 1:
            break
    if current and len(lines) < 2:
        lines.append(current)
    if len(lines) == 2 and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip(" .") + "..."
    return lines[:2]


def draw_voice_overlay(frame, show_wake, show_sound):
    """Draw compact voice/sound feedback in the bottom-left corner."""
    if not (show_wake or show_sound):
        return

    with lock:
        voice = dict(latest_readings.get("voice", {}))

    heard = (voice.get("last_heard") or voice.get("last_test_text") or "").strip()
    sound_msg = (voice.get("last_sound_message") or "audio detected").strip()

    if show_wake:
        prefix = "JANET"
        color = (0, 255, 0)
        detail = heard or "wake word heard"
    else:
        prefix = "SOUND"
        color = (0, 255, 255)
        detail = heard or sound_msg or "audio detected"

    lines = _wrap_overlay_text(f"{prefix}: {detail}", max_chars=90)
    if not lines:
        return

    # 50% smaller than V13.17, tucked into the bottom-left corner.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.28
    thickness = 1
    line_gap = 4
    x = 10
    line_height = 12
    y = max(16, frame.shape[0] - 12 - ((len(lines) - 1) * (line_height + line_gap)))

    for idx, line in enumerate(lines):
        yy = y + idx * (line_height + line_gap)
        cv2.putText(frame, line, (x, yy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, yy), font, scale, color, thickness, cv2.LINE_AA)


def front_camera_worker():
    """Own the OAK-D pipeline independently of any browser connection."""
    global front_camera_error
    set_front_placeholder()
    while not shutdown_event.is_set():
        try:
            with lock:
                latest_readings["status"] = "front camera starting"
            with dai.Pipeline() as pipeline:
                camera = pipeline.create(dai.node.Camera).build()
                detection = pipeline.create(dai.node.DetectionNetwork).build(
                    camera, dai.NNModelDescription(DETECTION_MODEL_NAME)
                )
                detection.setConfidenceThreshold(DETECTION_CONFIDENCE)
                label_map = detection.getClasses()
                with lock:
                    latest_readings.setdefault("detection_model", {}).update({
                        "name": DETECTION_MODEL_NAME,
                        "family": DETECTION_MODEL_FAMILY,
                        "purpose": DETECTION_MODEL_PURPOSE,
                        "runtime": DETECTION_RUNTIME,
                        "confidence": DETECTION_CONFIDENCE,
                        "labels_count": len(label_map or []),
                        "labels_preview": list(label_map[:12]) if label_map else [],
                    })
                q_rgb = detection.passthrough.createOutputQueue()
                q_det = detection.out.createOutputQueue()
                pipeline.start()
                front_camera_ready.set()
                front_camera_error = ""
                with lock:
                    latest_readings["status"] = "front camera active"
                print("Front camera worker started: OAK-D detection is active")

                fps_counter = 0
                fps_timer = time.time()
                current_fps = 0.0
                face_frame_counter = 0

                while pipeline.isRunning() and not shutdown_event.is_set():
                    in_rgb = q_rgb.get()
                    in_det = q_det.get()
                    frame = in_rgb.getCvFrame()
                    detections = in_det.detections

                    detection_readings = []
                    for det in detections:
                        bbox = frame_norm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                        label = label_map[det.label] if label_map and det.label < len(label_map) else str(det.label)
                        confidence = int(det.confidence * 100)
                        detection_readings.append({"label": label, "confidence": confidence, "xmin": det.xmin, "ymin": det.ymin, "xmax": det.xmax, "ymax": det.ymax})
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        cv2.putText(frame, f"{label} {confidence}%", (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    check_object_greetings_and_centering(detection_readings, frame.shape[1], frame)

                    fps_counter += 1
                    now = time.time()
                    elapsed = now - fps_timer
                    if elapsed >= 1.0:
                        current_fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_timer = now

                    show_wake = now < voice_wake_overlay_until
                    show_sound = now < voice_sound_overlay_until
                    with lock:
                        latest_readings["detections"] = detection_readings
                        latest_readings["fps"] = round(current_fps, 1)
                        latest_readings.setdefault("voice", {})["wake_word_active"] = show_wake
                        latest_readings.setdefault("voice", {})["sound_active"] = show_sound

                    draw_voice_overlay(frame, show_wake, show_sound)

                    # Store the raw annotated object frame for adding known faces.
                    with face_lock:
                        globals()["face_last_frame"] = frame.copy()

                    # Lightweight face recognition overlay. It runs every few frames to keep FPS stable.
                    face_frame_counter += 1
                    if face_frame_counter % max(1, int(FACE_DETECTION_EVERY_N_FRAMES)) == 0:
                        globals()["last_face_results"] = process_faces_for_frame(frame)
                    for face in globals().get("last_face_results", []):
                        x, y, w, h = int(face.get("x", 0)), int(face.get("y", 0)), int(face.get("w", 0)), int(face.get("h", 0))
                        name = face.get("name", "Unknown")
                        score = face.get("score")
                        color = FACE_OVERLAY_COLOR_KNOWN if name != "Unknown" else FACE_OVERLAY_COLOR_UNKNOWN
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = f"{name}" if score is None else f"{name} {score}"
                        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                    set_latest_front_frame(frame)

        except Exception as exc:
            front_camera_ready.clear()
            front_camera_error = str(exc)
            print(f"Front camera worker error: {exc}")
            with lock:
                latest_readings["status"] = "front camera error"
            set_front_placeholder("Front Cam Error", str(exc)[:70])
            time.sleep(FRONT_CAMERA_RESTART_DELAY)


def start_front_camera_worker():
    """Start the OAK-D worker once, even if no browser has connected yet."""
    global front_camera_thread
    with front_camera_start_lock:
        if front_camera_thread is not None and front_camera_thread.is_alive():
            return False
        front_camera_thread = threading.Thread(target=front_camera_worker, daemon=True)
        front_camera_thread.start()
        front_camera_started.set()
        return True


def generate_front_stream():
    """MJPEG stream backed by the always-on front camera worker."""
    start_front_camera_worker()
    last_id = -1
    while not shutdown_event.is_set():
        with front_stream_condition:
            front_stream_condition.wait_for(
                lambda: shutdown_event.is_set() or (front_latest_jpeg is not None and front_latest_frame_id != last_id),
                timeout=1.0,
            )
            jpeg = front_latest_jpeg
            frame_id = front_latest_frame_id
        if jpeg is None:
            frame = make_front_placeholder_frame()
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                time.sleep(0.2)
                continue
            jpeg = encoded.tobytes()
        last_id = frame_id
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def mjpeg_from_frame(frame):
    success, jpeg = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


def placeholder_stream(title="Rear Camera Disabled", subtitle="Set REAR_CAMERA_ENABLED=True when connected"):
    while not shutdown_event.is_set():
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, title, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, subtitle, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 190), 1)
        packet = mjpeg_from_frame(frame)
        if packet:
            yield packet
        time.sleep(1)


def open_rear_camera():
    """Open only configured indexes, quietly, and return the first working camera."""
    for idx in REAR_CAMERA_INDEXES:
        device_path = f"/dev/video{idx}"
        if not os.path.exists(device_path):
            continue

        print(f"Checking rear camera on {device_path}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REAR_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REAR_CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            continue

        # Try a few frames; some cameras need a moment to wake up.
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Rear camera locked on {device_path}")
                return cap
            time.sleep(0.05)

        cap.release()

    return None


def generate_rear_stream():
    if not REAR_CAMERA_ENABLED:
        yield from placeholder_stream()
        return

    cap = open_rear_camera()
    if cap is None:
        print("Rear camera unavailable. Showing placeholder stream instead.")
        yield from placeholder_stream("Rear Cam Unavailable", "Check cable/index then reload")
        return

    try:
        while not shutdown_event.is_set():
            success, frame = cap.read()
            if not success or frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Rear Frame Drop", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            packet = mjpeg_from_frame(frame)
            if packet:
                yield packet
    finally:
        cap.release()

@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html style="margin:0; padding:0;">
<head>
    <title>Janet V13.30 Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { background: #0b0d12; color: #f3f6ff; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; touch-action: manipulation; }
        header { width: 100%; padding: 20px; background: #151923; border-bottom: 1px solid #2d3547; text-align: center; }
        .container { display: grid; grid-template-columns: minmax(320px, 1fr) 300px; gap: 20px; padding: 20px; width: 100%; box-sizing: border-box; max-width: 1040px; }
        .container.full-tab { grid-template-columns: 1fr; max-width: 1040px; }
        .container.full-tab .side-panels { display: none; }
        .container.full-tab .motor-panel, .container.full-tab .voice-test-panel, .container.full-tab .face-panel-full, .container.full-tab .object-panel-full { max-width: none; }
        .container.full-tab .tabs { max-width: none; }
        
        .tabs { display: flex; gap: 5px; margin-bottom: 10px; flex-wrap: wrap; }
        .tab { 
            padding: 10px 20px; background: #1d2330; border: 1px solid #2d3547; 
            color: #9ca7bd; cursor: pointer; border-radius: 8px 8px 0 0; transition: 0.2s;
        }
        .tab.active { background: #2d3547; color: #57f287; border-bottom-color: #57f287; }
        .main-tabs { margin-bottom: 12px; }
        .settings-tabs { margin-top: -4px; margin-bottom: 12px; padding-left: 8px; border-left: 3px solid #57f287; }
        .settings-tabs .tab { font-size: 14px; padding: 8px 14px; border-radius: 8px; }
        .speech-panel-full { background:#151923; border:1px solid #2d3547; border-radius:12px; padding:18px; width:100%; max-width:700px; box-sizing:border-box; }
        .container.full-tab .speech-panel-full { max-width:none; padding:22px; }

        .video-container { position: relative; width: 100%; max-width: 700px; background: black; border-radius: 12px; overflow: hidden; border: 1px solid #2d3547;}
        .screenshot-btn { position:absolute; top:10px; left:10px; z-index:30; width:40px; height:34px; border-radius:10px; border:1px solid #57f287; background:rgba(11,13,18,.72); color:#57f287; cursor:pointer; font-size:18px; backdrop-filter: blur(3px); }
        .screenshot-btn:hover { background:rgba(87,242,135,.20); }
        .main-feed { width: 100%; display: block; }
        .hidden { display: none; }

        .mirror { 
            position: absolute; top: 10px; right: 10px; 
            width: 120px; height: 75px; 
            border: 3px solid #57f287; border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.5);
            overflow: hidden; z-index: 10;
        }
        .mirror img { width: 100%; height: 100%; object-fit: cover; }

        .controls { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
        .arrow { 
            position: absolute; width: 68px; height: 68px; 
            background: rgba(87, 242, 135, 0.3); border: 2px solid #57f287; 
            border-radius: 50%; display: flex; align-items: center; justify-content: center; 
            font-weight: bold; color: #57f287; cursor: pointer; pointer-events: auto;
            user-select: none; -webkit-user-select: none; touch-action: none; transition: 0.08s; font-size: 22px;
        }
        .arrow:active, .arrow.pressed { background: rgba(87, 242, 135, 0.8); filter: brightness(1.25); }
        .up { top: 8px; left: 50%; transform: translateX(-50%); }
        .down { bottom: 8px; left: 50%; transform: translateX(-50%); }
        .left { left: 8px; top: 50%; transform: translateY(-50%); }
        .right { right: 8px; top: 50%; transform: translateY(-50%); }
        .stop { top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 107, 107, 0.3); border-color: #ff6b6b; color: #ff6b6b; }
        .stop:active { background: rgba(255, 107, 107, 0.8); }
        .panel { background: #151923; border: 1px solid #2d3547; border-radius: 16px; padding: 16px; }
        .sensor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .sensor-card { background: #1d2330; padding: 12px; border-radius: 12px; text-align: center; border: 1px solid #2d3547; }
        .val { font-size: 24px; font-weight: bold; display: block; transition: color 0.3s; }
        .ok { color: #57f287; }
        .warn { color: #ffcc66; }
        .bad { color: #ff6b6b; }
        .det-card { background: #1d2330; border: 1px solid #2d3547; border-radius: 12px; padding: 10px; margin-bottom: 10px; }
        .det-label { font-size: 18px; font-weight: bold; }
        .det-conf { color: #57f287; font-size: 14px; }
        .motor-panel { background: #151923; border: 1px solid #2d3547; border-radius: 12px; padding: 18px; width: 100%; max-width: 700px; box-sizing: border-box; }
        .container.full-tab .motor-panel { padding: 22px; }
        .motor-row { display: grid; grid-template-columns: 160px 1fr 80px; gap: 10px; align-items: center; margin: 12px 0; }
        .motor-row input { width: 100%; }
        .preset-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
        .preset-btn { background:#1d2330; color:#f3f6ff; border:1px solid #2d3547; border-radius:10px; padding:10px 14px; cursor:pointer; }
        .preset-btn:hover { border-color:#57f287; color:#57f287; }
        .save-btn { background:#57f287; color:#0b0d12; border:none; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:bold; }
        .motor-note { color:#9ca7bd; font-size:14px; line-height:1.4; }
        .voice-line { background:#1d2330; border:1px solid #2d3547; border-radius:10px; padding:9px; margin:7px 0; font-size:14px; }
        .voice-label { color:#9ca7bd; display:block; font-size:12px; margin-bottom:3px; }
        .voice-test-panel { background:#151923; border:1px solid #2d3547; border-radius:12px; padding:18px; width:100%; max-width:700px; box-sizing:border-box; }
        .container.full-tab .voice-test-panel { padding: 22px; }
        .voice-tab-grid { display:grid; grid-template-columns:minmax(0, 1.35fr) minmax(260px, 0.65fr); gap:18px; align-items:start; }
        .voice-tools-column, .voice-control-column { background:#1d2330; border:1px solid #2d3547; border-radius:14px; padding:16px; box-sizing:border-box; }
        .voice-control-column .voice-line { background:#151923; }
        .voice-test-row { display:grid; grid-template-columns:160px 1fr; gap:10px; align-items:center; margin:12px 0; }
        .voice-test-row input { width:100%; box-sizing:border-box; background:#1d2330; color:#f3f6ff; border:1px solid #2d3547; border-radius:8px; padding:9px; }
        .record-btn { background:#57f287; color:#0b0d12; border:none; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:bold; }
        .wake-active { color:#57f287; font-weight:bold; }
        .sound-active { color:#ffcc66; font-weight:bold; }
        .level-bar { width:100%; height:12px; background:#0b0d12; border:1px solid #2d3547; border-radius:999px; overflow:hidden; margin-top:5px; }
        .level-fill { height:100%; width:0%; background:#57f287; transition:width .2s; }
        .device-list { background:#0b0d12; border:1px solid #2d3547; color:#9ca7bd; padding:10px; border-radius:10px; white-space:pre-wrap; font-size:12px; max-height:240px; overflow:auto; text-align:left; }
        .face-panel-full, .object-panel-full { background:#151923; border:1px solid #2d3547; border-radius:12px; padding:18px; width:100%; max-width:700px; box-sizing:border-box; }
        .container.full-tab .face-panel-full, .container.full-tab .object-panel-full { max-width:none; padding:22px; }
        .face-tab-grid, .object-tab-grid { display:grid; grid-template-columns:minmax(260px, .9fr) minmax(0, 1.1fr); gap:18px; align-items:start; }
        .face-card { background:#1d2330; border:1px solid #2d3547; border-radius:14px; padding:14px; margin-bottom:12px; }
        .face-row { display:grid; grid-template-columns:150px 1fr; gap:10px; align-items:center; margin:10px 0; }
        .face-row input { width:100%; box-sizing:border-box; background:#0b0d12; color:#f3f6ff; border:1px solid #2d3547; border-radius:8px; padding:9px; }
        .face-list-item { background:#0b0d12; border:1px solid #2d3547; border-radius:10px; padding:10px; margin:8px 0; display:grid; grid-template-columns:1fr auto; gap:10px; align-items:start; }
        .face-thumb-grid { display:grid; grid-template-columns:repeat(5, 46px); gap:7px; margin-top:8px; }
        .face-thumb { width:46px; height:46px; object-fit:cover; border-radius:8px; border:1px solid #2d3547; cursor:pointer; background:#151923; }
        .face-thumb:hover { border-color:#57f287; transform:scale(1.06); }
        .thumb-wrap { position:relative; width:46px; height:46px; }
        .thumb-x { position:absolute; top:-6px; right:-6px; width:18px; height:18px; border-radius:50%; border:1px solid #0b0d12; background:#ff3b3b; color:white; font-size:12px; line-height:16px; cursor:pointer; font-weight:bold; padding:0; }
        .thumb-x:hover { filter:brightness(1.2); transform:scale(1.08); }
        .small-danger { background:#ff6b6b; color:#0b0d12; border:none; border-radius:8px; padding:7px 10px; cursor:pointer; font-weight:bold; }
        .detection-panel-full { background:#151923; border:1px solid #2d3547; border-radius:12px; padding:18px; width:100%; max-width:700px; box-sizing:border-box; }
        .container.full-tab .detection-panel-full { max-width:none; padding:22px; }
        .detection-tab-grid { display:grid; grid-template-columns:minmax(260px, .8fr) minmax(0, 1.2fr); gap:18px; align-items:start; }
        .model-card { background:#1d2330; border:1px solid #2d3547; border-radius:14px; padding:14px; margin-bottom:12px; }
        .model-line { display:grid; grid-template-columns:150px 1fr; gap:10px; margin:7px 0; font-size:14px; }
        .model-key { color:#9ca7bd; }
        .model-value { color:#f3f6ff; word-break:break-word; }
        .scan-result { background:#0b0d12; border:1px solid #2d3547; border-radius:10px; padding:10px; margin:8px 0; text-align:left; }
        .scan-tag { display:inline-block; border:1px solid #57f287; color:#57f287; border-radius:999px; padding:2px 8px; font-size:12px; margin-left:6px; }
        .scan-tag.warn { border-color:#ffcc66; color:#ffcc66; }
        .scan-tag.bad { border-color:#ff6b6b; color:#ff6b6b; }
        .routine-panel-full { background:#151923; border:1px solid #2d3547; border-radius:12px; padding:18px; width:100%; max-width:700px; box-sizing:border-box; }
        .container.full-tab .routine-panel-full { max-width:none; padding:22px; }
        .routine-grid { display:grid; grid-template-columns:minmax(260px, .75fr) minmax(0, 1.25fr); gap:18px; align-items:start; }
        .routine-card { background:#1d2330; border:1px solid #2d3547; border-radius:14px; padding:14px; margin-bottom:12px; }
        .routine-buttons { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
        .routine-btn { background:#0b0d12; color:#f3f6ff; border:1px solid #2d3547; border-radius:12px; padding:14px; cursor:pointer; text-align:left; min-height:92px; }
        .routine-btn:hover { border-color:#57f287; color:#57f287; }
        .routine-btn-title { display:block; font-size:18px; font-weight:bold; margin-bottom:5px; }
        .routine-btn-desc { display:block; color:#9ca7bd; font-size:13px; line-height:1.35; }
        .routine-stop-btn { background:#ff6b6b; color:#0b0d12; border:none; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:bold; }
        .routine-progress { width:100%; height:12px; background:#0b0d12; border:1px solid #2d3547; border-radius:999px; overflow:hidden; margin-top:8px; }
        .routine-progress-fill { height:100%; width:0%; background:#57f287; transition:width .25s; }

        .side-panels { display: flex; flex-direction: column; gap: 20px; }
        @media (max-width: 760px) {
            header { padding: 12px; }
            header h1 { font-size: 22px; }
            .container { display: flex; flex-direction: column; padding: 10px; gap: 10px; max-width: none; }
            .video-container { max-width: none; border-radius: 10px; }
            .main-feed { width: 100%; height: auto; }
            .mirror { width: 92px; height: 58px; top: 10px; right: 8px; border-width: 2px; }
            .arrow { width: 64px; height: 64px; font-size: 22px; background: rgba(87, 242, 135, 0.42); }
            .stop { width: 72px; height: 72px; font-size: 13px; }
            .tabs { width: 100%; }
            .tab { flex: 1; text-align: center; padding: 10px 6px; font-size: 14px; }
            .settings-tabs { padding-left: 0; border-left: none; margin-top: 0; }
            .settings-tabs .tab { flex: 1 1 30%; font-size: 13px; padding: 8px 5px; }
            .side-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .container.full-tab .side-panels { display: none; }
            .voice-panel { grid-column: 1 / -1; }
            .panel { padding: 10px; border-radius: 12px; }
            .panel h3 { font-size: 16px; margin-bottom: 8px; }
            .sensor-grid { grid-template-columns: 1fr; gap: 6px; }
            .sensor-card { padding: 8px 4px; font-size: 12px; }
            .val { font-size: 18px; }
            .det-card { padding: 8px; margin-bottom: 6px; }
            .det-label { font-size: 15px; }
            .det-conf { font-size: 12px; }
            .motor-panel, .routine-panel-full, .voice-test-panel, .detection-panel-full, .speech-panel-full, .face-panel-full, .object-panel-full { max-width: none; padding: 12px; }
            .voice-tab-grid, .detection-tab-grid, .face-tab-grid, .object-tab-grid, .routine-grid { grid-template-columns: 1fr; gap: 10px; }
            .voice-tools-column, .voice-control-column { padding: 12px; }
            .face-card { padding: 12px; }
            .motor-row, .voice-test-row { grid-template-columns: 1fr; gap: 4px; }
            .preset-row { flex-direction: column; }
            .routine-buttons { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body style="margin:0; padding:0;">
    <header><h1 style="margin:0">Janet V13.30 Control</h1></header>
    <div id="main-container" class="container">
        <div style="display:flex; flex-direction:column;">
            <div class="tabs main-tabs">
                <div id="tab-front" class="tab active" onclick="switchView('front')">Front View</div>
                <div id="tab-rear" class="tab" onclick="switchView('rear')">Rear View</div>
                <div id="tab-settings" class="tab" onclick="openSettings()">Settings</div>
            </div>
            <div id="settings-tabs" class="tabs settings-tabs hidden">
                <div id="subtab-motors" class="tab" onclick="switchView('motors')">Motors</div>
                <div id="subtab-routines" class="tab" onclick="switchView('routines')">Routines</div>
                <div id="subtab-detection" class="tab" onclick="switchView('detection')">Detection</div>
                <div id="subtab-voice" class="tab" onclick="switchView('voice')">Voice</div>
                <div id="subtab-speech" class="tab" onclick="switchView('speech')">Speech</div>
                <div id="subtab-face" class="tab" onclick="switchView('face')">Faces</div>
                <div id="subtab-object" class="tab" onclick="switchView('object')">Objects</div>
            </div>
            <div id="video-box" class="video-container">
                <button id="screenshot-btn" class="screenshot-btn" onclick="takeScreenshot(event)" title="Save front camera screenshot">📸</button>
                <img id="feed-front" class="main-feed" src="/video_front">
                <img id="feed-rear" class="main-feed hidden" src="/video_rear">
                <div id="pip-mirror" class="mirror">
                    <img id="pip-img" src="/video_rear">
                </div>
                <div class="controls">
                    <div class="arrow up" data-dir="forward">▲</div>
                    <div class="arrow down" data-dir="backward">▼</div>
                    <div class="arrow left" data-dir="left">◀</div>
                    <div class="arrow right" data-dir="right">▶</div>
                    <div class="arrow stop" data-dir="stop">STOP</div>
                </div>
            </div>
            <div id="motors-panel" class="motor-panel hidden">
                <h2 style="margin-top:0">Motor Settings</h2>
                <p class="motor-note">This tab now uses the full page width, so the distance and detection panels are hidden while tuning motors.</p>
                <p class="motor-note">Default movement is now half duration. Adjust below, then press Save. Acceleration is software-pulsed unless your Arduino firmware supports real speed ramping.</p>
                <div class="motor-row">
                    <label for="duration-input">Duration</label>
                    <input id="duration-input" type="range" min="0.05" max="3" step="0.05" value="0.25" oninput="syncMotorLabels()">
                    <span id="duration-label">0.25s</span>
                </div>
                <div class="motor-row">
                    <label for="accel-input">Acceleration</label>
                    <input id="accel-input" type="range" min="0" max="1" step="0.05" value="0" oninput="syncMotorLabels()">
                    <span id="accel-label">0.00</span>
                </div>
                <button class="save-btn" onclick="saveMotorSettings()">Save Settings</button>
                <div class="preset-row">
                    <button class="preset-btn" onclick="applyPreset('soft')">Soft acceleration</button>
                    <button class="preset-btn" onclick="applyPreset('normal')">Normal</button>
                    <button class="preset-btn" onclick="applyPreset('racing')">Racing mode</button>
                </div>
                <p id="motor-status" class="motor-note">Loading motor settings...</p>
            </div>
            <div id="routines-panel" class="routine-panel-full hidden">
                <h2 style="margin-top:0">Motor Routines</h2>
                <p class="motor-note">Cable-safe dance routines. Each dance runs for <b>20 seconds</b>, uses balanced movement cycles, and returns to the start point at the end.</p>
                <div class="routine-grid">
                    <div>
                        <div class="routine-card">
                            <h3 style="margin-top:0">Routine status</h3>
                            <div class="model-line"><span class="model-key">Status</span><span id="routine-status" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Current</span><span id="routine-current" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Steps done</span><span id="routine-steps" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Duration</span><span id="routine-duration" class="model-value">20s</span></div>
                            <div class="routine-progress"><div id="routine-progress-fill" class="routine-progress-fill"></div></div>
                            <p id="routine-message" class="motor-note">Motor routines ready.</p>
                            <button class="routine-stop-btn" onclick="stopRoutine()">Stop Routine</button>
                        </div>
                        <div class="routine-card">
                            <h3 style="margin-top:0">Cable note</h3>
                            <p class="motor-note">These are deliberately small dances. A “step” is based on the manual arrow movement. For cable safety, each routine uses equal opposite moves, avoids more than 5 consecutive steps in one direction, then waits/stops at the start point until the 20 seconds is complete.</p>
                            <button class="preset-btn" onclick="startRandomRoutine()">Run Random Dance</button>
                        </div>
                    </div>
                    <div>
                        <div class="routine-card">
                            <h3 style="margin-top:0">Dance moves</h3>
                            <div id="routine-buttons" class="routine-buttons"><p class="motor-note">Loading routines...</p></div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="voice-test-panel" class="voice-test-panel hidden">
                <div class="voice-tab-grid">
                    <div class="voice-tools-column">
                        <h2 style="margin-top:0">Voice Test</h2>
                        <p class="motor-note">This tab now uses the full page width. Voice tools are on the left and live Voice Control is on the right.</p>
                        <p class="motor-note">Use this page to prove whether Janet can hear the USB microphone. Try <b>List microphones</b>, then <b>Test recognition</b> while saying: Janet move forward.</p>
                        <p class="motor-note">Recognition now uses <b>adaptive software boost</b>. If Janet hears sound but cannot understand words, check the raw and boosted files shown below.</p>
                        <div class="voice-test-row">
                            <label for="voice-device-input">arecord device</label>
                            <input id="voice-device-input" type="text" value="plughw:CARD=Y02,DEV=0" placeholder="plughw:CARD=Y02,DEV=0">
                        </div>
                        <button class="preset-btn" onclick="saveVoiceSettings()">Save Voice Device</button>
                        <button class="preset-btn" onclick="listVoiceDevices()">List Microphones</button>
                        <pre id="voice-devices-list" class="device-list">Press List Microphones to show ALSA capture devices.</pre>
                        <hr style="border-color:#2d3547; margin:16px 0;">
                        <div class="voice-test-row">
                            <label for="test-duration-input">Recognition test duration</label>
                            <input id="test-duration-input" type="number" min="1" max="8" step="1" value="6">
                        </div>
                        <button class="record-btn" onclick="testVoiceRecognition()">Test Recognition</button>
                        <p id="voice-test-status" class="motor-note">Recognition test ready.</p>
                        <p id="voice-debug-files" class="motor-note">Debug files will appear after a test.</p>
                        <hr style="border-color:#2d3547; margin:16px 0;">
                        <p class="motor-note">Record a short sample from Janet's USB microphone. This pauses live voice listening while the sample is recording.</p>
                        <div class="voice-test-row">
                            <label for="sample-duration-input">Sample duration</label>
                            <input id="sample-duration-input" type="number" min="1" max="30" step="1" value="5">
                        </div>
                        <div class="voice-test-row">
                            <label for="sample-filename-input">File name</label>
                            <input id="sample-filename-input" type="text" value="janet_voice_sample.wav">
                        </div>
                        <button class="record-btn" onclick="recordVoiceSample()">Start Recording Sample</button>
                        <p id="voice-record-status" class="motor-note">Ready to record.</p>
                    </div>
                    <div class="voice-control-column">
                        <h2 style="margin-top:0">Voice Control</h2>
                        <div class="voice-line"><span class="voice-label">Status</span><span id="voice-status">-</span></div>
                        <div class="voice-line"><span class="voice-label">Last heard</span><span id="voice-heard">-</span></div>
                        <div class="voice-line"><span class="voice-label">Last action</span><span id="voice-action">-</span></div>
                        <div class="voice-line"><span class="voice-label">Mic device</span><span id="voice-device">-</span></div>
                        <div class="voice-line"><span class="voice-label">Mic level</span><span id="voice-level-text">-</span><div class="level-bar"><div id="voice-level-fill" class="level-fill"></div></div></div>
                        <div class="voice-line"><span class="voice-label">Wake word</span><span id="voice-wake">-</span></div>
                        <div class="voice-line"><span class="voice-label">Sound detector</span><span id="voice-sound">-</span></div>
                    </div>
                </div>
            </div>
            <div id="speech-panel" class="speech-panel-full hidden">
                <h2 style="margin-top:0">Speech / Speaker Output</h2>
                <p class="motor-note">This page tests Janet's speaker separately from the microphone. Voice input can work while speaker output is still on the wrong ALSA playback device.</p>
                <p class="motor-note"><b>Important:</b> Your working USB speaker/microphone is <code>Y02 / BY Y02</code>. Janet now defaults to <code>plughw:CARD=Y02,DEV=0</code> and avoids HDMI/vc4hdmi. If sound disappears again, check the mixer volume or press <b>Set USB Volume 85%</b>.</p>
                <div class="voice-tab-grid">
                    <div class="voice-tools-column">
                        <h3 style="margin-top:0">Speaker Test</h3>
                        <div class="voice-test-row">
                            <label for="speech-device-input">aplay device</label>
                            <input id="speech-device-input" type="text" value="plughw:CARD=Y02,DEV=0" placeholder="plughw:CARD=Y02,DEV=0">
                        </div>
                        <button class="preset-btn" onclick="saveSpeechSettings()">Save Speaker Device</button>
                        <button class="preset-btn" onclick="listSpeechDevices()">List Speakers</button>
                        <pre id="speech-devices-list" class="device-list">Press List Speakers to show ALSA playback devices.</pre>
                        <hr style="border-color:#2d3547; margin:16px 0;">
                        <button class="record-btn" onclick="testSpeakerBeep()">Test Selected Speaker Beep</button>
                        <button class="preset-btn" onclick="setSpeechVolume()">Set USB Volume 85%</button>
                        <button class="preset-btn" onclick="testAllSpeakers()">Try All Speakers</button>
                        <p class="motor-note">If you hear this beep, ALSA speaker output is working. If the selected device is HDMI/vc4hdmi0, press Try All Speakers or List Speakers first.</p>
                        <hr style="border-color:#2d3547; margin:16px 0;">
                        <div class="voice-test-row">
                            <label for="speech-phrase-input">Test phrase</label>
                            <input id="speech-phrase-input" type="text" value="Hello, I am Janet. My speaker is working.">
                        </div>
                        <button class="record-btn" onclick="speakTestPhrase()">Speak Test Phrase</button>
                        <p class="motor-note">Janet now prefers the real <b>espeak-ng</b> voice engine. She generates a WAV first, then plays it through the selected Y02 speaker device.</p>
                    </div>
                    <div class="voice-control-column">
                        <h3 style="margin-top:0">Speech Status</h3>
                        <div class="voice-line"><span class="voice-label">Status</span><span id="speech-status">-</span></div>
                        <div class="voice-line"><span class="voice-label">Speaker device</span><span id="speech-device">-</span></div>
                        <div class="voice-line"><span class="voice-label">Last message</span><span id="speech-message">-</span></div>
                        <div class="voice-line"><span class="voice-label">Last error</span><span id="speech-error">-</span></div>
                        <div class="voice-line"><span class="voice-label">Last phrase</span><span id="speech-phrase">-</span></div>
                        <div class="voice-line"><span class="voice-label">TTS engine</span><span id="speech-tts">-</span></div>
                        <div class="voice-line"><span class="voice-label">Object hello</span><span id="speech-object-greeting">-</span></div>
                        <div class="voice-line"><span class="voice-label">Face/Object centering</span><span id="speech-centering">-</span></div>
                        <div class="voice-line"><span class="voice-label">Mixer volume</span><span id="speech-volume">-</span></div>
                    </div>
                </div>
            </div>
            <div id="detection-panel" class="detection-panel-full hidden">
                <h2 style="margin-top:0">Detection Model</h2>
                <p class="motor-note">This tab shows the current OAK-D detection setup and scans for local/candidate models. It does not hot-swap the running camera pipeline, so the working feed stays safe.</p>
                <div class="detection-tab-grid">
                    <div>
                        <div class="model-card">
                            <h3 style="margin-top:0">Current model</h3>
                            <div class="model-line"><span class="model-key">Name</span><span id="det-model-name" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Family</span><span id="det-model-family" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Runtime</span><span id="det-runtime" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Confidence</span><span id="det-confidence" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">FPS</span><span id="det-fps" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Labels</span><span id="det-labels" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">DepthAI</span><span id="det-depthai-version" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">OpenCV</span><span id="det-opencv-version" class="model-value">-</span></div>
                        </div>
                        <div class="model-card">
                            <h3 style="margin-top:0">Safe recommendation</h3>
                            <p id="det-recommendation" class="motor-note">Loading...</p>
                            <button class="record-btn" onclick="scanDetectionModels()">Scan for newer / better models</button>
                            <p id="det-scan-status" class="motor-note">Press scan to inspect candidates and local cache.</p>
                        </div>
                    </div>
                    <div>
                        <div class="model-card">
                            <h3 style="margin-top:0">Scan results</h3>
                            <div id="det-scan-results"><p class="motor-note">No scan yet.</p></div>
                        </div>
                        <div class="model-card">
                            <h3 style="margin-top:0">Live detections</h3>
                            <div id="det-live-list"><p class="motor-note">No detections yet.</p></div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="face-panel" class="face-panel-full hidden">
                <h2 style="margin-top:0">Face Recognition</h2>
                <p class="motor-note">Local OpenCV face recognition. Add a known face while the person is visible in the front camera, then Janet will draw their name over the video feed.</p>
                <div class="face-tab-grid">
                    <div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Face system</h3>
                            <div class="model-line"><span class="model-key">Status</span><span id="face-status" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Available</span><span id="face-available" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Known samples</span><span id="face-known-count" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Storage</span><span id="face-storage" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Greeting routine</span><span id="face-greeting-status" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Last greeting</span><span id="face-last-greeting" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Speech hello</span><span id="face-speech-greeting-status" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Last said</span><span id="face-last-speech-greeting" class="model-value">-</span></div>
                            <div class="face-row">
                                <label for="face-enabled-input">Recognition</label>
                                <select id="face-enabled-input" style="background:#0b0d12;color:#f3f6ff;border:1px solid #2d3547;border-radius:8px;padding:9px;">
                                    <option value="true">Enabled</option>
                                    <option value="false">Disabled</option>
                                </select>
                            </div>
                            <div class="face-row">
                                <label for="face-threshold-input">Match threshold</label>
                                <input id="face-threshold-input" type="range" min="20" max="120" step="1" value="55" oninput="document.getElementById('face-threshold-label').textContent=this.value">
                            </div>
                            <p class="motor-note">Threshold: <span id="face-threshold-label">55</span>. Lower = stricter, higher = more forgiving.</p>
                            <p class="motor-note">Movement greeting: Janet now uses left/right centering only, no forward/back face movement.</p>
                            <p class="motor-note">Speech hello: Janet says “Hi Name” for any known face, also with a 10-minute unseen cooldown.</p>
                            <button class="record-btn" onclick="saveFaceSettings()">Save Face Settings</button>
                            <p id="face-message" class="motor-note">Ready.</p>
                        </div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Add known face</h3>
                            <p class="motor-note">Put one person clearly in front of Janet, type their name, then press Add. Add 2–3 samples per person from slightly different angles for better matching.</p>
                            <div class="face-row">
                                <label for="face-name-input">Name</label>
                                <input id="face-name-input" type="text" placeholder="Nico">
                            </div>
                            <button class="record-btn" onclick="addKnownFace()">Add Face From Camera</button>
                        </div>
                    </div>
                    <div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Known faces</h3>
                            <div id="face-list"><p class="motor-note">Loading...</p></div>
                        </div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Currently seen</h3>
                            <div id="face-seen-list"><p class="motor-note">No faces seen yet.</p></div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="object-panel" class="object-panel-full hidden">
                <h2 style="margin-top:0">Object Memory</h2>
                <p class="motor-note">Janet saves a cropped picture when she acknowledges an object. The same object label is only acknowledged every 30 minutes, so YOLO jitter should not make her keep saying the same thing.</p>
                <div class="object-tab-grid">
                    <div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Object system</h3>
                            <div class="model-line"><span class="model-key">Status</span><span id="object-status" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Known objects</span><span id="object-known-count" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Saved samples</span><span id="object-samples-count" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Cooldown</span><span id="object-cooldown" class="model-value">30 minutes</span></div>
                            <div class="model-line"><span class="model-key">Room scan</span><span id="object-scan-status" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Last return</span><span id="object-last-return" class="model-value">-</span></div>
                            <div class="model-line"><span class="model-key">Storage</span><span id="object-storage" class="model-value">-</span></div>
                            <p id="object-message" class="motor-note">Ready.</p>
                        </div>
                        <div class="face-card">
                            <h3 style="margin-top:0">How it works</h3>
                            <p class="motor-note">When Janet says “Hi object name”, she saves the current object crop here. She will turn left/right only to face newly acknowledged objects, then after a short quiet period she turns back toward her starting angle.</p>
                        </div>
                    </div>
                    <div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Known objects</h3>
                            <div id="object-list"><p class="motor-note">Loading...</p></div>
                        </div>
                        <div class="face-card">
                            <h3 style="margin-top:0">Currently seen objects</h3>
                            <div id="object-seen-list"><p class="motor-note">No objects seen yet.</p></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="side-panels">
            <div class="panel section sensors-panel">
                <h3 style="margin-top:0">Distance Sensors</h3>
                <div class="sensor-grid">
                    <div class="sensor-card">Front<span class="val" id="f">-</span></div>
                    <div class="sensor-card">Back<span class="val" id="b">-</span></div>
                    <div class="sensor-card">Left<span class="val" id="l">-</span></div>
                    <div class="sensor-card">Right<span class="val" id="r">-</span></div>
                </div>
            </div>
            <div class="panel section detections-panel">
                <h3 id="detections-title" style="margin-top:0">Detections (0 FPS)</h3>
                <div id="detections-list">
                    <p style="color:#9ca7bd; font-size:14px;">No detections yet...</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        let lastMoveAt = 0;
        async function move(dir) {
            const now = Date.now();
            if (dir !== 'stop' && now - lastMoveAt < 80) return;
            lastMoveAt = now;
            try {
                const d = document.getElementById('duration-input')?.value || '';
                const a = document.getElementById('accel-input')?.value || '';
                const url = '/move/' + dir + '?duration=' + encodeURIComponent(d) + '&acceleration=' + encodeURIComponent(a) + '&t=' + now;
                fetch(url, { cache: 'no-store', keepalive: false }).catch(e => console.log("Move error", e));
            } catch (e) { console.log("Move error", e); }
        }
        function takeScreenshot(ev) {
            if (ev) { ev.preventDefault(); ev.stopPropagation(); }
            const a = document.createElement('a');
            a.href = '/screenshot_front?t=' + Date.now();
            a.download = '';
            document.body.appendChild(a);
            a.click();
            a.remove();
        }
        function bindMotorButtons() {
            document.querySelectorAll('.arrow[data-dir]').forEach(btn => {
                const dir = btn.dataset.dir;
                const trigger = (ev) => {
                    ev.preventDefault();
                    ev.stopPropagation();
                    btn.classList.add('pressed');
                    move(dir);
                };
                btn.addEventListener('pointerdown', trigger, { passive: false });
                btn.addEventListener('touchstart', trigger, { passive: false });
                btn.addEventListener('mousedown', trigger);
                ['pointerup','pointercancel','pointerleave','touchend','mouseup'].forEach(name => {
                    btn.addEventListener(name, () => btn.classList.remove('pressed'));
                });
            });
        }
        let currentSettingsView = 'motors';
        function openSettings() {
            switchView(currentSettingsView || 'motors');
        }
        function setTabClass(id, active) {
            const el = document.getElementById(id);
            if (el) el.className = active ? 'tab active' : 'tab';
        }
        function switchView(view) {
            const isCameraView = view === 'front' || view === 'rear';
            const isSettingsView = !isCameraView;
            if (isSettingsView) currentSettingsView = view;

            setTabClass('tab-front', view === 'front');
            setTabClass('tab-rear', view === 'rear');
            setTabClass('tab-settings', isSettingsView);

            setTabClass('subtab-motors', view === 'motors');
            setTabClass('subtab-routines', view === 'routines');
            setTabClass('subtab-detection', view === 'detection');
            setTabClass('subtab-voice', view === 'voice');
            setTabClass('subtab-speech', view === 'speech');
            setTabClass('subtab-face', view === 'face');
            setTabClass('subtab-object', view === 'object');

            const settingsTabs = document.getElementById('settings-tabs');
            const front = document.getElementById('feed-front');
            const rear = document.getElementById('feed-rear');
            const pipBox = document.getElementById('pip-mirror');
            const pip = document.getElementById('pip-img');
            const motorPanel = document.getElementById('motors-panel');
            const routinesPanel = document.getElementById('routines-panel');
            const voiceTestPanel = document.getElementById('voice-test-panel');
            const detectionPanel = document.getElementById('detection-panel');
            const speechPanel = document.getElementById('speech-panel');
            const facePanel = document.getElementById('face-panel');
            const objectPanel = document.getElementById('object-panel');
            const mainContainer = document.getElementById('main-container');
            const videoBox = document.getElementById('video-box');

            settingsTabs.classList.toggle('hidden', !isSettingsView);
            videoBox.classList.toggle('hidden', !isCameraView);
            mainContainer.classList.toggle('full-tab', isSettingsView);

            motorPanel.classList.toggle('hidden', view !== 'motors');
            routinesPanel.classList.toggle('hidden', view !== 'routines');
            voiceTestPanel.classList.toggle('hidden', view !== 'voice');
            detectionPanel.classList.toggle('hidden', view !== 'detection');
            speechPanel.classList.toggle('hidden', view !== 'speech');
            facePanel.classList.toggle('hidden', view !== 'face');
            objectPanel.classList.toggle('hidden', view !== 'object');

            if (view === 'front') {
                front.classList.remove('hidden');
                rear.classList.add('hidden');
                pipBox.classList.remove('hidden');
                pip.src = '/video_rear';
            } else if (view === 'rear') {
                front.classList.add('hidden');
                rear.classList.remove('hidden');
                pipBox.classList.remove('hidden');
                pip.src = '/video_front';
            } else {
                front.classList.add('hidden');
                rear.classList.add('hidden');
                pipBox.classList.add('hidden');
            }

            if (view === 'routines') loadRoutineInfo();
            if (view === 'detection') loadDetectionInfo();
            if (view === 'face') loadFaceInfo();
            if (view === 'object') loadObjectInfo();
        }
        function syncMotorLabels() {
            const d = Number(document.getElementById('duration-input').value);
            const a = Number(document.getElementById('accel-input').value);
            document.getElementById('duration-label').textContent = d.toFixed(2) + 's';
            document.getElementById('accel-label').textContent = a.toFixed(2);
        }
        async function loadMotorSettings() {
            try {
                const res = await fetch('/motor_settings');
                const data = await res.json();
                document.getElementById('duration-input').value = data.duration;
                document.getElementById('accel-input').value = data.acceleration;
                syncMotorLabels();
                document.getElementById('motor-status').textContent = 'Current preset: ' + data.preset + ' | Motors: ' + (data.motors_available ? 'available' : 'not available');
            } catch (e) {}
        }
        async function saveMotorSettings() {
            const payload = { duration: Number(document.getElementById('duration-input').value), acceleration: Number(document.getElementById('accel-input').value) };
            const res = await fetch('/motor_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
            const data = await res.json();
            document.getElementById('motor-status').textContent = data.message || 'Settings saved';
            loadMotorSettings();
        }
        async function applyPreset(name) {
            const res = await fetch('/motor_preset/' + name, { method:'POST' });
            const data = await res.json();
            document.getElementById('motor-status').textContent = data.message || 'Preset applied';
            loadMotorSettings();
        }
        function renderRoutineButtons(items) {
            const box = document.getElementById('routine-buttons');
            if (!box) return;
            if (!items || items.length === 0) {
                box.innerHTML = '<p class="motor-note">No routines available.</p>';
                return;
            }
            box.innerHTML = '';
            items.forEach(item => {
                box.innerHTML += `<button class="routine-btn" onclick="startRoutine('${item.id}')"><span class="routine-btn-title">${item.emoji || '🤖'} ${item.name}</span><span class="routine-btn-desc">${item.description || ''}<br>${item.duration_seconds || 20}s return-home routine</span></button>`;
            });
        }
        function updateRoutinePanel(r) {
            if (!r) return;
            const elapsed = Number(r.elapsed_seconds || 0);
            const target = Number(r.target_seconds || 20);
            const pct = target > 0 ? Math.max(0, Math.min(100, (elapsed / target) * 100)) : 0;
            document.getElementById('routine-status').textContent = r.status || '-';
            document.getElementById('routine-current').textContent = r.current_name || (r.active ? 'Running' : '-');
            document.getElementById('routine-steps').textContent = r.steps_done || 0;
            document.getElementById('routine-duration').textContent = elapsed.toFixed(1) + 's / ' + target.toFixed(0) + 's';
            document.getElementById('routine-progress-fill').style.width = pct + '%';
            document.getElementById('routine-message').textContent = r.last_message || 'Ready';
            if (r.routines) renderRoutineButtons(r.routines);
        }
        async function loadRoutineInfo() {
            try {
                const res = await fetch('/routine_info');
                const data = await res.json();
                updateRoutinePanel(data.routines || data);
            } catch(e) {
                document.getElementById('routine-message').textContent = 'Could not load routines: ' + e;
            }
        }
        async function startRoutine(id) {
            try {
                document.getElementById('routine-message').textContent = 'Starting routine...';
                const res = await fetch('/routine_start/' + encodeURIComponent(id), { method:'POST' });
                const data = await res.json();
                updateRoutinePanel(data.routines || data);
            } catch(e) {
                document.getElementById('routine-message').textContent = 'Start failed: ' + e;
            }
        }
        async function startRandomRoutine() {
            try {
                document.getElementById('routine-message').textContent = 'Choosing a random dance...';
                const res = await fetch('/routine_start/random', { method:'POST' });
                const data = await res.json();
                updateRoutinePanel(data.routines || data);
            } catch(e) {
                document.getElementById('routine-message').textContent = 'Random dance failed: ' + e;
            }
        }
        async function stopRoutine() {
            try {
                const res = await fetch('/routine_stop', { method:'POST' });
                const data = await res.json();
                updateRoutinePanel(data.routines || data);
            } catch(e) {
                document.getElementById('routine-message').textContent = 'Stop failed: ' + e;
            }
        }
        async function recordVoiceSample() {
            const duration = Number(document.getElementById('sample-duration-input').value || 5);
            const filename = document.getElementById('sample-filename-input').value || 'janet_voice_sample.wav';
            document.getElementById('voice-record-status').textContent = 'Starting recording...';
            try {
                const res = await fetch('/voice_record_sample', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({ duration, filename })
                });
                const data = await res.json();
                document.getElementById('voice-record-status').textContent = data.message || data.status || 'Recording started';
            } catch (e) {
                document.getElementById('voice-record-status').textContent = 'Recording request failed: ' + e;
            }
        }
        async function saveVoiceSettings() {
            const device = document.getElementById('voice-device-input').value || 'auto';
            try {
                const res = await fetch('/voice_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ device }) });
                const data = await res.json();
                document.getElementById('voice-test-status').textContent = data.message || 'Voice settings saved';
            } catch(e) { document.getElementById('voice-test-status').textContent = 'Save failed: ' + e; }
        }
        async function listVoiceDevices() {
            document.getElementById('voice-devices-list').textContent = 'Checking microphones...';
            try {
                const res = await fetch('/voice_devices');
                const data = await res.json();
                document.getElementById('voice-devices-list').textContent = data.devices || data.message || 'No output';
                if (data.suggested_device) {
                    document.getElementById('voice-device-input').value = data.suggested_device;
                    document.getElementById('voice-test-status').textContent = 'Suggested device selected: ' + data.suggested_device;
                }
            } catch(e) { document.getElementById('voice-devices-list').textContent = 'Device list failed: ' + e; }
        }
        async function testVoiceRecognition() {
            const duration = Number(document.getElementById('test-duration-input').value || 4);
            document.getElementById('voice-test-status').textContent = 'Recording recognition test... say: Janet move forward';
            try {
                const res = await fetch('/voice_test_once', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ duration }) });
                const data = await res.json();
                document.getElementById('voice-test-status').textContent = data.message || JSON.stringify(data);
            } catch(e) { document.getElementById('voice-test-status').textContent = 'Recognition test failed: ' + e; }
        }
        async function saveSpeechSettings() {
            const device = document.getElementById('speech-device-input').value || 'auto';
            try {
                const res = await fetch('/speech_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ device }) });
                const data = await res.json();
                document.getElementById('speech-message').textContent = data.message || 'Speech settings saved';
            } catch(e) { document.getElementById('speech-error').textContent = 'Save failed: ' + e; }
        }
        async function listSpeechDevices() {
            document.getElementById('speech-devices-list').textContent = 'Checking speakers...';
            try {
                const res = await fetch('/speech_devices');
                const data = await res.json();
                document.getElementById('speech-devices-list').textContent = data.devices || data.message || 'No output';
                if (data.suggested_device) {
                    document.getElementById('speech-device-input').value = data.suggested_device;
                    document.getElementById('speech-message').textContent = 'Suggested speaker selected: ' + data.suggested_device;
                }
            } catch(e) { document.getElementById('speech-devices-list').textContent = 'Speaker list failed: ' + e; }
        }
        async function setSpeechVolume() {
            document.getElementById('speech-message').textContent = 'Setting USB speaker volume to 85%...';
            try {
                const res = await fetch('/speech_set_volume', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ percent: 85 }) });
                const data = await res.json();
                document.getElementById('speech-message').textContent = data.message || 'Volume set';
                if (data.error) document.getElementById('speech-error').textContent = data.error;
            } catch(e) { document.getElementById('speech-error').textContent = 'Volume set failed: ' + e; }
        }
        async function testSpeakerBeep() {
            document.getElementById('speech-message').textContent = 'Playing speaker test beep...';
            try {
                const res = await fetch('/speech_test_beep', { method:'POST' });
                const data = await res.json();
                document.getElementById('speech-message').textContent = data.message || 'Beep test requested';
                if (data.error) document.getElementById('speech-error').textContent = data.error;
            } catch(e) { document.getElementById('speech-error').textContent = 'Beep test failed: ' + e; }
        }
        async function testAllSpeakers() {
            document.getElementById('speech-message').textContent = 'Trying all likely speaker devices... listen for a beep.';
            try {
                const res = await fetch('/speech_test_all', { method:'POST' });
                const data = await res.json();
                document.getElementById('speech-message').textContent = data.message || 'Trying all speakers';
                if (data.error) document.getElementById('speech-error').textContent = data.error;
            } catch(e) { document.getElementById('speech-error').textContent = 'Try all speakers failed: ' + e; }
        }
        async function speakTestPhrase() {
            const text = document.getElementById('speech-phrase-input').value || 'Hello, I am Janet. My speaker is working.';
            document.getElementById('speech-message').textContent = 'Trying to speak...';
            try {
                const res = await fetch('/speech_say', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ text }) });
                const data = await res.json();
                document.getElementById('speech-message').textContent = data.message || 'Speech requested';
                if (data.error) document.getElementById('speech-error').textContent = data.error;
            } catch(e) { document.getElementById('speech-error').textContent = 'Speech failed: ' + e; }
        }
        function renderScanResults(results) {
            const box = document.getElementById('det-scan-results');
            if (!box) return;
            if (!results || results.length === 0) {
                box.innerHTML = '<p class="motor-note">No local model files found. Candidate names are still shown by the scanner.</p>';
                return;
            }
            box.innerHTML = '';
            results.forEach(item => {
                const tagClass = item.status === 'current' ? '' : (item.status === 'local-file' ? 'warn' : 'bad');
                const tag = `<span class="scan-tag ${tagClass}">${item.status || 'candidate'}</span>`;
                const meta = item.path ? `<div class="motor-note">${item.path}</div>` : '';
                box.innerHTML += `<div class="scan-result"><b>${item.name || 'unknown'}</b>${tag}<div class="motor-note">${item.note || ''}</div>${meta}</div>`;
            });
        }
        async function loadDetectionInfo() {
            try {
                const res = await fetch('/detection_info');
                const data = await res.json();
                const m = data.model || {};
                document.getElementById('det-model-name').textContent = m.name || '-';
                document.getElementById('det-model-family').textContent = m.family || '-';
                document.getElementById('det-runtime').textContent = m.runtime || '-';
                document.getElementById('det-confidence').textContent = m.confidence === undefined ? '-' : Number(m.confidence).toFixed(2);
                document.getElementById('det-fps').textContent = (data.fps || 0).toFixed ? Number(data.fps || 0).toFixed(1) + ' FPS' : data.fps;
                document.getElementById('det-labels').textContent = (m.labels_count || 0) + ' labels' + (m.labels_preview && m.labels_preview.length ? ' — ' + m.labels_preview.join(', ') : '');
                document.getElementById('det-depthai-version').textContent = data.depthai_version || '-';
                document.getElementById('det-opencv-version').textContent = data.opencv_version || '-';
                document.getElementById('det-recommendation').textContent = data.recommendation || '-';
                document.getElementById('det-scan-status').textContent = m.last_scan_message || 'Ready.';
                renderScanResults(m.scan_results || []);
            } catch(e) {
                document.getElementById('det-scan-status').textContent = 'Could not load detection info: ' + e;
            }
        }
        async function scanDetectionModels() {
            document.getElementById('det-scan-status').textContent = 'Scanning local cache and candidate model names...';
            try {
                const res = await fetch('/detection_scan', { method:'POST' });
                const data = await res.json();
                document.getElementById('det-scan-status').textContent = data.message || 'Scan complete';
                renderScanResults(data.results || []);
                loadDetectionInfo();
            } catch(e) {
                document.getElementById('det-scan-status').textContent = 'Scan failed: ' + e;
            }
        }
        function renderFaceList(items) {
            const box = document.getElementById('face-list');
            if (!box) return;
            if (!items || items.length === 0) {
                box.innerHTML = '<p class="motor-note">No known faces yet.</p>';
                return;
            }
            box.innerHTML = '';
            items.forEach(item => {
                const safeName = String(item.name || '').replace(/'/g, "\\'");
                const images = item.images || [];
                let thumbs = '';
                if (images.length > 0) {
                    thumbs = '<div class="face-thumb-grid">' + images.map((img, idx) => {
                        const src = img.thumb_url || img.url || '';
                        const href = img.url || src;
                        const sampleIndex = img.sample_index !== undefined ? img.sample_index : ((img.sample || 1) - 1);
                        const label = (item.name || 'face') + ' sample ' + (idx + 1);
                        return `<div class="thumb-wrap"><a href="${href}" target="_blank" title="${label}"><img class="face-thumb" src="${src}" alt="${label}"></a><button class="thumb-x" title="Delete this sample" onclick="removeFaceSample(event, ${sampleIndex})">×</button></div>`;
                    }).join('') + '</div>';
                } else {
                    thumbs = '<div class="motor-note">No saved thumbnail yet. Add another sample to create one.</div>';
                }
                box.innerHTML += `<div class="face-list-item"><div><b>${item.name}</b><br><span class="motor-note">${item.samples} sample(s)</span>${thumbs}</div><button class="small-danger" onclick="removeKnownFace('${safeName}')">Remove all</button></div>`;
            });
        }
        function renderSeenFaces(items) {
            const box = document.getElementById('face-seen-list');
            if (!box) return;
            if (!items || items.length === 0) {
                box.innerHTML = '<p class="motor-note">No faces seen yet.</p>';
                return;
            }
            box.innerHTML = '';
            items.forEach(face => {
                const score = face.score === null || face.score === undefined ? '-' : face.score;
                box.innerHTML += `<div class="scan-result"><b>${face.name || 'Unknown'}</b><span class="scan-tag ${face.name === 'Unknown' ? 'warn' : ''}">score ${score}</span><div class="motor-note">x:${face.x} y:${face.y} size:${face.w}x${face.h}</div></div>`;
            });
        }
        async function loadFaceInfo() {
            try {
                const res = await fetch('/face_info');
                const data = await res.json();
                const f = data.face || {};
                document.getElementById('face-status').textContent = f.status || '-';
                document.getElementById('face-available').textContent = f.available ? 'yes' : 'no';
                document.getElementById('face-known-count').textContent = f.known_count || 0;
                document.getElementById('face-storage').textContent = f.storage || '-';
                document.getElementById('face-greeting-status').textContent = f.greeting_status || '-';
                document.getElementById('face-last-greeting').textContent = f.last_greeting || '-';
                document.getElementById('face-speech-greeting-status').textContent = f.speech_greeting_status || '-';
                document.getElementById('face-last-speech-greeting').textContent = f.last_speech_greeting || '-';
                document.getElementById('face-message').textContent = f.last_message || 'Ready.';
                document.getElementById('face-enabled-input').value = f.enabled ? 'true' : 'false';
                document.getElementById('face-threshold-input').value = f.threshold || 55;
                document.getElementById('face-threshold-label').textContent = f.threshold || 55;
                renderFaceList(f.known_faces || []);
                renderSeenFaces(f.last_seen || []);
            } catch(e) {
                document.getElementById('face-message').textContent = 'Could not load face info: ' + e;
            }
        }
        async function saveFaceSettings() {
            const enabled = document.getElementById('face-enabled-input').value === 'true';
            const threshold = Number(document.getElementById('face-threshold-input').value || 55);
            try {
                const res = await fetch('/face_settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ enabled, threshold }) });
                const data = await res.json();
                document.getElementById('face-message').textContent = data.message || 'Face settings saved';
                loadFaceInfo();
            } catch(e) { document.getElementById('face-message').textContent = 'Save failed: ' + e; }
        }
        async function addKnownFace() {
            const name = document.getElementById('face-name-input').value || '';
            document.getElementById('face-message').textContent = 'Capturing face from front camera...';
            try {
                const res = await fetch('/face_add', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ name }) });
                const data = await res.json();
                document.getElementById('face-message').textContent = data.message || 'Done';
                loadFaceInfo();
            } catch(e) { document.getElementById('face-message').textContent = 'Add failed: ' + e; }
        }
        async function removeKnownFace(name) {
            if (!confirm('Remove all samples for ' + name + '?')) return;
            try {
                const res = await fetch('/face_remove', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ name }) });
                const data = await res.json();
                document.getElementById('face-message').textContent = data.message || 'Removed';
                loadFaceInfo();
            } catch(e) { document.getElementById('face-message').textContent = 'Remove failed: ' + e; }
        }
        async function removeFaceSample(ev, sampleIndex) {
            if (ev) { ev.preventDefault(); ev.stopPropagation(); }
            if (!confirm('Delete just this face sample?')) return;
            try {
                const res = await fetch('/face_remove_sample', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ sample_index: sampleIndex }) });
                const data = await res.json();
                document.getElementById('face-message').textContent = data.message || 'Sample removed';
                loadFaceInfo();
            } catch(e) { document.getElementById('face-message').textContent = 'Sample remove failed: ' + e; }
        }
        function renderObjectList(items) {
            const box = document.getElementById('object-list');
            if (!box) return;
            if (!items || items.length === 0) {
                box.innerHTML = '<p class="motor-note">No known objects yet. Janet will add them automatically when she says hello to objects.</p>';
                return;
            }
            box.innerHTML = '';
            items.forEach(item => {
                const safeLabel = String(item.label || '').replace(/'/g, "\\'");
                const images = item.images || [];
                let thumbs = '';
                if (images.length > 0) {
                    thumbs = '<div class="face-thumb-grid">' + images.map((img, idx) => {
                        const src = img.thumb_url || img.url || '';
                        const href = img.url || src;
                        const sampleIndex = img.sample_index !== undefined ? img.sample_index : ((img.sample || 1) - 1);
                        const id = img.id || '';
                        const label = (item.label || 'object') + ' sample ' + (idx + 1);
                        return `<div class="thumb-wrap"><a href="${href}" target="_blank" title="${label}"><img class="face-thumb" src="${src}" alt="${label}"></a><button class="thumb-x" title="Delete this object sample" onclick="removeObjectSample(event, ${sampleIndex}, '${id}')">×</button></div>`;
                    }).join('') + '</div>';
                } else {
                    thumbs = '<div class="motor-note">No saved thumbnail yet.</div>';
                }
                box.innerHTML += `<div class="face-list-item"><div><b>${item.label}</b><br><span class="motor-note">${item.samples} sample(s)</span>${thumbs}</div><button class="small-danger" onclick="removeKnownObject('${safeLabel}')">Remove all</button></div>`;
            });
        }
        function renderSeenObjects(items) {
            const box = document.getElementById('object-seen-list');
            if (!box) return;
            if (!items || items.length === 0) {
                box.innerHTML = '<p class="motor-note">No objects seen yet.</p>';
                return;
            }
            box.innerHTML = '';
            items.forEach(obj => {
                box.innerHTML += `<div class="scan-result"><b>${obj.label || 'object'}</b><span class="scan-tag">${obj.confidence || 0}%</span><div class="motor-note">x:${Number(obj.xmin || 0).toFixed(2)} → ${Number(obj.xmax || 0).toFixed(2)}</div></div>`;
            });
        }
        async function loadObjectInfo() {
            try {
                const res = await fetch('/object_info');
                const data = await res.json();
                const o = data.object || {};
                document.getElementById('object-status').textContent = o.status || '-';
                document.getElementById('object-known-count').textContent = o.known_count || 0;
                document.getElementById('object-samples-count').textContent = o.samples_count || 0;
                document.getElementById('object-cooldown').textContent = Math.round((o.acknowledge_cooldown_seconds || 1800) / 60) + ' minutes';
                document.getElementById('object-scan-status').textContent = o.scan_status || '-';
                document.getElementById('object-last-return').textContent = o.last_scan_return || '-';
                document.getElementById('object-storage').textContent = o.storage || '-';
                document.getElementById('object-message').textContent = o.last_message || 'Ready.';
                renderObjectList(o.known_objects || []);
                renderSeenObjects(o.last_seen || []);
            } catch(e) {
                document.getElementById('object-message').textContent = 'Could not load object info: ' + e;
            }
        }
        async function removeObjectSample(ev, sampleIndex, sampleId) {
            if (ev) { ev.preventDefault(); ev.stopPropagation(); }
            if (!confirm('Delete just this object sample?')) return;
            try {
                const res = await fetch('/object_remove_sample', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ sample_index: sampleIndex, sample_id: sampleId }) });
                const data = await res.json();
                document.getElementById('object-message').textContent = data.message || 'Object sample removed';
                loadObjectInfo();
            } catch(e) { document.getElementById('object-message').textContent = 'Object sample remove failed: ' + e; }
        }
        async function removeKnownObject(label) {
            if (!confirm('Remove all samples for ' + label + '?')) return;
            try {
                const res = await fetch('/object_remove', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ label }) });
                const data = await res.json();
                document.getElementById('object-message').textContent = data.message || 'Object removed';
                loadObjectInfo();
            } catch(e) { document.getElementById('object-message').textContent = 'Object remove failed: ' + e; }
        }
        function getColorClass(val) {
            if (val < 0 || val < 20) return 'bad';
            if (val < 50) return 'warn';
            return 'ok';
        }
        async function updateReadings() {
            try {
                const res = await fetch('/readings');
                const data = await res.json();
                const sonar = data.sonar || {};
                const sensors = { 'f': sonar.front, 'b': sonar.back, 'l': sonar.left, 'r': sonar.right };
                for (const [id, val] of Object.entries(sensors)) {
                    const el = document.getElementById(id);
                    el.textContent = (val === undefined || val === null) ? '-' : val + 'cm';
                    el.className = 'val ' + getColorClass(val);
                }
                const voice = data.voice || {};
                const fps = (data.fps === undefined || data.fps === null) ? 0 : Number(data.fps);
                document.getElementById('detections-title').textContent = 'Detections (' + fps.toFixed(1) + ' FPS)';
                document.getElementById('voice-status').textContent = voice.status || '-';
                document.getElementById('voice-heard').textContent = voice.last_heard || '-';
                document.getElementById('voice-action').textContent = voice.last_action || voice.last_error || '-';
                document.getElementById('voice-device').textContent = voice.voice_device || '-';
                const level = Number(voice.mic_level || 0);
                const db = voice.mic_level_db === null || voice.mic_level_db === undefined ? '-' : voice.mic_level_db + ' dB';
                document.getElementById('voice-level-text').textContent = level.toFixed(0) + '% / ' + db;
                document.getElementById('voice-level-fill').style.width = Math.max(0, Math.min(100, level)) + '%';
                if (voice.voice_device) { document.getElementById('voice-device-input').value = voice.voice_device; }
                if (voice.last_test_message) { document.getElementById('voice-test-status').textContent = voice.last_test_message; }
                const wakeEl = document.getElementById('voice-wake');
                wakeEl.textContent = voice.wake_word_active ? 'JANET heard' : 'Waiting';
                wakeEl.className = voice.wake_word_active ? 'wake-active' : '';
                const soundEl = document.getElementById('voice-sound');
                soundEl.textContent = voice.sound_active ? 'SOUND heard' : (voice.last_sound_message || 'Waiting');
                soundEl.className = voice.sound_active ? 'sound-active' : '';
                if (voice.last_raw_file || voice.last_boosted_file) {
                    document.getElementById('voice-debug-files').textContent = 'Raw: ' + (voice.last_raw_file || '-') + ' | Boosted: ' + (voice.last_boosted_file || '-');
                }
                if (voice.sample_message) { document.getElementById('voice-record-status').textContent = voice.sample_message; }
                const speech = data.speech || {};
                if (document.getElementById('speech-status')) {
                    document.getElementById('speech-status').textContent = speech.status || '-';
                    document.getElementById('speech-device').textContent = speech.speaker_device || '-';
                    document.getElementById('speech-message').textContent = speech.last_message || '-';
                    document.getElementById('speech-error').textContent = speech.last_error || '-';
                    document.getElementById('speech-phrase').textContent = speech.last_phrase || '-';
                    document.getElementById('speech-tts').textContent = speech.tts_engine || (speech.espeak_available ? 'espeak available' : 'built-in robot voice fallback');
                    document.getElementById('speech-object-greeting').textContent = speech.object_greeting_status || speech.last_object_greeting || '-';
                    document.getElementById('speech-centering').textContent = speech.centering_status || speech.last_centering_action || '-';
                    document.getElementById('speech-volume').textContent = speech.volume_status || '-';
                    if (speech.speaker_device) { document.getElementById('speech-device-input').value = speech.speaker_device; }
                }
                const detList = document.getElementById('detections-list');
                detList.innerHTML = '';
                if (data.detections && data.detections.length > 0) {
                    data.detections.forEach(det => {
                        detList.innerHTML += `
                            <div class="det-card">
                                <div class="det-label">${det.label}</div>
                                <div class="det-conf">${det.confidence}% confidence</div>
                            </div>
                        `;
                    });
                } else {
                    detList.innerHTML = '<p style="color:#9ca7bd; font-size:14px;">No detections</p>';
                }
                const liveDet = document.getElementById('det-live-list');
                if (liveDet) {
                    liveDet.innerHTML = detList.innerHTML;
                }
                const detFps = document.getElementById('det-fps');
                if (detFps) { detFps.textContent = fps.toFixed(1) + ' FPS'; }
                const routines = data.routines || {};
                if (document.getElementById('routine-status')) {
                    updateRoutinePanel(routines);
                }
                const face = data.face || {};
                if (document.getElementById('face-status')) {
                    document.getElementById('face-status').textContent = face.status || '-';
                    document.getElementById('face-available').textContent = face.available ? 'yes' : 'no';
                    document.getElementById('face-known-count').textContent = face.known_count || 0;
                    document.getElementById('face-storage').textContent = face.storage || '-';
                    document.getElementById('face-greeting-status').textContent = face.greeting_status || '-';
                    document.getElementById('face-last-greeting').textContent = face.last_greeting || '-';
                    document.getElementById('face-speech-greeting-status').textContent = face.speech_greeting_status || '-';
                    document.getElementById('face-last-speech-greeting').textContent = face.last_speech_greeting || '-';
                    document.getElementById('face-message').textContent = face.last_message || 'Ready.';
                    renderFaceList(face.known_faces || []);
                    renderSeenFaces(face.last_seen || []);
                }
                const objectInfo = data.object || {};
                if (document.getElementById('object-status')) {
                    document.getElementById('object-status').textContent = objectInfo.status || '-';
                    document.getElementById('object-known-count').textContent = objectInfo.known_count || 0;
                    document.getElementById('object-samples-count').textContent = objectInfo.samples_count || 0;
                    document.getElementById('object-cooldown').textContent = Math.round((objectInfo.acknowledge_cooldown_seconds || 1800) / 60) + ' minutes';
                    document.getElementById('object-scan-status').textContent = objectInfo.scan_status || '-';
                    document.getElementById('object-last-return').textContent = objectInfo.last_scan_return || '-';
                    document.getElementById('object-storage').textContent = objectInfo.storage || '-';
                    document.getElementById('object-message').textContent = objectInfo.last_message || 'Ready.';
                    renderObjectList(objectInfo.known_objects || []);
                    renderSeenObjects(objectInfo.last_seen || []);
                }
            } catch (e) {}
        }
        setInterval(updateReadings, 500);
        bindMotorButtons();
        loadMotorSettings();
        loadRoutineInfo();
    </script>
</body>
</html>
""")


def format_cm(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if value < 0:
        return "unavailable"
    return f"{value:.2f} cm"


def build_service_report():
    """Human-readable status summary for service/API requests."""
    with lock:
        data = dict(latest_readings)
        sonar = dict(data.get("sonar", {}))
        detections = list(data.get("detections", []))
        voice = dict(data.get("voice", {}))
        face = dict(data.get("face", {}))
        fps = float(data.get("fps", 0) or 0)
        status = data.get("status", "unknown")

    camera_active = front_camera_ready.is_set() or fps > 0
    if camera_active:
        fps_line = f"{fps:.1f} (The camera is now fully active)."
    else:
        fps_line = f"{fps:.1f} (The camera is starting or not active yet: {status})."

    if detections:
        objects_line = "She currently sees: " + ", ".join(f"{d.get('label', 'object')} ({d.get('confidence', '?')}%)" for d in detections[:6]) + "."
    else:
        objects_line = "She is scanning, but no specific objects are currently being detected."

    last_seen_faces = face.get("last_seen") or []
    face_enabled = bool(face.get("enabled"))
    if last_seen_faces:
        face_names = ", ".join(f.get("name", "Unknown") for f in last_seen_faces[:6])
        faces_line = f"The system is {'Enabled' if face_enabled else 'Disabled'} and currently sees: {face_names}."
    else:
        faces_line = f"The system is {'Enabled' if face_enabled else 'Disabled'} and ready, but no faces have been spotted yet."

    heard = voice.get("last_heard") or ""
    heard_line = f"Someone just asked her: \"{heard}\"!" if heard else "Nothing has been recognised yet."
    mic_level = int(float(voice.get("mic_level") or 0))
    mic_db = voice.get("mic_level_db")
    db_text = f"{float(mic_db):.1f} dB" if mic_db is not None else "unknown dB"
    if mic_level >= 45:
        signal_note = "a much stronger signal than before."
    elif mic_level >= 15:
        signal_note = "a usable signal, but it may still be a little quiet."
    else:
        signal_note = "a weak or quiet signal."

    lower_heard = heard.lower()
    action = voice.get("last_action") or voice.get("last_error") or "No action recorded."
    if heard and VOICE_WAKE_WORD not in lower_heard:
        action_line = f"Because there was no \"Janet\" wake-word, she didn't respond."
    elif heard and VOICE_WAKE_WORD in lower_heard:
        action_line = f"Wake-word detected. Action: {action}"
    else:
        action_line = action

    return "\n".join([
        "📷 Vision & Detections",
        f"*   FPS: {fps_line}",
        f"*   Objects: {objects_line}",
        f"*   Faces: {faces_line}",
        "",
        "📡 Sonar (Distances)",
        f"*   Front: {format_cm(sonar.get('front'))}",
        f"*   Back: {format_cm(sonar.get('back'))}",
        f"*   Left: {format_cm(sonar.get('left'))}",
        f"*   Right: {format_cm(sonar.get('right'))}",
        "",
        "🎤 Voice Status",
        f"*   Last Heard: {heard_line}",
        f"*   Sound Level: {mic_level}% ({db_text}) — {signal_note}",
        f"*   Action: {action_line}",
    ]) + "\n"


@app.route("/screenshot_front")
def screenshot_front_route():
    start_front_camera_worker()
    with front_stream_condition:
        jpeg = front_latest_jpeg
    if jpeg is None:
        frame = make_front_placeholder_frame()
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            return jsonify({"status": "error", "message": "No screenshot frame available yet"}), 503
        jpeg = encoded.tobytes()
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    filename = f"{SCREENSHOT_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(SCREENSHOT_DIR, filename)
    with open(path, "wb") as f:
        f.write(jpeg)
    return send_file(path, mimetype="image/jpeg", as_attachment=True, download_name=filename)


@app.route("/service_report")
@app.route("/service_status")
@app.route("/api/service_report")
def service_report_route():
    return Response(build_service_report(), mimetype="text/plain; charset=utf-8")


@app.route("/service_report.json")
def service_report_json_route():
    with lock:
        data = dict(latest_readings)
    return jsonify({
        "status": "ok",
        "report": build_service_report(),
        "camera_active": front_camera_ready.is_set(),
        "frame_age_seconds": round(time.time() - front_latest_frame_time, 2) if front_latest_frame_time else None,
        "readings": data,
    })

@app.route("/video_front")
def video_front(): return Response(generate_front_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_rear")
def video_rear(): return Response(generate_rear_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


def scan_local_detection_models():
    """Safely inspect local model/cache files and candidate model names.

    This does not start a second DepthAI pipeline and does not download or switch
    models. It is designed to be safe while Janet's video feed is running.
    """
    results = []
    seen = set()

    def add_result(name, status, note, path=""):
        key = (name, status, path)
        if key in seen:
            return
        seen.add(key)
        results.append({"name": name, "status": status, "note": note, "path": path})

    add_result(DETECTION_MODEL_NAME, "current", "Currently running in the stable OAK-D detection pipeline.")

    # Candidate names only prove that Janet knows about them; they still need a
    # controlled test before becoming the running model.
    for candidate in DETECTION_MODEL_CANDIDATES:
        name = candidate.get("name", "")
        try:
            # Creating the description is lightweight. Avoid pipeline.build/start here.
            _ = dai.NNModelDescription(name)
            add_result(name, candidate.get("type", "candidate"), candidate.get("note", "Candidate model name accepted by DepthAI description object."))
        except Exception as e:
            add_result(name, "candidate", f"Candidate listed, but descriptor check raised: {e}")

    # Scan common local cache/document folders for model artefacts.
    suffixes = (".blob", ".onnx", ".json", ".yaml", ".yml", ".xml", ".bin")
    max_files = 60
    found = 0
    for base in DETECTION_CACHE_DIRS:
        base = os.path.expanduser(base)
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            # Avoid deep or irrelevant directories.
            depth = root[len(base):].count(os.sep)
            if depth > 5:
                dirs[:] = []
                continue
            lowered_root = root.lower()
            relevant_root = any(word in lowered_root for word in ("depthai", "luxonis", "model", "yolo", "blob", "nn"))
            for filename in files:
                low = filename.lower()
                if not low.endswith(suffixes):
                    continue
                path = os.path.join(root, filename)
                relevant_file = any(word in low for word in ("yolo", "depth", "blob", "model", "nn"))
                if relevant_root or relevant_file:
                    add_result(filename, "local-file", "Local model/cache artefact found. This may be useful for a future controlled model test.", path)
                    found += 1
                    if found >= max_files:
                        break
            if found >= max_files:
                break
        if found >= max_files:
            break

    return results


def detection_info_payload():
    with lock:
        model = dict(latest_readings.get("detection_model", {}))
        fps = latest_readings.get("fps", 0)
        detections = list(latest_readings.get("detections", []))
    recommendation = (
        "Keep yolov6-nano as Janet's stable baseline. For better accuracy, test one candidate at a time "
        "in a copied file, starting with yolov6-tiny or another nano/small model. Do not hot-swap while the robot is moving."
    )
    return {
        "status": "ok",
        "model": model,
        "fps": fps,
        "detections": detections,
        "depthai_version": getattr(dai, "__version__", "unknown"),
        "opencv_version": getattr(cv2, "__version__", "unknown"),
        "python_version": sys.version.split()[0],
        "recommendation": recommendation,
    }


@app.route("/detection_info")
def detection_info_route():
    return jsonify(detection_info_payload())


@app.route("/detection_scan", methods=["POST"])
def detection_scan_route():
    results = scan_local_detection_models()
    message = f"Scan complete: {len(results)} candidate/cache item(s) listed. No running model was changed."
    with lock:
        latest_readings.setdefault("detection_model", {})["scan_results"] = results
        latest_readings.setdefault("detection_model", {})["last_scan_message"] = message
    return jsonify({"status": "ok", "message": message, "results": results})


@app.route("/face_image/<path:filename>")
def face_image_route(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(FACE_IMAGE_DIR, safe_name)
    if not safe_name or not os.path.exists(path):
        return ("", 404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/face_info")
def face_info_route():
    with lock:
        face = dict(latest_readings.get("face", {}))
    return jsonify({"status": "ok", "face": face})


@app.route("/face_settings", methods=["POST"])
def face_settings_route():
    payload = request.get_json(silent=True) or {}
    settings = set_face_settings(enabled=payload.get("enabled"), threshold=payload.get("threshold"))
    return jsonify({"status": "ok", "settings": settings, "message": "Face settings saved"})


@app.route("/face_add", methods=["POST"])
def face_add_route():
    payload = request.get_json(silent=True) or {}
    ok, message = add_known_face_from_latest_frame(payload.get("name"))
    code = 200 if ok else 400
    return jsonify({"status": "ok" if ok else "error", "message": message, "known_faces": known_face_summary()}), code


@app.route("/face_remove", methods=["POST"])
def face_remove_route():
    payload = request.get_json(silent=True) or {}
    ok, message = remove_known_face(payload.get("name"))
    code = 200 if ok else 404
    return jsonify({"status": "ok" if ok else "error", "message": message, "known_faces": known_face_summary()}), code


@app.route("/face_remove_sample", methods=["POST"])
def face_remove_sample_route():
    payload = request.get_json(silent=True) or {}
    ok, message = remove_known_face_sample(sample_index=payload.get("sample_index"), sample_id=payload.get("sample_id"))
    code = 200 if ok else 404
    return jsonify({"status": "ok" if ok else "error", "message": message, "known_faces": known_face_summary()}), code


@app.route("/object_image/<path:filename>")
def object_image_route(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(OBJECT_IMAGE_DIR, safe_name)
    if not safe_name or not os.path.exists(path):
        return ("", 404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/object_info")
def object_info_route():
    with lock:
        obj = dict(latest_readings.get("object", {}))
    return jsonify({"status": "ok", "object": obj})


@app.route("/object_remove_sample", methods=["POST"])
def object_remove_sample_route():
    payload = request.get_json(silent=True) or {}
    ok, message = remove_known_object_sample(sample_index=payload.get("sample_index"), sample_id=payload.get("sample_id"))
    code = 200 if ok else 404
    return jsonify({"status": "ok" if ok else "error", "message": message, "known_objects": known_object_summary()}), code


@app.route("/object_remove", methods=["POST"])
def object_remove_route():
    payload = request.get_json(silent=True) or {}
    ok, message = remove_known_object(payload.get("label") or payload.get("name"))
    code = 200 if ok else 404
    return jsonify({"status": "ok" if ok else "error", "message": message, "known_objects": known_object_summary()}), code


@app.route("/readings")
def readings():
    routine_payload = get_routine_status_payload()
    with lock:
        latest_readings.setdefault("voice", {})["wake_word_active"] = time.time() < voice_wake_overlay_until
        latest_readings.setdefault("voice", {})["sound_active"] = time.time() < voice_sound_overlay_until
        data = dict(latest_readings)
        data["motors_available"] = motors is not None
        data["reverse_motor_logic"] = REVERSE_MOTOR_LOGIC
        data["rear_camera_enabled"] = REAR_CAMERA_ENABLED
        data["rear_camera_indexes"] = REAR_CAMERA_INDEXES
        data["motor_settings"] = get_motor_settings()
        data["routines"] = routine_payload
        data["voice_settings"] = get_voice_config()
        data["speech_settings"] = get_speech_config()
        data["face_settings"] = dict(face_settings)
        data["front_camera_auto_start"] = FRONT_CAMERA_AUTO_START
        data["front_camera_ready"] = front_camera_ready.is_set()
        data["front_camera_error"] = front_camera_error
        data["front_frame_age_seconds"] = round(time.time() - front_latest_frame_time, 2) if front_latest_frame_time else None
        return jsonify(data)

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/routine_info")
def routine_info_route():
    return jsonify({"status": "ok", "routines": get_routine_status_payload()})


@app.route("/routine_start/<routine_id>", methods=["POST"])
def routine_start_route(routine_id):
    ok, message = start_motor_routine(routine_id)
    code = 200 if ok else 409
    return jsonify({"status": "ok" if ok else "error", "message": message, "routines": get_routine_status_payload()}), code


@app.route("/routine_stop", methods=["POST"])
def routine_stop_route():
    stop_motor_routine()
    return jsonify({"status": "ok", "message": "Routine stop requested", "routines": get_routine_status_payload()})


@app.route("/motor_settings", methods=["GET", "POST"])
def motor_settings_route():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        settings = set_motor_settings(
            duration=payload.get("duration"),
            acceleration=payload.get("acceleration"),
            preset="custom",
        )
        return jsonify({**settings, "motors_available": motors is not None, "message": "Motor settings saved"})
    settings = get_motor_settings()
    return jsonify({**settings, "motors_available": motors is not None, "presets": MOTOR_PRESETS})


@app.route("/motor_preset/<preset>", methods=["POST"])
def motor_preset_route(preset):
    if preset not in MOTOR_PRESETS:
        return jsonify({"status": "error", "message": "Unknown preset"}), 404
    chosen = MOTOR_PRESETS[preset]
    settings = set_motor_settings(
        duration=chosen["duration"],
        acceleration=chosen["acceleration"],
        preset=preset,
    )
    return jsonify({**settings, "status": "ok", "message": f"{chosen['label']} applied"})



@app.route("/speech_settings", methods=["GET", "POST"])
def speech_settings_route():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        settings = set_speech_config(device=payload.get("device"), rate=payload.get("rate"), channels=payload.get("channels"))
        update_speech_status(speaker_device=settings["device"], last_error="", last_message=f"Speaker device set to {settings['device']}")
        return jsonify({**settings, "status": "ok", "message": f"Speaker device set to {settings['device']}"})
    return jsonify(get_speech_config())


@app.route("/speech_devices")
def speech_devices_route():
    parts = []
    for args, title in [(["aplay", "-l"], "aplay -l"), (["aplay", "-L"], "aplay -L")]:
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=5)
            output = (result.stdout or result.stderr or "").strip()
            parts.append(f"===== {title} =====\n{output or '(no output)'}")
        except Exception as e:
            parts.append(f"===== {title} =====\nERROR: {e}")
    candidates = list_aplay_output_candidates()
    suggested = detect_aplay_output_device()
    current = get_speech_config().get("device", "auto")
    if current in {"auto", "default"} or "vc4hdmi" in current.lower() or "hdmi" in current.lower():
        set_speech_config(device=suggested)
    candidate_lines = []
    for item in candidates:
        flag = "SKIP HDMI" if item.get("avoid") else "TRY"
        candidate_lines.append(f"{flag:9s} score={item.get('score', 0):>4}  {item.get('device')}  {item.get('description', '')}")
    devices = "\n\n".join(parts) + f"\n\n===== Janet ranked speaker candidates =====\n" + ("\n".join(candidate_lines) or "No candidates found") + f"\n\n===== Janet suggested speaker =====\n{suggested}"
    update_speech_status(aplay_devices=devices, speaker_device=get_speech_config().get("device"), last_message=f"Suggested speaker: {suggested}")
    return jsonify({"status": "ok", "devices": devices, "suggested_device": suggested, "candidates": candidates})


@app.route("/speech_set_volume", methods=["POST"])
def speech_set_volume_route():
    payload = request.get_json(silent=True) or {}
    percent = payload.get("percent", SPEECH_VOLUME_PERCENT)
    ok, message = set_speaker_volume(percent=percent, device=get_speech_config().get("device"), quiet=False)
    code = 200 if ok else 500
    return jsonify({"status": "ok" if ok else "error", "message": message, "error": "" if ok else message, "speech": latest_readings.get("speech", {})}), code


@app.route("/speech_test_beep", methods=["POST"])
def speech_test_beep_route():
    worker = threading.Thread(target=speech_test_beep_worker, daemon=True)
    worker.start()
    return jsonify({"status": "playing", "message": "Speaker beep test started", "speech": latest_readings.get("speech", {})})


@app.route("/speech_test_all", methods=["POST"])
def speech_test_all_route():
    worker = threading.Thread(target=speech_test_all_speakers_worker, daemon=True)
    worker.start()
    return jsonify({"status": "testing", "message": "Trying all likely speaker devices", "speech": latest_readings.get("speech", {})})


@app.route("/speech_say", methods=["POST"])
def speech_say_route():
    payload = request.get_json(silent=True) or {}
    text = safe_speech_text(payload.get("text"))
    worker = threading.Thread(target=speech_say_worker, args=(text,), daemon=True)
    worker.start()
    return jsonify({"status": "speaking", "message": "Speech test started", "text": text, "speech": latest_readings.get("speech", {})})


@app.route("/voice_settings", methods=["GET", "POST"])
def voice_settings_route():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        settings = set_voice_config(device=payload.get("device"), rate=payload.get("rate"), channels=payload.get("channels"))
        update_voice_status(voice_device=settings["device"], last_error="")
        return jsonify({**settings, "status": "ok", "message": f"Voice device set to {settings['device']}"})
    return jsonify(get_voice_config())


@app.route("/voice_devices")
def voice_devices_route():
    parts = []
    for args, title in [(["arecord", "-l"], "arecord -l"), (["arecord", "-L"], "arecord -L")]:
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=5)
            output = (result.stdout or result.stderr or "").strip()
            parts.append(f"===== {title} =====\n{output or '(no output)'}")
        except Exception as e:
            parts.append(f"===== {title} =====\nERROR: {e}")
    suggested = detect_arecord_capture_device()
    # Make the diagnostic page useful immediately: if current device is auto/default, switch it.
    current = get_voice_config().get("device", "auto")
    if current in {"auto", "default"}:
        set_voice_config(device=suggested)
    devices = "\n\n".join(parts) + f"\n\n===== Janet suggested device =====\n{suggested}"
    update_voice_status(arecord_devices=devices, voice_device=get_voice_config().get("device"))
    return jsonify({"status": "ok", "devices": devices, "suggested_device": suggested})


@app.route("/voice_test_once", methods=["POST"])
def voice_test_once_route():
    # Stop the live listener first so the USB microphone is not busy.
    stop_live_voice_capture()
    if not SPEECH_AVAILABLE:
        return jsonify({"status": "error", "message": "SpeechRecognition is not installed"}), 500
    payload = request.get_json(silent=True) or {}
    duration = int(clamp_float(payload.get("duration"), 1, 8, 6))
    cfg = get_voice_config()
    os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
    test_path = os.path.join(VOICE_SAMPLE_DIR, ".janet_voice_test.wav")
    voice_recording_event.set()
    try:
        update_voice_status(status=f"voice test recording {duration}s", last_test_message="Recording test... say: Janet move forward", voice_device=cfg["device"], last_error="")
        rc, stdout, stderr, interrupted = run_arecord_capture(test_path, duration, cfg, interruptible=False)
        if rc != 0:
            err = (stderr or stdout or "arecord failed").strip()
            suggested = detect_arecord_capture_device()
            if suggested and suggested != cfg.get("device"):
                set_voice_config(device=suggested)
                cfg = get_voice_config()
                update_voice_status(status=f"retrying with {cfg['device']}", voice_device=cfg["device"], last_test_message=f"Default failed; retrying with {cfg['device']}")
                rc, stdout, stderr, interrupted = run_arecord_capture(test_path, duration, cfg, interruptible=False)
                if rc != 0:
                    err = (stderr or stdout or "arecord failed").strip()
                    update_voice_status(status="voice test failed", last_error=err, last_test_message=err)
                    return jsonify({"status": "error", "message": err, "suggested_device": suggested}), 500
            else:
                update_voice_status(status="voice test failed", last_error=err, last_test_message=err)
                return jsonify({"status": "error", "message": err}), 500
        stats = wav_audio_stats(test_path)
        if stats.get("level", 0) >= VOICE_SOUND_LEVEL_THRESHOLD:
            mark_sound_heard(stats)
        recognizer = sr.Recognizer()
        try:
            audio = make_google_audio_from_wav(recognizer, test_path)
            text = recognizer.recognize_google(audio).lower().strip()
        except sr.UnknownValueError:
            text = ""
        if VOICE_WAKE_WORD in text:
            mark_wake_word_heard(text)
        direction = parse_voice_command(text)
        msg = f"Level {stats.get('level', 0)}% / {stats.get('db', -120)} dB, adaptive target {VOICE_TARGET_DB} dB. Heard: {text or '(nothing recognised)'}"
        if direction:
            msg += f". Command detected: {direction}"
        update_voice_status(status="listening via arecord", last_heard=text, last_test_text=text, last_test_message=msg, mic_level=stats.get("level", 0), mic_level_db=stats.get("db", -120.0), last_sound_message=f"Sound heard at {stats.get('level', 0)}% / {stats.get('db', -120)} dB" if stats.get("level", 0) >= VOICE_SOUND_LEVEL_THRESHOLD else "Too quiet / no clear sound", last_error="" if text else "Could not understand audio")
        return jsonify({"status": "ok", "message": msg, "heard": text, "direction": direction, "level": stats, "device": cfg["device"]})
    except Exception as e:
        msg = str(e)
        update_voice_status(status="voice test error", last_error=msg, last_test_message=msg)
        return jsonify({"status": "error", "message": msg}), 500
    finally:
        voice_recording_event.clear()
        try:
            if os.path.exists(test_path): os.remove(test_path)
        except Exception:
            pass


@app.route("/voice_record_sample", methods=["POST"])
def voice_record_sample_route():
    if voice_recording_event.is_set():
        return jsonify({"status": "busy", "message": "Already recording a voice sample"}), 409
    payload = request.get_json(silent=True) or {}
    duration = int(clamp_float(payload.get("duration"), 1, VOICE_SAMPLE_MAX_DURATION, VOICE_SAMPLE_DEFAULT_DURATION))
    filename = safe_sample_filename(payload.get("filename"))
    worker = threading.Thread(target=record_voice_sample_worker, args=(duration, filename), daemon=True)
    worker.start()
    return jsonify({
        "status": "recording",
        "duration": duration,
        "filename": filename,
        "message": f"Recording {duration}s sample to {filename}",
    })

@app.route("/move/<direction>")
def control_robot(direction):
    duration = request.args.get("duration")
    acceleration = request.args.get("acceleration")
    ok, message = execute_move_async(direction, duration=duration, acceleration=acceleration)
    status = "ok" if ok else "error"
    code = 200 if ok else 500
    return jsonify({"status": status, "direction": direction, "message": message, "motor_settings": get_motor_settings()}), code

if __name__ == "__main__":
    try:
        init_face_system()
    except Exception as e:
        print(f"Face system init failed: {e}")
        update_face_status(status="init failed", available=False, last_message=str(e))
    try:
        load_known_objects()
    except Exception as e:
        print(f"Object memory init failed: {e}")
        update_object_status(status="init failed", available=False, last_message=str(e))
    sonar_thread = threading.Thread(target=sonar_loop, daemon=True)
    sonar_thread.start()
    if FRONT_CAMERA_AUTO_START:
        start_front_camera_worker()
    try:
        detected_voice_device = normalise_voice_device(get_voice_config().get("device", "auto"))
        set_voice_config(device=detected_voice_device)
        print(f"Voice capture device: {detected_voice_device}")
        update_voice_status(voice_device=detected_voice_device)
    except Exception as e:
        print(f"Voice device auto-select failed: {e}")
    try:
        detected_speech_device = normalise_speech_device(get_speech_config().get("device", "auto"))
        set_speech_config(device=detected_speech_device)
        print(f"Speech playback device: {detected_speech_device}")
        set_speaker_volume(SPEECH_VOLUME_PERCENT, device=detected_speech_device, quiet=True)
        update_speech_status(speaker_device=detected_speech_device, status="ready", last_message="Speech output ready on Y02 USB speaker")
    except Exception as e:
        print(f"Speech device auto-select failed: {e}")
        update_speech_status(status="speaker auto-select failed", last_error=str(e))
    voice_worker_thread = threading.Thread(target=voice_motor_worker, daemon=True)
    voice_worker_thread.start()
    voice_listener_thread = threading.Thread(target=voice_listener_loop, daemon=True)
    voice_listener_thread.start()
    face_greeting_thread = threading.Thread(target=face_greeting_worker, daemon=True)
    face_greeting_thread.start()
    face_speech_greeting_thread = threading.Thread(target=face_speech_greeting_worker, daemon=True)
    face_speech_greeting_thread.start()
    object_speech_greeting_thread = threading.Thread(target=object_speech_greeting_worker, daemon=True)
    object_speech_greeting_thread.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
