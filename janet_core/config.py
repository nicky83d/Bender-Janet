from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# General
APP_NAME = "Janet"
APP_VERSION = "V16"
HOST = "0.0.0.0"
PORT = 5000


# Startup greeting
STARTUP_SPEECH_ENABLED = False
STARTUP_SPEECH_DELAY_SECONDS = 2.0
STARTUP_SPEECH_TEXT_EN = "Hello, I am Bender-Janet. I have just woken up, and my speaker is working."
STARTUP_SPEECH_TEXT_ZH = "你好，我是 Bender-Janet。我刚刚醒来，我的扬声器正在工作。"


# Boot routine
BOOT_ROUTINE_ENABLED = True
BOOT_SPEECH_ENABLED = False
BOOT_TOTAL_STEPS = 6
BOOT_INITIAL_DELAY_SECONDS = 0.6
BOOT_MOVEMENT_DURATION = 0.22
BOOT_MOVEMENT_SETTLE_SECONDS = 0.45
BOOT_SONAR_CHANGE_THRESHOLD_CM = 1.5
BOOT_CAMERA_READY_TIMEOUT_SECONDS = 25.0
BOOT_STEP_PAUSE_SECONDS = 0.0
BOOT_HERMES_DISCOVER_ON_BOOT = True
BOOT_WEATHER_QUESTION = "What is the weather forecast for Bournemouth, UK tomorrow? Reply in one short spoken sentence only."
BOOT_SUPPRESS_RECOGNITION_UNTIL_COMPLETE = True

# Motors / Arduino I2C
MOTOR_I2C_BUS = 1
MOTOR_I2C_ADDR = 0x08
REVERSE_MOTOR_LOGIC = True
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

# Cable-safe routines
MOTOR_DANCE_DURATION_SECONDS = 20.0
MOTOR_DANCE_STEP_GAP = 0.08
MOTOR_DANCE_MAX_CONSECUTIVE_STEPS = 5
MOTOR_DANCE_RETURN_HOME = True
MOTOR_DANCE_ROUTINES = {
    "cable_wiggle": {"name": "Cable Wiggle", "emoji": "〰️", "description": "Small forward/back steps with gentle left/right wiggles.", "pattern": ["forward", "backward", "left", "right", "left", "right", "backward", "forward"]},
    "box_shuffle": {"name": "Box Shuffle", "emoji": "◼️", "description": "Tiny box-shaped shuffle.", "pattern": ["forward", "right", "backward", "left", "forward", "left", "backward", "right"]},
    "happy_bounce": {"name": "Happy Bounce", "emoji": "🤖", "description": "Short forward/back steps and quick turns.", "pattern": ["forward", "backward", "forward", "backward", "left", "right", "left", "right"]},
    "disco_twist": {"name": "Disco Twist", "emoji": "🕺", "description": "More turning than travelling.", "pattern": ["left", "right", "left", "right", "forward", "backward", "right", "left"]},
    "rock_music": {"name": "Rock Music", "emoji": "🎸", "description": "Punchy forward/back hits and left/right snaps.", "pattern": ["forward", "backward", "forward", "backward", "left", "right", "right", "left"]},
    "salsa_dance": {"name": "Salsa Dance", "emoji": "💃", "description": "Side-to-side salsa feel.", "pattern": ["left", "right", "forward", "backward", "right", "left", "forward", "backward"]},
    "waltz": {"name": "Waltz", "emoji": "🎻", "description": "Gentle box-step rhythm.", "pattern": ["forward", "left", "backward", "right", "forward", "right", "backward", "left"]},
    "dance": {"name": "Dance", "emoji": "🎶", "description": "Simple all-round dance mix.", "pattern": ["forward", "backward", "left", "right", "right", "left", "backward", "forward"]},
    "drum_and_bass": {"name": "Drum and Bass", "emoji": "🥁", "description": "Fast twitchy turns and short bassline bounces.", "pattern": ["left", "right", "left", "right", "forward", "backward", "forward", "backward", "right", "left", "backward", "forward"]},
}

# Sonars, BOARD numbering
TRIG_PINS = {"front": 13, "back": 16, "left": 29, "right": 33}
ECHO_PINS = {"front": 15, "back": 18, "left": 31, "right": 35}
SONAR_INTERVAL = 0.2

# Cameras / Vision
FRONT_CAMERA_AUTO_START = True
FRONT_CAMERA_RESTART_DELAY = 2.0
FRONT_CAMERA_PLACEHOLDER_WIDTH = 640
FRONT_CAMERA_PLACEHOLDER_HEIGHT = 480
REAR_CAMERA_ENABLED = False
REAR_CAMERA_INDEXES = [0, 14, 1, 2]
REAR_CAMERA_WIDTH = 640
REAR_CAMERA_HEIGHT = 480
DETECTION_MODEL_NAME = "yolov6-nano"
DETECTION_CONFIDENCE = 0.50
DETECTION_MODEL_CANDIDATES = [
    {"name": "yolov6-nano", "type": "known-good", "note": "Current stable Janet model."},
    {"name": "yolov6-tiny", "type": "candidate", "note": "Try for more accuracy with modest extra load."},
    {"name": "yolov6-s", "type": "candidate", "note": "May reduce FPS on Pi/OAK-D."},
    {"name": "yolov8n", "type": "candidate", "note": "Modern nano detector if supported."},
    {"name": "yolo11n", "type": "candidate", "note": "Newer YOLO-family nano candidate if supported."},
]

# Face recognition
FACE_DIR = DATA_DIR / "known_faces"
FACE_DB_FILE = FACE_DIR / "known_faces.npz"
FACE_IMAGE_DIR = FACE_DIR / "images"
FACE_RECOGNITION_ENABLED = True
FACE_DETECTION_EVERY_N_FRAMES = 4
FACE_RECOGNITION_THRESHOLD = 55.0
FACE_TEMPLATE_SIZE = (80, 80)
FACE_MIN_SIZE = (48, 48)
FACE_GREETING_COOLDOWN_SECONDS = 10 * 60
FACE_GREETING_TEMPLATE = "Hi {name}"

# Object memory
OBJECT_DIR = DATA_DIR / "known_objects"
OBJECT_DB_FILE = OBJECT_DIR / "known_objects.json"
OBJECT_IMAGE_DIR = OBJECT_DIR / "images"
OBJECT_MAX_SAMPLES_PER_LABEL = 12
OBJECT_GREETING_COOLDOWN_SECONDS = 30 * 60
OBJECT_GREETING_TEMPLATE = "Hi {name}"
OBJECT_SKIP_LABELS = {"person"}
OBJECT_MIN_CONFIDENCE = 50

# Centering turns only left/right, no forward movement
TARGET_CENTERING_ENABLED = True
TARGET_CENTERING_DEADBAND = 0.12
TARGET_CENTERING_TURN_DURATION = 0.16
TARGET_CENTERING_COOLDOWN_SECONDS = 1.8

# Voice input
VOICE_ENABLED = True
VOICE_WAKE_WORD = "janet"
VOICE_FORWARD_PHRASES = ("move forward", "forward", "go forward")
VOICE_BACKWARD_PHRASES = ("move back", "move backward", "backward", "go back")
VOICE_LEFT_PHRASES = ("turn left", "left")
VOICE_RIGHT_PHRASES = ("turn right", "right")
VOICE_STOP_PHRASES = ("stop", "halt")
VOICE_QUESTION_PREFIXES = ("what", "what's", "who", "how", "why", "where", "when", "tell me", "explain")
VOICE_PHRASE_TIME_LIMIT = 4
VOICE_ARECORD_DEVICE = "auto"
VOICE_ARECORD_RATE = "16000"
VOICE_ARECORD_CHANNELS = "1"
VOICE_SAMPLE_DIR = DATA_DIR / "voice_samples"
VOICE_TARGET_DB = -18.0
VOICE_MAX_AUTO_GAIN = 12.0
VOICE_RECOGNITION_GAIN = 3.0
VOICE_SOUND_LEVEL_THRESHOLD = 8
VOICE_WAKE_OVERLAY_SECONDS = 2.5
VOICE_SOUND_OVERLAY_SECONDS = 1.5
VOICE_MIN_FRONT_DISTANCE_CM = 20

# Speech output
SPEECH_ENABLED = True
SPEECH_APLAY_DEVICE = "plughw:CARD=Y02,DEV=0"
SPEECH_APLAY_RATE = "16000"
SPEECH_APLAY_CHANNELS = "1"
SPEECH_WORKING_USB_CARD = "Y02"
SPEECH_VOLUME_PERCENT = 100
SPEECH_SAMPLE_DIR = DATA_DIR / "speech_samples"
SPEECH_TEST_PHRASE = "Hello, I am Janet. My speaker is working."
SPEECH_TTS_VOICE = "en-gb"
SPEECH_TTS_SPEED = "145"
SPEECH_TTS_AMPLITUDE = "170"
SPEECH_VOLUME_CONTROLS = ("Master", "Speaker", "PCM", "Headphone", "Headphones", "Playback")
SPEECH_BILINGUAL_PAUSE_SECONDS = 0.0



# Natural TTS / Edge TTS fallback
# ElevenLabs library voices may require a paid plan. Edge TTS gives Janet a
# natural online fallback for English + Chinese and still caches audio locally.
NATURAL_TTS_ENGINE = "edge"  # edge recommended; also supports auto, elevenlabs, local
EDGE_TTS_ENABLED = True
EDGE_TTS_CACHE_DIR = DATA_DIR / "edge_tts_cache"
EDGE_TTS_ENGLISH_VOICE = "en-GB-RyanNeural"
EDGE_TTS_CHINESE_VOICE = "zh-CN-XiaoxiaoNeural"
EDGE_TTS_ENGLISH_VOICE_NAME = "Edge Ryan"
EDGE_TTS_CHINESE_VOICE_NAME = "Edge Xiaoxiao"
EDGE_TTS_TIMEOUT = 45.0
EDGE_TTS_SAMPLE_RATE = 16000
EDGE_TTS_KEEP_AUDIO_CACHE = True
EDGE_TTS_REQUIRE_CONVERTER = True  # ffmpeg or mpg123 is needed to turn mp3 into WAV for aplay

# Optional ElevenLabs Text-to-Speech
# V14.7 uses Edge TTS by default for natural English + Chinese. ElevenLabs is optional only.
# Janet no longer auto-reads API keys from ElevenLabs.txt/API-ElevenLabs.txt. Enter a key
# in Skills -> Natural TTS if you later upgrade/enable ElevenLabs.
ELEVENLABS_ENABLED = False
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_API_KEY_ENV = "ELEVENLABS_API_KEY"
ELEVENLABS_API_KEY_FILES = ()
ELEVENLABS_CACHE_DIR = DATA_DIR / "elevenlabs_cache"
ELEVENLABS_SETTINGS_FILE = DATA_DIR / "elevenlabs_settings.json"
ELEVENLABS_ENGLISH_VOICE_ID = "RlSVB64yXMZJjq67jbB1"  # Bren
ELEVENLABS_CHINESE_VOICE_ID = "APSIkVZudNbPAwyPoeVO"  # Sage
ELEVENLABS_ENGLISH_VOICE_NAME = "Bren"
ELEVENLABS_CHINESE_VOICE_NAME = "Sage"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "pcm_16000"
ELEVENLABS_SAMPLE_RATE = 16000
ELEVENLABS_TIMEOUT = 30.0
ELEVENLABS_BILINGUAL_ENABLED = True
ELEVENLABS_TRANSLATE_WITH_HERMES = True
ELEVENLABS_KEEP_AUDIO_CACHE = True
ELEVENLABS_STABILITY = 0.55
ELEVENLABS_SIMILARITY_BOOST = 0.75
ELEVENLABS_STYLE = 0.15
ELEVENLABS_USE_SPEAKER_BOOST = True
ELEVENLABS_MAX_TEXT_CHARS = 900
ELEVENLABS_BASIC_VOCABULARY = (
    "Hello, I am Bender-Janet. I have just woken up, and my speaker is working.",
    "Hello, I am Janet.",
    "Hello, I am Janet. My speaker is working.",
    "Hi Nico",
    "Hi Paul",
    "I can see a tv.",
    "I can see a potted plant.",
    "I can see a chair.",
    "I can see a couch.",
    "I can see a bottle.",
    "I can see a cup.",
    "I can see a laptop.",
    "I can see a keyboard.",
    "I can see a clock.",
    "I can see a vase.",
    "I can see a car.",
    "I can see a cat.",
    "I can see a bed.",
    "I am listening.",
    "I am ready.",
    "Motors stopped.",
)

# Screenshots
SCREENSHOT_DIR = DATA_DIR / "screenshots"
SCREENSHOT_PREFIX = "janet_front"

# Hermes AI
HERMES_BASE_URL = "http://192.168.50.186:8642"
HERMES_API_KEY = "change-me-local-dev"
HERMES_MODEL = "gemma4:31b-cloud"
HERMES_ENDPOINT = "/v1/chat/completions"
HERMES_TIMEOUT = 12.0

HERMES_DISCOVERY_PORTS = (8642, 8000, 9119, 3000, 8080, 5000, 5173, 11434, 7860, 9000)
HERMES_SETTINGS_FILE = DATA_DIR / "hermes_settings.json"
HERMES_AUTO_DISCOVER_ON_START = True
HERMES_INCLUDE_ROBOT_CONTEXT_BY_DEFAULT = False

# Hermes reliability / self-repair
HERMES_REPAIR_ON_FAILURE = True
HERMES_REPAIR_ON_BOOT = True
HERMES_REPAIR_CHAT_TEST = True
HERMES_REPAIR_RETRIES = 2
