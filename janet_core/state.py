import threading
import time
from . import config


def default_readings():
    return {
        "fps": 0.0,
        "detections": [],
        "sonar": {"front": -1, "back": -1, "left": -1, "right": -1},
        "sonar_available": False,
        "status": "starting",
        "detection": {"ai_enabled": True, "message": "AI detection enabled"},
        "oak_d": {
            "current_example": "bender-janet",
            "selected_examples": [],
            "selected_modes": [],
            "mode": "normal",
            "message": "Normal operating mode",
            "model": config.DETECTION_MODEL_NAME,
        },
        "motors_available": False,
        "motor_settings": {},
        "detection_model": {"name": config.DETECTION_MODEL_NAME, "confidence": config.DETECTION_CONFIDENCE, "labels_count": 0, "labels_preview": [], "scan_results": [], "last_scan_message": "Not scanned yet"},
        "face": {"available": False, "enabled": True, "status": "starting", "known_faces": [], "known_count": 0, "last_seen": [], "threshold": config.FACE_RECOGNITION_THRESHOLD, "last_message": "Face system starting", "last_greeting": "", "last_greeting_person": "", "greeting_cooldown_seconds": config.FACE_GREETING_COOLDOWN_SECONDS},
        "object": {"available": True, "enabled": True, "status": "starting", "known_objects": [], "known_count": 0, "samples_count": 0, "last_seen": [], "last_message": "Object memory starting", "acknowledge_cooldown_seconds": config.OBJECT_GREETING_COOLDOWN_SECONDS},
        "voice": {"available": False, "enabled": config.VOICE_ENABLED, "status": "starting", "last_heard": "", "last_action": "", "last_error": "", "mic_level": 0, "mic_level_db": None, "wake_word_active": False, "sound_active": False, "last_sound_message": "", "voice_device": config.VOICE_ARECORD_DEVICE},
        "speech": {"available": True, "enabled": config.SPEECH_ENABLED, "status": "starting", "speaker_device": config.SPEECH_APLAY_DEVICE, "last_message": "Speech starting", "last_error": "", "last_phrase": "", "tts_engine": "", "volume_status": ""},
        "elevenlabs": {"enabled": config.ELEVENLABS_ENABLED, "bilingual": config.ELEVENLABS_BILINGUAL_ENABLED, "status": "starting", "api_key_found": False, "api_key_source": "", "api_key_saved": False, "engine": config.NATURAL_TTS_ENGINE, "english_voice_name": config.ELEVENLABS_ENGLISH_VOICE_NAME, "chinese_voice_name": config.ELEVENLABS_CHINESE_VOICE_NAME, "cache_count": 0, "cache_dir": str(config.ELEVENLABS_CACHE_DIR), "edge_enabled": config.EDGE_TTS_ENABLED, "edge_available": False, "edge_cache_count": 0, "edge_cache_dir": str(config.EDGE_TTS_CACHE_DIR), "edge_english_voice": config.EDGE_TTS_ENGLISH_VOICE, "edge_chinese_voice": config.EDGE_TTS_CHINESE_VOICE, "last_error": "", "last_text": "", "last_chinese_text": ""},
        "routines": {"available": True, "active": False, "current_id": "", "current_name": "", "status": "ready", "last_message": "Motor routines ready", "elapsed_seconds": 0.0, "target_seconds": config.MOTOR_DANCE_DURATION_SECONDS, "steps_done": 0, "routines": []},
        "hermes": {"base_url": config.HERMES_BASE_URL, "endpoint": config.HERMES_ENDPOINT, "model": config.HERMES_MODEL, "status": "ready", "last_error": "", "last_answer": "", "api_key_set": bool(config.HERMES_API_KEY)},
        "boot": {
            "enabled": config.BOOT_ROUTINE_ENABLED,
            "active": bool(config.BOOT_ROUTINE_ENABLED),
            "complete": False,
            "step_index": 0,
            "total_steps": config.BOOT_TOTAL_STEPS,
            "step": "starting",
            "last_message": "Boot routine starting",
            "last_error": "",
            "started_at": None,
            "completed_at": None,
            "history": [],
        },
        "front_camera_ready": False,
        "front_camera_error": "",
        "front_frame_age_seconds": None,
        "updated_at": time.time(),
    }


class JanetState:
    def __init__(self):
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.latest = default_readings()

    def update(self, section=None, **values):
        with self.lock:
            target = self.latest if section is None else self.latest.setdefault(section, {})
            target.update(values)
            self.latest["updated_at"] = time.time()

    def get(self):
        with self.lock:
            import copy
            return copy.deepcopy(self.latest)

    def section(self, name):
        with self.lock:
            import copy
            return copy.deepcopy(self.latest.get(name, {}))
