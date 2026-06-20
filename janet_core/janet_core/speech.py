import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

from . import config
from .utils import clamp_float, ensure_dir


class SpeechManager:
    def __init__(self, state):
        self.state = state
        self.config_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.eleven_lock = threading.Lock()
        self.cfg = {
            "device": config.SPEECH_APLAY_DEVICE,
            "rate": config.SPEECH_APLAY_RATE,
            "channels": config.SPEECH_APLAY_CHANNELS,
        }
        self.eleven_cfg = {
            "engine": getattr(config, "NATURAL_TTS_ENGINE", "edge"),
            "api_key": "",
            "enabled": bool(getattr(config, "ELEVENLABS_ENABLED", True)),
            "bilingual": bool(getattr(config, "ELEVENLABS_BILINGUAL_ENABLED", True)),
            "translate_with_hermes": bool(getattr(config, "ELEVENLABS_TRANSLATE_WITH_HERMES", True)),
            "english_voice_id": getattr(config, "ELEVENLABS_ENGLISH_VOICE_ID", "RlSVB64yXMZJjq67jbB1"),
            "chinese_voice_id": getattr(config, "ELEVENLABS_CHINESE_VOICE_ID", "APSIkVZudNbPAwyPoeVO"),
            "model_id": getattr(config, "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            "output_format": getattr(config, "ELEVENLABS_OUTPUT_FORMAT", "pcm_16000"),
            "cache_enabled": bool(getattr(config, "ELEVENLABS_KEEP_AUDIO_CACHE", True)),
            "edge_enabled": bool(getattr(config, "EDGE_TTS_ENABLED", True)),
            "edge_english_voice": getattr(config, "EDGE_TTS_ENGLISH_VOICE", "en-GB-RyanNeural"),
            "edge_chinese_voice": getattr(config, "EDGE_TTS_CHINESE_VOICE", "zh-CN-XiaoxiaoNeural"),
        }
        self.translation_provider = None
        self.session = requests.Session()
        self.session.trust_env = False
        self.load_elevenlabs_settings()

    def start(self):
        ensure_dir(config.SPEECH_SAMPLE_DIR)
        ensure_dir(config.ELEVENLABS_CACHE_DIR)
        ensure_dir(getattr(config, "EDGE_TTS_CACHE_DIR", config.DATA_DIR / "edge_tts_cache"))
        self.update(
            status="ready",
            speaker_device=self.cfg["device"],
            tts_engine=self.detect_output_engine(),
        )
        self.update_elevenlabs_status(status="ready", last_error="")
        if config.SPEECH_ENABLED:
            self.set_volume(config.SPEECH_VOLUME_PERCENT, quiet=True)

    # -----------------------
    # Status / config helpers
    # -----------------------
    def update(self, **values):
        current = self.state.section("speech")
        current.update(values)
        current["speaker_device"] = self.get_config()["device"]
        current["available"] = True
        current["enabled"] = config.SPEECH_ENABLED
        current["espeak_available"] = bool(self.detect_tts_engine())
        current["tts_engine"] = values.get("tts_engine", current.get("tts_engine") or self.detect_output_engine())
        self.state.update("speech", **current)

    def update_elevenlabs_status(self, **values):
        current = self.state.section("elevenlabs")
        current.update(values)
        info = self.elevenlabs_info(include_key=False)
        current.update({k: v for k, v in info.items() if k != "api_key_preview"})
        self.state.update("elevenlabs", **current)

    def get_config(self):
        with self.config_lock:
            return dict(self.cfg)

    def set_config(self, device=None, rate=None, channels=None):
        with self.config_lock:
            if device:
                self.cfg["device"] = str(device).strip()
            if rate:
                self.cfg["rate"] = str(rate).strip()
            if channels:
                self.cfg["channels"] = str(channels).strip()
        self.update(status="ready", last_message=f"Speaker set to {self.cfg['device']}")
        return self.get_config()

    def load_elevenlabs_settings(self):
        path = getattr(config, "ELEVENLABS_SETTINGS_FILE", config.DATA_DIR / "elevenlabs_settings.json")
        try:
            if Path(path).exists():
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                for key in self.eleven_cfg:
                    if key in data:
                        if isinstance(self.eleven_cfg[key], bool):
                            self.eleven_cfg[key] = bool(data[key])
                        else:
                            self.eleven_cfg[key] = str(data[key])
        except Exception as e:
            print(f"ElevenLabs settings load failed: {e}")

    def save_elevenlabs_settings(self):
        path = Path(getattr(config, "ELEVENLABS_SETTINGS_FILE", config.DATA_DIR / "elevenlabs_settings.json"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(self.eleven_cfg)
            data["updated_at"] = time.time()
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True, str(path)
        except Exception as e:
            return False, str(e)

    def set_elevenlabs_config(self, **values):
        with self.eleven_lock:
            clear_api_key = bool(values.get("clear_api_key"))
            for key, value in values.items():
                if key == "clear_api_key":
                    continue
                if key not in self.eleven_cfg or value is None:
                    continue
                if key == "api_key":
                    cleaned = str(value or "").strip()
                    # Empty API key field means: keep the existing saved key.
                    # Use clear_api_key=true to remove it.
                    if clear_api_key:
                        self.eleven_cfg["api_key"] = ""
                    elif cleaned:
                        self.eleven_cfg["api_key"] = cleaned
                    continue
                if isinstance(self.eleven_cfg[key], bool):
                    if isinstance(value, str):
                        self.eleven_cfg[key] = value.lower() in {"1", "true", "yes", "on"}
                    else:
                        self.eleven_cfg[key] = bool(value)
                else:
                    self.eleven_cfg[key] = str(value).strip()
        ok, msg = self.save_elevenlabs_settings()
        self.update_elevenlabs_status(status="settings saved" if ok else "settings save failed", last_error="" if ok else msg)
        return self.elevenlabs_info(include_key=False)

    def set_translation_provider(self, provider):
        self.translation_provider = provider

    # -----------------------
    # Speaker / ALSA helpers
    # -----------------------
    def detect_tts_engine(self):
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def detect_output_engine(self):
        engine = str(self.eleven_cfg.get("engine") or "edge").lower()
        if engine in {"edge", "auto"} and self.eleven_cfg.get("edge_enabled") and self.edge_tts_available()[0]:
            return "edge-tts"
        if engine in {"elevenlabs", "auto"} and self.eleven_cfg.get("enabled") and self.get_elevenlabs_api_key()[0]:
            return "elevenlabs"
        return os.path.basename(self.detect_tts_engine() or "none")

    def list_speakers(self):
        parts = []
        for args, title in [(["aplay", "-l"], "aplay -l"), (["aplay", "-L"], "aplay -L")]:
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=5)
                parts.append(f"===== {title} =====\n{(r.stdout or r.stderr or '').strip() or '(no output)'}")
            except Exception as e:
                parts.append(f"===== {title} =====\nERROR: {e}")
        text = "\n\n".join(parts)
        self.update(aplay_devices=text)
        return text

    def card_for_device(self, device=None):
        device = str(device or self.get_config()["device"])
        m = re.search(r"CARD=([^,\s]+)", device)
        if m:
            return m.group(1)
        m = re.search(r"(?:plughw|hw):(\d+)(?:,|$)", device)
        if m:
            return m.group(1)
        if "Y02" in device:
            return config.SPEECH_WORKING_USB_CARD
        return ""

    def set_volume(self, percent=None, device=None, quiet=False):
        percent = int(clamp_float(percent, 0, 100, config.SPEECH_VOLUME_PERCENT))
        card = self.card_for_device(device)
        if not card or not shutil.which("amixer"):
            msg = "amixer/card unavailable"
            if not quiet:
                self.update(status="volume error", volume_status=msg, last_error=msg)
            return False, msg
        try:
            r = subprocess.run(["amixer", "-c", str(card), "scontrols"], capture_output=True, text=True, timeout=5)
            controls = re.findall(r"Simple mixer control '([^']+)'", r.stdout or "")
            preferred = [c for c in config.SPEECH_VOLUME_CONTROLS if c in controls] or controls[:3]
            changed = []
            for ctl in preferred:
                res = subprocess.run(["amixer", "-c", str(card), "sset", ctl, f"{percent}%", "unmute"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    changed.append(ctl)
            if changed:
                msg = f"Volume set to {percent}% on card {card}: {', '.join(changed)}"
                self.update(volume_status=msg, volume_card=str(card), volume_percent=percent)
                return True, msg
            msg = f"No volume controls changed on card {card}"
            if not quiet:
                self.update(status="volume error", volume_status=msg, last_error=msg)
            return False, msg
        except Exception as e:
            msg = f"Volume helper error: {e}"
            if not quiet:
                self.update(status="volume error", volume_status=msg, last_error=msg)
            return False, msg

    def generate_tone(self, path, duration=1.0, frequency=660.0, volume=0.45):
        ensure_dir(os.path.dirname(path))
        rate = int(self.get_config().get("rate", config.SPEECH_APLAY_RATE))
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t)
        fade_len = max(1, int(rate * 0.04))
        fade = np.ones_like(tone)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        samples = np.clip(tone * fade * volume * 32767, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(samples.tobytes())

    def play_wav(self, path):
        cfg = self.get_config()
        self.set_volume(config.SPEECH_VOLUME_PERCENT, quiet=True)
        with self.playback_lock:
            r = subprocess.run(["aplay", "-D", cfg["device"], str(path)], capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout or "", r.stderr or ""

    def test_beep(self):
        threading.Thread(target=self._test_beep_worker, daemon=True).start()
        return True, "Speaker beep queued"

    def _test_beep_worker(self):
        path = config.SPEECH_SAMPLE_DIR / "janet_speaker_test.wav"
        self.update(status="playing beep", last_message=f"Playing beep via {self.get_config()['device']}", last_error="")
        try:
            self.generate_tone(path)
            rc, out, err = self.play_wav(path)
            if rc == 0:
                self.update(status="ready", last_message="Speaker test beep played", last_file=str(path), last_error="")
            else:
                self.update(status="speaker test failed", last_error=(err or out).strip(), last_file=str(path))
        except Exception as e:
            self.update(status="speaker test error", last_error=str(e))


    # -----------------------
    # Edge TTS natural fallback helpers
    # -----------------------
    def edge_tts_available(self):
        """Return (available, detail) for the edge-tts Python module/CLI and WAV converter."""
        try:
            import edge_tts  # noqa: F401
            module_ok = True
        except Exception as e:
            module_ok = False
            module_err = str(e)
        else:
            module_err = "python module available"
        converter = shutil.which("ffmpeg") or shutil.which("mpg123")
        if module_ok and converter:
            return True, f"edge-tts available, converter: {os.path.basename(converter)}"
        if not module_ok:
            return False, f"edge-tts not installed: {module_err}. Install with: pip install edge-tts"
        return False, "edge-tts is installed but ffmpeg/mpg123 is missing. Install with: sudo apt install -y ffmpeg"

    def edge_cache_path_for(self, text, language, voice):
        text = self.clean_text(text)
        digest = hashlib.sha1(f"edge|{language}|{voice}|{text}".encode("utf-8")).hexdigest()[:16]
        filename = f"edge_{language}_{self._slug(text)}_{digest}.wav"
        return Path(getattr(config, "EDGE_TTS_CACHE_DIR", config.DATA_DIR / "edge_tts_cache")) / language / filename

    def edge_mp3_path_for(self, wav_path):
        return Path(str(wav_path)[:-4] + ".mp3") if str(wav_path).lower().endswith(".wav") else Path(str(wav_path) + ".mp3")

    def convert_mp3_to_wav(self, mp3_path, wav_path):
        mp3_path, wav_path = Path(mp3_path), Path(wav_path)
        ensure_dir(wav_path.parent)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3_path), "-ar", str(getattr(config, "EDGE_TTS_SAMPLE_RATE", 16000)), "-ac", "1", str(wav_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=float(getattr(config, "EDGE_TTS_TIMEOUT", 45.0)))
            if r.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                return True, "converted with ffmpeg"
            return False, (r.stderr or r.stdout or "ffmpeg conversion failed").strip()
        mpg123 = shutil.which("mpg123")
        if mpg123:
            r = subprocess.run([mpg123, "-w", str(wav_path), str(mp3_path)], capture_output=True, text=True, timeout=float(getattr(config, "EDGE_TTS_TIMEOUT", 45.0)))
            if r.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                return True, "converted with mpg123"
            return False, (r.stderr or r.stdout or "mpg123 conversion failed").strip()
        return False, "No MP3 converter found. Install one: sudo apt install -y ffmpeg"

    def edge_tts_request_to_wav(self, text, voice, wav_path, language="en"):
        available, detail = self.edge_tts_available()
        if not available:
            return False, detail
        text = self.clean_text(text)
        if not text:
            return False, "No text to speak"
        wav_path = Path(wav_path)
        mp3_path = self.edge_mp3_path_for(wav_path)
        ensure_dir(mp3_path.parent)
        try:
            # Use python -m edge_tts so it works inside JanetEnv even if the edge-tts
            # script itself is not on PATH.
            cmd = [sys.executable, "-m", "edge_tts", "--voice", str(voice), "--text", text, "--write-media", str(mp3_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=float(getattr(config, "EDGE_TTS_TIMEOUT", 45.0)))
            if r.returncode != 0:
                return False, (r.stderr or r.stdout or "edge-tts failed").strip()
            if not mp3_path.exists() or mp3_path.stat().st_size < 100:
                return False, "edge-tts did not create an MP3 file"
            return self.convert_mp3_to_wav(mp3_path, wav_path)
        except Exception as e:
            return False, f"edge-tts request failed: {e}"

    def get_or_create_edge_audio(self, text, language="en", voice=None):
        text = self.clean_text(text)
        voice = voice or (self.eleven_cfg.get("edge_english_voice") if language == "en" else self.eleven_cfg.get("edge_chinese_voice"))
        path = self.edge_cache_path_for(text, language, voice)
        if path.exists() and path.stat().st_size > 44:
            return True, path, "edge cache hit"
        ok, msg = self.edge_tts_request_to_wav(text, voice, path, language=language)
        if ok:
            return True, path, msg
        return False, path, msg

    def speak_edge_bilingual(self, text, chinese_text=None):
        text = self.clean_text(text)
        if not text:
            return False, "No text"
        en_voice = self.eleven_cfg.get("edge_english_voice") or getattr(config, "EDGE_TTS_ENGLISH_VOICE", "en-GB-RyanNeural")
        zh_voice = self.eleven_cfg.get("edge_chinese_voice") or getattr(config, "EDGE_TTS_CHINESE_VOICE", "zh-CN-XiaoxiaoNeural")
        self.update(status="speaking", last_phrase=text, tts_engine="edge-tts", last_error="")
        self.update_elevenlabs_status(status="speaking English with Edge TTS", last_text=text, last_error="")
        ok, en_path, msg = self.get_or_create_edge_audio(text, "en", en_voice)
        if not ok:
            self.update_elevenlabs_status(status="edge English generation failed", last_error=msg)
            return False, msg
        rc, out, err = self.play_wav(en_path)
        if rc != 0:
            msg2 = (err or out or "aplay failed").strip()
            self.update_elevenlabs_status(status="edge English playback failed", last_error=msg2)
            return False, msg2
        last_files = [str(en_path)]
        if self.eleven_cfg.get("bilingual"):
            chinese_text = self.clean_text(chinese_text) if chinese_text else self.translate_to_chinese(text)
            self.update_elevenlabs_status(status="speaking Chinese with Edge TTS", last_chinese_text=chinese_text)
            ok, zh_path, msg = self.get_or_create_edge_audio(chinese_text, "zh", zh_voice)
            if not ok:
                self.update_elevenlabs_status(status="edge Chinese generation failed", last_error=msg)
                return False, msg
            time.sleep(float(getattr(config, "SPEECH_BILINGUAL_PAUSE_SECONDS", 0.25)))
            rc, out, err = self.play_wav(zh_path)
            if rc != 0:
                msg2 = (err or out or "aplay failed").strip()
                self.update_elevenlabs_status(status="edge Chinese playback failed", last_error=msg2)
                return False, msg2
            last_files.append(str(zh_path))
        self.update(status="ready", last_message=f"Spoke with Edge TTS: {text}", last_file=" | ".join(last_files), tts_engine="edge-tts", last_error="")
        self.update_elevenlabs_status(status="ready", last_error="", last_files=last_files, edge_cache_count=self.edge_cache_count())
        return True, "Spoke with Edge TTS"

    def edge_cache_count(self):
        try:
            return len(list(Path(getattr(config, "EDGE_TTS_CACHE_DIR", config.DATA_DIR / "edge_tts_cache")).rglob("*.wav")))
        except Exception:
            return 0

    # -----------------------
    # ElevenLabs helpers
    # -----------------------
    def get_elevenlabs_api_key(self):
        # V14.7: no more automatic ElevenLabs.txt/API-ElevenLabs.txt pickup.
        # Use the webpage field under Skills -> Natural TTS, or an env var if preferred.
        key = str(self.eleven_cfg.get("api_key") or "").strip()
        if key:
            return key, "web settings"
        env_key = os.environ.get(getattr(config, "ELEVENLABS_API_KEY_ENV", "ELEVENLABS_API_KEY"), "").strip()
        if env_key:
            return env_key, "environment"
        return "", "not set"

    def _slug(self, text, max_len=34):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_")[:max_len]
        return slug or "speech"

    def cache_path_for(self, text, language, voice_id):
        text = str(text or "").strip()
        digest = hashlib.sha1(f"{language}|{voice_id}|{text}".encode("utf-8")).hexdigest()[:16]
        filename = f"{language}_{self._slug(text)}_{digest}.wav"
        return Path(config.ELEVENLABS_CACHE_DIR) / language / filename

    def write_pcm_wav(self, path, pcm_bytes, rate=None):
        rate = int(rate or getattr(config, "ELEVENLABS_SAMPLE_RATE", 16000))
        ensure_dir(Path(path).parent)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm_bytes)

    def elevenlabs_request_to_wav(self, text, voice_id, output_path, language="en"):
        key, key_source = self.get_elevenlabs_api_key()
        if not key:
            return False, "ElevenLabs API key not set. Enter it in Skills -> Natural TTS, or use Edge TTS without ElevenLabs."
        text = self.clean_text(text)
        if not text:
            return False, "No text to speak"
        output_format = self.eleven_cfg.get("output_format") or getattr(config, "ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
        url = f"{getattr(config, 'ELEVENLABS_API_URL', 'https://api.elevenlabs.io/v1/text-to-speech').rstrip('/')}/{voice_id}"
        params = {"output_format": output_format}
        headers = {
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }
        payload = {
            "text": text,
            "model_id": self.eleven_cfg.get("model_id") or getattr(config, "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            "voice_settings": {
                "stability": float(getattr(config, "ELEVENLABS_STABILITY", 0.55)),
                "similarity_boost": float(getattr(config, "ELEVENLABS_SIMILARITY_BOOST", 0.75)),
                "style": float(getattr(config, "ELEVENLABS_STYLE", 0.15)),
                "use_speaker_boost": bool(getattr(config, "ELEVENLABS_USE_SPEAKER_BOOST", True)),
            },
        }
        try:
            r = self.session.post(url, params=params, headers=headers, json=payload, timeout=float(getattr(config, "ELEVENLABS_TIMEOUT", 30.0)))
            if r.status_code >= 300:
                return False, f"ElevenLabs HTTP {r.status_code}: {r.text[:500]}"
            if output_format.startswith("pcm_"):
                self.write_pcm_wav(output_path, r.content, rate=getattr(config, "ELEVENLABS_SAMPLE_RATE", 16000))
            else:
                # Best effort: save whatever was returned. aplay can only play WAV/PCM,
                # so use pcm_16000 for Janet unless you add an mp3 player.
                Path(output_path).write_bytes(r.content)
            return True, f"Downloaded ElevenLabs {language} audio using key from {key_source}"
        except Exception as e:
            return False, f"ElevenLabs request failed: {e}"

    def clean_text(self, text):
        text = re.sub(r"[\r\n\t]+", " ", str(text or config.SPEECH_TEST_PHRASE)).strip()
        text = re.sub(r"\s+", " ", text)
        return text[: int(getattr(config, "ELEVENLABS_MAX_TEXT_CHARS", 900))]

    def get_or_create_elevenlabs_audio(self, text, language="en", voice_id=None):
        text = self.clean_text(text)
        voice_id = voice_id or (self.eleven_cfg.get("english_voice_id") if language == "en" else self.eleven_cfg.get("chinese_voice_id"))
        path = self.cache_path_for(text, language, voice_id)
        if path.exists() and path.stat().st_size > 44:
            return True, path, "cache hit"
        ok, msg = self.elevenlabs_request_to_wav(text, voice_id, path, language=language)
        if ok:
            return True, path, msg
        return False, path, msg

    def local_chinese_translation(self, text):
        text = self.clean_text(text)
        exact = {
            "Hello, I am Janet.": "你好，我是 Janet。",
            "Hello, I am Janet. My speaker is working.": "你好，我是 Janet。我的扬声器正常工作。",
            "Hello, I am Bender-Janet. I have just woken up, and my speaker is working.": "你好，我是 Bender-Janet。我刚刚醒来，我的扬声器正在工作。",
            "I am listening.": "我在听。",
            "I am ready.": "我准备好了。",
            "Motors stopped.": "电机已停止。",
        }
        if text in exact:
            return exact[text]
        obj = {
            "tv": "电视", "television": "电视", "potted plant": "盆栽", "plant": "植物",
            "chair": "椅子", "couch": "沙发", "sofa": "沙发", "bottle": "瓶子",
            "cup": "杯子", "laptop": "笔记本电脑", "keyboard": "键盘", "clock": "时钟",
            "vase": "花瓶", "car": "汽车", "cat": "猫", "dog": "狗", "bed": "床",
            "refrigerator": "冰箱", "umbrella": "雨伞", "bench": "长椅", "teddy bear": "泰迪熊",
            "traffic light": "交通灯", "book": "书", "mouse": "鼠标", "remote": "遥控器",
        }
        m = re.match(r"^Hi\s+(.+)$", text, re.I)
        if m:
            target = m.group(1).strip()
            return "你好，" + obj.get(target.lower(), target) + "。"
        m = re.match(r"^I can see an?\s+(.+?)\.?$", text, re.I)
        if m:
            target = m.group(1).strip()
            return "我看见了" + obj.get(target.lower(), target) + "。"
        return ""

    def translate_to_chinese(self, text):
        text = self.clean_text(text)
        local = self.local_chinese_translation(text)
        if local:
            return local
        if self.eleven_cfg.get("translate_with_hermes") and self.translation_provider:
            try:
                translated = self.translation_provider(text)
                translated = str(translated or "").strip()
                if translated:
                    return translated[: int(getattr(config, "ELEVENLABS_MAX_TEXT_CHARS", 900))]
            except Exception as e:
                self.update_elevenlabs_status(last_error=f"Hermes Chinese translation failed: {e}")
        # Last fallback: Sage repeats the English text. This is not ideal, but it
        # avoids blocking speech when Hermes is offline.
        return text

    def speak_elevenlabs_bilingual(self, text, chinese_text=None):
        text = self.clean_text(text)
        if not text:
            return False, "No text"
        english_voice = self.eleven_cfg.get("english_voice_id") or config.ELEVENLABS_ENGLISH_VOICE_ID
        chinese_voice = self.eleven_cfg.get("chinese_voice_id") or config.ELEVENLABS_CHINESE_VOICE_ID
        self.update(status="speaking", last_phrase=text, tts_engine="elevenlabs", last_error="")
        self.update_elevenlabs_status(status="speaking English", last_text=text, last_error="")

        ok, en_path, msg = self.get_or_create_elevenlabs_audio(text, "en", english_voice)
        if not ok:
            self.update_elevenlabs_status(status="english generation failed", last_error=msg)
            return False, msg
        rc, out, err = self.play_wav(en_path)
        if rc != 0:
            msg2 = (err or out or "aplay failed").strip()
            self.update_elevenlabs_status(status="english playback failed", last_error=msg2)
            return False, msg2

        last_files = [str(en_path)]
        if self.eleven_cfg.get("bilingual"):
            chinese_text = self.clean_text(chinese_text) if chinese_text else self.translate_to_chinese(text)
            self.update_elevenlabs_status(status="speaking Chinese", last_chinese_text=chinese_text)
            ok, zh_path, msg = self.get_or_create_elevenlabs_audio(chinese_text, "zh", chinese_voice)
            if not ok:
                self.update_elevenlabs_status(status="chinese generation failed", last_error=msg)
                return False, msg
            time.sleep(float(getattr(config, "SPEECH_BILINGUAL_PAUSE_SECONDS", 0.25)))
            rc, out, err = self.play_wav(zh_path)
            if rc != 0:
                msg2 = (err or out or "aplay failed").strip()
                self.update_elevenlabs_status(status="chinese playback failed", last_error=msg2)
                return False, msg2
            last_files.append(str(zh_path))

        self.update(status="ready", last_message=f"Spoke with ElevenLabs: {text}", last_file=" | ".join(last_files), tts_engine="elevenlabs", last_error="")
        self.update_elevenlabs_status(status="ready", last_error="", last_files=last_files, cache_count=self.cache_count())
        return True, "Spoke with ElevenLabs"

    def cache_count(self):
        try:
            return len(list(Path(config.ELEVENLABS_CACHE_DIR).rglob("*.wav")))
        except Exception:
            return 0

    def elevenlabs_info(self, include_key=False):
        key, source = self.get_elevenlabs_api_key()
        edge_available, edge_detail = self.edge_tts_available()
        info = {
            "engine": self.eleven_cfg.get("engine") or "edge",
            "enabled": bool(self.eleven_cfg.get("enabled")),
            "bilingual": bool(self.eleven_cfg.get("bilingual")),
            "translate_with_hermes": bool(self.eleven_cfg.get("translate_with_hermes")),
            "api_key_found": bool(key),
            "api_key_saved": bool(str(self.eleven_cfg.get("api_key") or "").strip()),
            "api_key_source": source if key else "not set",
            "api_key_preview": (key[:6] + "..." + key[-4:]) if (include_key and key and len(key) > 12) else "",
            "english_voice_id": self.eleven_cfg.get("english_voice_id"),
            "english_voice_name": getattr(config, "ELEVENLABS_ENGLISH_VOICE_NAME", "Bren"),
            "chinese_voice_id": self.eleven_cfg.get("chinese_voice_id"),
            "chinese_voice_name": getattr(config, "ELEVENLABS_CHINESE_VOICE_NAME", "Sage"),
            "model_id": self.eleven_cfg.get("model_id"),
            "output_format": self.eleven_cfg.get("output_format"),
            "cache_dir": str(config.ELEVENLABS_CACHE_DIR),
            "cache_count": self.cache_count(),
            "settings_file": str(getattr(config, "ELEVENLABS_SETTINGS_FILE", "")),
            "edge_enabled": bool(self.eleven_cfg.get("edge_enabled")),
            "edge_available": bool(edge_available),
            "edge_detail": edge_detail,
            "edge_english_voice": self.eleven_cfg.get("edge_english_voice"),
            "edge_chinese_voice": self.eleven_cfg.get("edge_chinese_voice"),
            "edge_english_voice_name": getattr(config, "EDGE_TTS_ENGLISH_VOICE_NAME", "Edge English"),
            "edge_chinese_voice_name": getattr(config, "EDGE_TTS_CHINESE_VOICE_NAME", "Edge Chinese"),
            "edge_cache_dir": str(getattr(config, "EDGE_TTS_CACHE_DIR", config.DATA_DIR / "edge_tts_cache")),
            "edge_cache_count": self.edge_cache_count(),
        }
        return info

    def precache_vocabulary(self, names=None, objects=None, extra_phrases=None, limit=80):
        phrases = list(getattr(config, "ELEVENLABS_BASIC_VOCABULARY", ()))
        for name in names or []:
            name = str(name).strip()
            if name:
                phrases.append(f"Hi {name}")
        for label in objects or []:
            label = str(label).strip()
            if label:
                phrases.append(f"Hi {label}")
                phrases.append(f"I can see a {label}.")
        for phrase in extra_phrases or []:
            if str(phrase).strip():
                phrases.append(str(phrase).strip())
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for p in phrases:
            key = p.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        unique = unique[: int(limit)]
        done, errors = [], []
        self.update_elevenlabs_status(status="pre-caching", last_error="")
        for phrase in unique:
            ok, msg = self.speak_natural_cache_only(phrase)
            if ok:
                done.append(phrase)
            else:
                errors.append(f"{phrase}: {msg}")
                if len(errors) >= 4:
                    break
        status = "pre-cache complete" if not errors else "pre-cache partial"
        self.update_elevenlabs_status(status=status, cache_count=self.cache_count(), last_error=" | ".join(errors[:4]))
        return {"ok": not bool(errors), "cached": done, "errors": errors, "cache_count": self.cache_count()}

    def speak_natural_cache_only(self, text):
        engine = str(self.eleven_cfg.get("engine") or "edge").lower()
        # Edge TTS is Janet's default natural cache for English + Chinese.
        if engine in {"edge", "auto"} and self.eleven_cfg.get("edge_enabled"):
            ok, path, msg = self.get_or_create_edge_audio(text, "en", self.eleven_cfg.get("edge_english_voice"))
            if ok and self.eleven_cfg.get("bilingual"):
                zh = self.translate_to_chinese(text)
                ok2, path2, msg2 = self.get_or_create_edge_audio(zh, "zh", self.eleven_cfg.get("edge_chinese_voice"))
                if not ok2:
                    return False, msg2
            if ok:
                return True, "cached with edge-tts"
            if engine == "edge":
                return False, msg
        if self.eleven_cfg.get("enabled") and self.get_elevenlabs_api_key()[0]:
            return self.speak_elevenlabs_cache_only(text)
        return False, "No natural TTS cache engine available"

    def speak_elevenlabs_cache_only(self, text):
        text = self.clean_text(text)
        ok, path, msg = self.get_or_create_elevenlabs_audio(text, "en", self.eleven_cfg.get("english_voice_id"))
        if not ok:
            return False, msg
        if self.eleven_cfg.get("bilingual"):
            zh = self.translate_to_chinese(text)
            ok2, path2, msg2 = self.get_or_create_elevenlabs_audio(zh, "zh", self.eleven_cfg.get("chinese_voice_id"))
            if not ok2:
                return False, msg2
        return True, "cached"

    def test_elevenlabs(self, text=None):
        threading.Thread(target=self.say, args=(text or config.SPEECH_TEST_PHRASE, False), daemon=True).start()
        return True, "Natural TTS test speech queued"

    # -----------------------
    # Public speech entrypoint
    # -----------------------
    def say_async(self, text, allow_beep_fallback=True, chinese_text=None):
        threading.Thread(target=self.say, args=(text, allow_beep_fallback, chinese_text), daemon=True).start()

    def say_bilingual_async(self, english_text, chinese_text, allow_beep_fallback=True):
        return self.say_async(english_text, allow_beep_fallback=allow_beep_fallback, chinese_text=chinese_text)

    def say_bilingual(self, english_text, chinese_text, allow_beep_fallback=True):
        return self.say(english_text, allow_beep_fallback=allow_beep_fallback, chinese_text=chinese_text)

    def say(self, text, allow_beep_fallback=True, chinese_text=None):
        text = self.clean_text(text)
        engine_pref = str(self.eleven_cfg.get("engine") or "edge").lower()

        # 1) Natural online voice: Edge TTS, cached locally.
        # This is Janet's default because it gives natural English + Chinese without ElevenLabs.
        if engine_pref in {"auto", "edge"} and self.eleven_cfg.get("edge_enabled"):
            ok, msg = self.speak_edge_bilingual(text, chinese_text=chinese_text)
            if ok:
                return
            self.update(status="edge-tts failed; trying fallback", last_error=msg, tts_engine="edge-tts")
            if engine_pref == "edge":
                return

        # 2) Optional ElevenLabs only if selected/allowed and a web-entered key exists.
        if engine_pref in {"auto", "elevenlabs"} and self.eleven_cfg.get("enabled") and self.get_elevenlabs_api_key()[0]:
            ok, msg = self.speak_elevenlabs_bilingual(text, chinese_text=chinese_text)
            if ok:
                return
            self.update(status="elevenlabs failed; falling back", last_error=msg, tts_engine="elevenlabs")
            if engine_pref == "elevenlabs":
                return

        # 3) Local espeak-ng fallback.
        path = config.SPEECH_SAMPLE_DIR / "janet_speech.wav"
        self.update(status="speaking", last_phrase=text, last_error="")
        engine = self.detect_tts_engine()
        try:
            if engine:
                args = [engine]
                if os.path.basename(engine) in {"espeak-ng", "espeak"}:
                    args += ["-v", config.SPEECH_TTS_VOICE, "-s", config.SPEECH_TTS_SPEED, "-a", config.SPEECH_TTS_AMPLITUDE]
                args += ["-w", str(path), text]
                gen = subprocess.run(args, capture_output=True, text=True, timeout=20)
                if gen.returncode != 0:
                    self.update(status="speech generation failed", last_error=(gen.stderr or gen.stdout).strip(), tts_engine=os.path.basename(engine))
                    return
                rc, out, err = self.play_wav(path)
                if rc == 0:
                    self.update(status="ready", last_message=f"Spoke: {text}", last_file=str(path), tts_engine=os.path.basename(engine), last_error="")
                else:
                    self.update(status="speech playback failed", last_error=(err or out).strip(), last_file=str(path), tts_engine=os.path.basename(engine))
                return
            if allow_beep_fallback:
                self.generate_tone(path, frequency=880.0)
                rc, out, err = self.play_wav(path)
                if rc == 0:
                    self.update(status="ready", last_message="No natural/local TTS engine, beep fallback played", tts_engine="beep fallback")
                else:
                    self.update(status="speech fallback failed", last_error=(err or out).strip())
            else:
                self.update(status="no speech engine", last_error="No natural TTS, Edge TTS, or espeak-ng/espeak found; beep fallback disabled")
        except Exception as e:
            self.update(status="speech error", last_error=str(e))
