import math
import os
import re
import wave
from pathlib import Path
import numpy as np


def clamp_float(value, minimum, maximum, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = fallback
    return max(minimum, min(value, maximum))


def safe_name(value, fallback="item"):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("._")
    return value or fallback


def pretty_label(label):
    label = str(label or "object").replace("_", " ").replace("-", " ").strip()
    replacements = {"tvmonitor": "TV monitor", "pottedplant": "potted plant", "diningtable": "dining table", "cell phone": "phone", "mobile phone": "phone"}
    label = replacements.get(label.lower().replace(" ", ""), replacements.get(label.lower(), label))
    label = re.sub(r"[^A-Za-z0-9 ]+", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label or "object"


def normalise_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def format_cm(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if value < 0:
        return "unavailable"
    return f"{value:.2f} cm"


def wav_audio_stats(path):
    try:
        with wave.open(str(path), "rb") as wf:
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
        level = int(max(0, min(100, ((db + 60.0) / 60.0) * 100)))
        return {"rms": round(rms, 1), "peak": round(peak, 1), "db": round(db, 1), "level": level}
    except Exception as e:
        return {"rms": 0, "peak": 0, "db": -120.0, "level": 0, "error": str(e)}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
