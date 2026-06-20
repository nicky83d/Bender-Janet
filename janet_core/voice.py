import io
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE=True
except ImportError:
    sr=None; SPEECH_RECOGNITION_AVAILABLE=False
import numpy as np
from . import config
from .utils import ensure_dir, wav_audio_stats


class VoiceManager:
    def __init__(self, state, motors=None, hermes=None, speech=None, service_report_fn=None):
        self.state=state; self.motors=motors; self.hermes=hermes; self.speech=speech; self.service_report_fn=service_report_fn
        self.cfg={"device": config.VOICE_ARECORD_DEVICE, "rate": config.VOICE_ARECORD_RATE, "channels": config.VOICE_ARECORD_CHANNELS}
        self.cfg_lock=threading.Lock(); self.command_queue=queue.Queue(maxsize=5)
        self.recording_event=threading.Event(); self.capture_lock=threading.Lock(); self.proc_lock=threading.Lock(); self.live_proc=None
        self.thread=None; self.motor_thread=None; self.wake_until=0; self.sound_until=0

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        ensure_dir(config.VOICE_SAMPLE_DIR)
        self.set_config(device=self.detect_device())
        self.update(available=SPEECH_RECOGNITION_AVAILABLE, status="starting", voice_device=self.get_config()["device"])
        self.motor_thread=threading.Thread(target=self._motor_worker, daemon=True); self.motor_thread.start()
        self.thread=threading.Thread(target=self._listen_loop, daemon=True); self.thread.start()

    def update(self, **values):
        current=self.state.section("voice"); current.update(values)
        current["wake_word_active"] = time.time() < self.wake_until
        current["sound_active"] = time.time() < self.sound_until
        current["voice_device"] = self.get_config()["device"]
        self.state.update("voice", **current)

    def get_config(self):
        with self.cfg_lock:
            return dict(self.cfg)

    def set_config(self, device=None, rate=None, channels=None):
        with self.cfg_lock:
            if device: self.cfg["device"] = device
            if rate: self.cfg["rate"] = str(rate)
            if channels: self.cfg["channels"] = str(channels)
        self.update(voice_device=self.cfg["device"])
        return self.get_config()

    def detect_device(self):
        try:
            r=subprocess.run(["arecord","-L"],capture_output=True,text=True,timeout=5)
            plug=[ln.strip() for ln in (r.stdout or '').splitlines() if ln.strip().startswith('plughw:')]
            for dev in plug:
                if 'CARD=Y02' in dev: return dev
            for dev in plug:
                if 'CARD=Device' in dev: return dev
            if plug: return plug[0]
        except Exception: pass
        return "default"

    def list_devices(self):
        parts=[]
        for args,title in [(["arecord","-l"],"arecord -l"),(["arecord","-L"],"arecord -L")]:
            try:
                r=subprocess.run(args,capture_output=True,text=True,timeout=5)
                parts.append(f"===== {title} =====\n{(r.stdout or r.stderr or '').strip() or '(no output)'}")
            except Exception as e: parts.append(f"===== {title} =====\nERROR: {e}")
        text='\n\n'.join(parts); self.update(arecord_devices=text); return text

    def run_arecord(self, output_path, duration, interruptible=False):
        cfg=self.get_config(); cmd=["arecord","-D",cfg["device"],"-d",str(int(duration)),"-r",cfg["rate"],"-c",cfg["channels"],"-f","S16_LE","-t","wav",str(output_path)]
        with self.capture_lock:
            proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if interruptible:
                with self.proc_lock: self.live_proc=proc
            try:
                deadline=time.time()+int(duration)+8
                while proc.poll() is None:
                    if interruptible and self.recording_event.is_set():
                        proc.terminate(); return -999,"","interrupted",True
                    if time.time()>deadline:
                        proc.kill(); return -998,"","arecord timed out",False
                    time.sleep(0.05)
                out,err=proc.communicate(timeout=1)
                return proc.returncode,out or '',err or '',False
            finally:
                if interruptible:
                    with self.proc_lock:
                        if self.live_proc is proc: self.live_proc=None

    def stop_live_capture(self):
        self.recording_event.set()
        with self.proc_lock: proc=self.live_proc
        if proc and proc.poll() is None:
            try: proc.terminate(); proc.wait(timeout=1)
            except Exception:
                try: proc.kill()
                except Exception: pass
        time.sleep(0.15)

    def prepare_audio(self, wav_path):
        import soundfile as sf
        data, samplerate = sf.read(str(wav_path), dtype="int16", always_2d=False)
        if getattr(data, "ndim", 1) > 1: data = data.mean(axis=1).astype(np.int16)
        samples=np.asarray(data,dtype=np.float32); rms=float(np.sqrt(np.mean(samples**2))) if samples.size else 0
        current_db=20*math.log10(max(rms,1.0)/32768.0); auto_gain=10**((config.VOICE_TARGET_DB-current_db)/20.0)
        total_gain=max(config.VOICE_RECOGNITION_GAIN, min(config.VOICE_MAX_AUTO_GAIN, max(1.0, auto_gain)))
        boosted=samples*total_gain; peak=float(np.max(np.abs(boosted))) if boosted.size else 0
        if peak>32000: boosted *= 32000.0/peak
        boosted=np.clip(boosted,-32768,32767).astype(np.int16)
        audio=sr.AudioData(boosted.tobytes(), int(samplerate), 2)
        if not shutil.which('flac'):
            bio=io.BytesIO(); sf.write(bio, boosted, int(samplerate), format='FLAC', subtype='PCM_16')
            flac_bytes=bio.getvalue(); audio.get_flac_data=lambda convert_rate=None, convert_width=2: flac_bytes
        return audio

    def parse_command(self, text):
        text=(text or '').lower().strip()
        if config.VOICE_WAKE_WORD not in text: return None, None
        after=text.split(config.VOICE_WAKE_WORD,1)[-1].strip(); command_text=after or text
        for p in config.VOICE_FORWARD_PHRASES:
            if p in command_text: return 'move','forward'
        for p in config.VOICE_BACKWARD_PHRASES:
            if p in command_text: return 'move','backward'
        for p in config.VOICE_LEFT_PHRASES:
            if p in command_text: return 'move','left'
        for p in config.VOICE_RIGHT_PHRASES:
            if p in command_text: return 'move','right'
        for p in config.VOICE_STOP_PHRASES:
            if p in command_text: return 'move','stop'
        if any(command_text.startswith(p) for p in config.VOICE_QUESTION_PREFIXES): return 'question', command_text
        return None, None

    def _motor_worker(self):
        while not self.state.shutdown_event.is_set():
            try: direction, heard = self.command_queue.get(timeout=0.2)
            except queue.Empty: continue
            if self.motors:
                ok,msg=self.motors.execute_move(direction)
                self.update(status='listening' if ok else 'motor error', last_action=msg, last_error='' if ok else msg, last_heard=heard)

    def _ask_hermes(self, question):
        if not self.hermes: return
        context=self.service_report_fn() if self.service_report_fn else None
        result=self.hermes.ask(question, context=context)
        answer=result.get('answer') or result.get('error') or 'Hermes did not answer.'
        self.update(last_action=f"Hermes: {answer[:120]}")
        if self.speech and result.get('ok'): self.speech.say_async(answer)

    def _listen_loop(self):
        if not config.VOICE_ENABLED: self.update(status='disabled'); return
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.update(status='speech_recognition missing', last_error='pip install SpeechRecognition')
            return
        try:
            subprocess.run(["arecord","--version"],capture_output=True,text=True,timeout=3)
        except Exception as e:
            self.update(status='arecord missing', last_error=str(e)); return
        recognizer=sr.Recognizer(); recognizer.dynamic_energy_threshold=True
        self.update(status='listening via arecord')
        while not self.state.shutdown_event.is_set():
            if self.recording_event.is_set(): time.sleep(0.2); continue
            tmp=config.VOICE_SAMPLE_DIR/'.janet_listen_tmp.wav'
            try:
                rc,out,err,interrupted=self.run_arecord(tmp, config.VOICE_PHRASE_TIME_LIMIT, interruptible=True)
                if interrupted: continue
                if rc!=0:
                    self.update(status='mic record error', last_error=(err or out).strip()); time.sleep(1); continue
                stats=wav_audio_stats(tmp)
                if stats.get('level',0)>=config.VOICE_SOUND_LEVEL_THRESHOLD:
                    self.sound_until=time.time()+config.VOICE_SOUND_OVERLAY_SECONDS
                self.update(status='recognising', mic_level=stats.get('level',0), mic_level_db=stats.get('db'))
                audio=self.prepare_audio(tmp)
                text=recognizer.recognize_google(audio).lower().strip()
                if config.VOICE_WAKE_WORD in text: self.wake_until=time.time()+config.VOICE_WAKE_OVERLAY_SECONDS
                self.update(status='listening via arecord', last_heard=text, last_error='')
                kind,value=self.parse_command(text)
                if kind=='move': self.command_queue.put_nowait((value,text)); self.update(last_action=f"Queued {value}")
                elif kind=='question': threading.Thread(target=self._ask_hermes, args=(value,), daemon=True).start()
            except sr.UnknownValueError:
                self.update(status='listening via arecord', last_action='Sound heard, but words not understood', last_error='Could not understand audio')
            except Exception as e:
                self.update(status='voice error', last_error=str(e)); time.sleep(1)
            finally:
                try: os.remove(tmp)
                except Exception: pass
