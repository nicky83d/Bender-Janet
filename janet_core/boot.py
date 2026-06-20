import threading
import time
from . import config
from .utils import format_cm


class BootManager:
    """Ordered boot routine for Janet.

    The routine speaks each result in English and Chinese before moving to the
    next step. Object/face acknowledgement stays suppressed until this routine
    marks boot.complete=True in state.
    """

    def __init__(self, state, speech, motors, sonars, vision, hermes, voice=None):
        self.state = state
        self.speech = speech
        self.motors = motors
        self.sonars = sonars
        self.vision = vision
        self.hermes = hermes
        self.voice = voice
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if not getattr(config, "BOOT_ROUTINE_ENABLED", True):
            self.state.update("boot", active=False, complete=True, step="disabled", last_message="Boot routine disabled")
            try:
                self.vision.allow_start(True)
                self.vision.start(force=True)
                if self.voice:
                    self.voice.start()
            except Exception:
                pass
            return False
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return True

    def _set_step(self, index, step, message="", error=""):
        history = []
        try:
            history = self.state.section("boot").get("history", [])
            history = list(history[-20:])
        except Exception:
            history = []
        if message:
            history.append({"time": time.time(), "step_index": index, "step": step, "message": message, "error": error})
        self.state.update(
            "boot",
            active=True,
            complete=False,
            step_index=index,
            total_steps=config.BOOT_TOTAL_STEPS,
            step=step,
            last_message=message,
            last_error=error,
            history=history,
        )

    def _finish(self, message="Boot complete"):
        history = list(self.state.section("boot").get("history", [])[-20:])
        history.append({"time": time.time(), "step_index": config.BOOT_TOTAL_STEPS, "step": "complete", "message": message, "error": ""})
        self.state.update(
            "boot",
            active=False,
            complete=True,
            step_index=config.BOOT_TOTAL_STEPS,
            step="complete",
            last_message=message,
            last_error="",
            completed_at=time.time(),
            history=history,
        )

    def say_step(self, english, chinese):
        """Say one short bilingual boot result, then continue quickly."""
        self.state.update("boot", last_message=english)
        if getattr(config, "BOOT_SPEECH_ENABLED", True):
            self.speech.say_bilingual(english, chinese, allow_beep_fallback=False)
        pause = float(getattr(config, "BOOT_STEP_PAUSE_SECONDS", 0.05))
        if pause > 0:
            time.sleep(pause)

    def _current_sonar(self):
        return self.state.section("sonar")

    def _wait_for_sonar(self, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not self.state.shutdown_event.is_set():
            readings = self._current_sonar()
            if any(isinstance(v, (int, float)) and v > 0 for v in readings.values()):
                return readings
            time.sleep(0.15)
        return self._current_sonar()

    @staticmethod
    def _changed(before, after, threshold):
        changes = {}
        for key in ("front", "back", "left", "right"):
            try:
                b = float(before.get(key, -1))
                a = float(after.get(key, -1))
            except Exception:
                continue
            if b > 0 and a > 0:
                changes[key] = round(a - b, 2)
        return changes, any(abs(v) >= threshold for v in changes.values())

    @staticmethod
    def _sonar_sentence(readings):
        return "Front {front}, back {back}, left {left}, and right {right}.".format(
            front=format_cm(readings.get("front")),
            back=format_cm(readings.get("back")),
            left=format_cm(readings.get("left")),
            right=format_cm(readings.get("right")),
        )

    @staticmethod
    def _sonar_sentence_zh(readings):
        def cm(v):
            try:
                v = float(v)
                if v > 0:
                    return f"{v:.0f}厘米"
            except Exception:
                pass
            return "没有读数"
        return f"前方{cm(readings.get('front'))}，后方{cm(readings.get('back'))}，左边{cm(readings.get('left'))}，右边{cm(readings.get('right'))}。"

    def _run(self):
        self.state.update("boot", active=True, complete=False, started_at=time.time(), last_error="")
        try:
            time.sleep(max(0.0, float(getattr(config, "BOOT_INITIAL_DELAY_SECONDS", 1.2))))

            # 1 - Wake/speaker announcement.
            self._set_step(1, "wake announcement", "Step 1: wake announcement")
            self.say_step(
                "Hello, I am Bender-Janet. I am awake, and my speaker is working.",
                "你好，我是 Bender-Janet。我醒了，扬声器工作正常。",
            )

            # 2 - Small motor movement only. We only confirm the motor commands work here;
            # sonar readings are checked separately in step 4.
            self._set_step(2, "motor movement test", "Step 2: testing motors")
            if not getattr(self.motors, "motors", None):
                self.say_step(
                    "Motor test skipped. Motor controller unavailable.",
                    "电机测试已跳过。电机控制器不可用。",
                )
            else:
                duration = float(getattr(config, "BOOT_MOVEMENT_DURATION", 0.22))
                ok_forward, msg_forward = self.motors.execute_move("forward", duration=duration, acceleration=0.0)
                time.sleep(duration + float(getattr(config, "BOOT_MOVEMENT_SETTLE_SECONDS", 0.45)))
                ok_backward, msg_backward = self.motors.execute_move("backward", duration=duration, acceleration=0.0)
                time.sleep(duration + float(getattr(config, "BOOT_MOVEMENT_SETTLE_SECONDS", 0.45)))
                self.motors.execute_move("stop")
                if ok_forward and ok_backward:
                    self.say_step(
                        "Motors working.",
                        "电机工作正常。",
                    )
                else:
                    self.say_step(
                        "Motor warning. Movement command failed.",
                        "电机警告。移动指令失败。",
                    )

            # 3 - Camera readiness. The camera was started first by JanetController
            # so it could warm up while steps 1 and 2 were running.
            self._set_step(3, "camera", "Step 3: checking camera")
            self.vision.allow_start(True)
            self.vision.start(force=True)
            deadline = time.time() + float(getattr(config, "BOOT_CAMERA_READY_TIMEOUT_SECONDS", 25.0))
            while time.time() < deadline and not self.state.shutdown_event.is_set():
                data = self.state.get()
                if data.get("front_camera_ready"):
                    break
                time.sleep(0.25)
            data = self.state.get()
            if data.get("front_camera_ready"):
                self.say_step(
                    "Camera working.",
                    "摄像头工作正常。",
                )
            else:
                self.say_step(
                    "Camera warning. Camera is not ready yet.",
                    "摄像头警告。摄像头还没有准备好。",
                )

            # 4 - Check all sonar sensors internally, but do not speak values.
            self._set_step(4, "sonar check", "Step 4: checking sonars")
            readings = self._wait_for_sonar(timeout=2.0)
            working = []
            for name in ("front", "back", "left", "right"):
                try:
                    if float(readings.get(name, -1)) > 0:
                        working.append(name)
                except Exception:
                    pass
            if len(working) == 4:
                self.say_step(
                    "Sonars working.",
                    "声呐传感器工作正常。",
                )
            elif working:
                self.say_step(
                    "Sonar warning. Some sensors are working.",
                    "声呐警告。部分传感器正在工作。",
                )
            else:
                self.say_step(
                    "Sonar warning. No clear readings.",
                    "声呐警告。没有清楚的读数。",
                )

            # 5 - Hermes and weather. Repair/discover automatically, then keep
            # spoken feedback short: success or issue only.
            self._set_step(5, "Hermes link and weather", "Step 5: checking Hermes and asking weather")
            repair = self.hermes.connect_or_repair(do_chat=True, save=True)
            hermes_ok = bool(repair.get("ok"))
            if hermes_ok:
                self.say_step(
                    "Hermes working.",
                    "Hermes 工作正常。",
                )
                question = getattr(config, "BOOT_WEATHER_QUESTION", "What is the weather forecast for Bournemouth, UK tomorrow? Reply in one short spoken sentence only.")
                weather = self.hermes.ask(question, max_tokens=90, auto_discover=True)
                if weather.get("ok"):
                    answer = str(weather.get("answer", "")).strip()
                    if answer:
                        answer = " ".join(answer.split())[:220]
                        self.say_step(
                            answer,
                            self.speech.translate_to_chinese(answer),
                        )
                    else:
                        self.say_step(
                            "Weather reply empty.",
                            "天气回复为空。",
                        )
                else:
                    self.say_step(
                        "Weather issue.",
                        "天气信息有问题。",
                    )
            else:
                self.say_step(
                    "Hermes issue.",
                    "Hermes 有问题。",
                )

            # 6 - Complete.
            self._set_step(6, "boot complete", "Step 6: boot complete")
            self.say_step(
                "Boot complete. I am ready.",
                "启动完成。我准备好了。",
            )
            self._finish("Boot routine complete. Janet is ready.")
            if self.voice:
                self.voice.start()
        except Exception as e:
            self.state.update("boot", active=False, complete=False, last_error=str(e), step="boot error", last_message="Boot routine failed")
            try:
                self.speech.say_bilingual(
                    f"Boot routine error. {e}",
                    "启动流程发生错误。请检查 Janet 的日志。",
                    allow_beep_fallback=False,
                )
            except Exception:
                pass
            # Keep Janet usable even if boot routine fails.
            try:
                self.vision.allow_start(True)
                self.vision.start(force=True)
                if self.voice:
                    self.voice.start()
            except Exception:
                pass
