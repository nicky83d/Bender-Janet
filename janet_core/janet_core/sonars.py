import threading
import time
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    GPIO = None
from . import config


class SonarArray:
    def __init__(self):
        if not GPIO_AVAILABLE:
            raise RuntimeError("RPi.GPIO not available")
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        for name in config.TRIG_PINS:
            GPIO.setup(config.TRIG_PINS[name], GPIO.OUT)
            GPIO.setup(config.ECHO_PINS[name], GPIO.IN)
            GPIO.output(config.TRIG_PINS[name], False)
        time.sleep(0.2)

    def _measure(self, trig, echo):
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)
        timeout_start = time.time(); pulse_start = pulse_end = None
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
        return {name: self._measure(config.TRIG_PINS[name], config.ECHO_PINS[name]) for name in config.TRIG_PINS}

    def cleanup(self):
        if GPIO_AVAILABLE:
            GPIO.cleanup()


class SonarManager:
    def __init__(self, state):
        self.state = state
        self.thread = None

    def start(self):
        self.state.update(sonar_available=GPIO_AVAILABLE)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        if not GPIO_AVAILABLE:
            self.state.update(status="sonars unavailable")
            return
        sonar = None
        try:
            sonar = SonarArray()
            self.state.update(sonar_available=True)
            while not self.state.shutdown_event.is_set():
                self.state.update(sonar=sonar.get_readings())
                time.sleep(config.SONAR_INTERVAL)
        except Exception as e:
            print(f"Sonar error: {e}")
            self.state.update(sonar_available=False, status=f"sonar error: {e}")
        finally:
            if sonar:
                sonar.cleanup()
