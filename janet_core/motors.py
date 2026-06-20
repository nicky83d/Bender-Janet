import queue
import random
import threading
import time
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
from . import config
from .utils import clamp_float


class MotorControllerI2C:
    def __init__(self, bus_num=config.MOTOR_I2C_BUS, addr=config.MOTOR_I2C_ADDR):
        if not I2C_AVAILABLE:
            raise RuntimeError("smbus/smbus2 is not installed")
        self.bus = smbus.SMBus(bus_num)
        self.addr = addr
        self.lock = threading.Lock()

    def send_command(self, command, duration=None):
        duration_ms = max(0, min(int(float(duration or 0) * 1000), 65535))
        payload = [duration_ms & 0xFF, (duration_ms >> 8) & 0xFF]
        with self.lock:
            self.bus.write_i2c_block_data(self.addr, command, payload)

    def forward(self, duration): self.send_command(0x01, duration)
    def backward(self, duration): self.send_command(0x02, duration)
    def left(self, duration): self.send_command(0x03, duration)
    def right(self, duration): self.send_command(0x04, duration)
    def stop(self, duration=0): self.send_command(0x00, duration)


class MotorManager:
    def __init__(self, state):
        self.state = state
        self.motors = None
        self.settings_lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.routine_stop_event = threading.Event()
        self.routine_lock = threading.Lock()
        self.routine_thread = None
        self.routine_state = {"active": False, "current_id": "", "current_name": "", "started_at": None, "steps_done": 0}
        self.settings = {"duration": config.MOTOR_DEFAULT_DURATION, "acceleration": config.MOTOR_DEFAULT_ACCELERATION, "preset": "normal"}

    def start(self):
        try:
            self.motors = MotorControllerI2C()
            print(f"MotorController initialized on I2C bus {config.MOTOR_I2C_BUS}, addr {hex(config.MOTOR_I2C_ADDR)}")
        except Exception as e:
            print(f"Motor init failed: {e}")
            self.motors = None
        self.state.update(motors_available=self.motors is not None, motor_settings=self.get_settings())
        self._update_routine_status(routines=self.routine_definitions())

    def get_settings(self):
        with self.settings_lock:
            return dict(self.settings)

    def set_settings(self, duration=None, acceleration=None, preset=None):
        with self.settings_lock:
            if duration is not None:
                self.settings["duration"] = clamp_float(duration, config.MOTOR_MIN_DURATION, config.MOTOR_MAX_DURATION, self.settings["duration"])
            if acceleration is not None:
                self.settings["acceleration"] = clamp_float(acceleration, config.MOTOR_MIN_ACCELERATION, config.MOTOR_MAX_ACCELERATION, self.settings["acceleration"])
            if preset is not None:
                self.settings["preset"] = preset
        self.state.update(motor_settings=self.get_settings())
        return self.get_settings()

    def apply_preset(self, preset):
        chosen = config.MOTOR_PRESETS.get(preset)
        if not chosen:
            return False, "Unknown preset"
        self.set_settings(chosen["duration"], chosen["acceleration"], preset)
        return True, f"{chosen['label']} applied"

    def _action_for(self, direction):
        if not self.motors:
            return None
        if config.REVERSE_MOTOR_LOGIC:
            mapping = {"forward": self.motors.backward, "backward": self.motors.forward, "left": self.motors.right, "right": self.motors.left, "stop": self.motors.stop}
        else:
            mapping = {"forward": self.motors.forward, "backward": self.motors.backward, "left": self.motors.left, "right": self.motors.right, "stop": self.motors.stop}
        return mapping.get(direction)

    def execute_move(self, direction, duration=None, acceleration=None):
        if direction not in {"forward", "backward", "left", "right", "stop"}:
            return False, "Invalid direction"
        if not self.motors:
            return False, "Motors not available"
        settings = self.get_settings()
        duration = clamp_float(duration, config.MOTOR_MIN_DURATION, config.MOTOR_MAX_DURATION, settings["duration"])
        acceleration = clamp_float(acceleration, config.MOTOR_MIN_ACCELERATION, config.MOTOR_MAX_ACCELERATION, settings["acceleration"])
        action = self._action_for(direction)
        try:
            if direction == "stop":
                self.stop_event.set()
                action(0)
                return True, "Stopped"
            self.stop_event.set(); time.sleep(0.005); self.stop_event.clear()
            with self.run_lock:
                if acceleration <= 0.01:
                    action(duration)
                else:
                    steps = max(2, min(8, int(2 + acceleration * 6)))
                    chunk = max(0.03, duration / steps)
                    for step in range(steps):
                        if self.stop_event.is_set(): break
                        action(chunk)
                        pause = 0.08 * acceleration * (1 - ((step + 1) / steps))
                        if pause > 0: time.sleep(pause)
            return True, f"Moving {direction} for {duration:.2f}s"
        except Exception as e:
            print(f"Motor command failed: {e}")
            return False, str(e)

    def execute_move_async(self, direction, duration=None, acceleration=None):
        if direction == "stop":
            return self.execute_move("stop", duration, acceleration)
        t = threading.Thread(target=self.execute_move, args=(direction,), kwargs={"duration": duration, "acceleration": acceleration}, daemon=True)
        t.start()
        return True, f"Queued {direction}"

    def routine_definitions(self):
        return [{"id": rid, "name": data["name"], "emoji": data.get("emoji", "🤖"), "description": data.get("description", ""), "duration_seconds": config.MOTOR_DANCE_DURATION_SECONDS, "return_home": True, "pattern": list(data.get("pattern", []))} for rid, data in config.MOTOR_DANCE_ROUTINES.items()]

    def _update_routine_status(self, **values):
        current = self.state.section("routines")
        current.update(values)
        current["routines"] = self.routine_definitions()
        self.state.update("routines", **current)

    def get_routine_status(self):
        with self.routine_lock:
            if self.routine_state.get("active") and self.routine_state.get("started_at"):
                elapsed = time.time() - self.routine_state["started_at"]
            else:
                elapsed = self.state.section("routines").get("elapsed_seconds", 0.0)
        self._update_routine_status(elapsed_seconds=round(elapsed, 1))
        return self.state.section("routines")

    @staticmethod
    def routine_is_balanced(pattern):
        return pattern.count("forward") == pattern.count("backward") and pattern.count("left") == pattern.count("right")

    def start_routine(self, routine_id):
        if routine_id == "random":
            routine_id = random.choice(list(config.MOTOR_DANCE_ROUTINES.keys()))
        if routine_id not in config.MOTOR_DANCE_ROUTINES:
            return False, f"Unknown routine: {routine_id}"
        if not self.motors:
            return False, "Motors not available"
        with self.routine_lock:
            if self.routine_state.get("active"):
                return False, f"A routine is already running: {self.routine_state.get('current_name')}"
            routine = config.MOTOR_DANCE_ROUTINES[routine_id]
            self.routine_stop_event.clear()
            self.routine_state.update({"active": True, "current_id": routine_id, "current_name": routine["name"], "started_at": time.time(), "steps_done": 0})
            self.routine_thread = threading.Thread(target=self._run_routine, args=(routine_id,), daemon=True)
            self.routine_thread.start()
        self._update_routine_status(active=True, current_id=routine_id, current_name=routine["name"], status="queued", last_message=f"Queued {routine['name']}", elapsed_seconds=0, steps_done=0)
        return True, f"Started {routine['name']}"

    def stop_routine(self):
        self.routine_stop_event.set()
        self.execute_move("stop")
        self._update_routine_status(status="stopping", last_message="Stopping routine...")

    def _run_routine(self, routine_id):
        routine = config.MOTOR_DANCE_ROUTINES[routine_id]
        name = routine["name"]
        pattern = [d for d in routine.get("pattern", []) if d in {"forward", "backward", "left", "right"}]
        if not pattern or not self.routine_is_balanced(pattern):
            self._update_routine_status(active=False, status="error", last_message=f"Routine {name} is not valid/balanced")
            return
        settings = self.get_settings()
        step_duration = clamp_float(settings.get("duration"), config.MOTOR_MIN_DURATION, config.MOTOR_MAX_DURATION, config.MOTOR_DEFAULT_DURATION)
        acceleration = settings.get("acceleration", 0.0)
        target_seconds = float(config.MOTOR_DANCE_DURATION_SECONDS)
        cycle_seconds = len(pattern) * (step_duration + config.MOTOR_DANCE_STEP_GAP)
        if cycle_seconds > target_seconds:
            step_duration = max(config.MOTOR_MIN_DURATION, (target_seconds / len(pattern)) - config.MOTOR_DANCE_STEP_GAP)
            cycle_seconds = len(pattern) * (step_duration + config.MOTOR_DANCE_STEP_GAP)
        start = time.time(); end_time = start + target_seconds
        steps_done = 0; cycles_done = 0
        try:
            while not self.state.shutdown_event.is_set() and not self.routine_stop_event.is_set():
                if end_time - time.time() < cycle_seconds: break
                for direction in pattern:
                    if self.state.shutdown_event.is_set() or self.routine_stop_event.is_set(): break
                    ok, msg = self.execute_move(direction, duration=step_duration, acceleration=acceleration)
                    steps_done += 1
                    self._update_routine_status(active=True, status="running", last_message=f"{name}: cycle {cycles_done+1}, step {steps_done} {direction}", steps_done=steps_done, elapsed_seconds=round(time.time()-start,1))
                    time.sleep(max(0.02, step_duration + config.MOTOR_DANCE_STEP_GAP))
                    if not ok: self.routine_stop_event.set(); break
                if self.routine_stop_event.is_set(): break
                cycles_done += 1
            self.execute_move("stop")
            while not self.routine_stop_event.is_set() and time.time() < end_time:
                self._update_routine_status(status="return-home hold", last_message=f"{name}: back at start, holding", steps_done=steps_done, elapsed_seconds=round(time.time()-start,1))
                time.sleep(0.25)
        finally:
            self.execute_move("stop")
            stopped = self.routine_stop_event.is_set()
            self.routine_stop_event.clear()
            elapsed = time.time() - start
            with self.routine_lock:
                self.routine_state.update({"active": False, "current_id": "", "current_name": "", "started_at": None, "steps_done": steps_done})
            msg = f"Stopped {name} after {elapsed:.1f}s" if stopped else f"Completed {name}: {cycles_done} balanced cycle(s), {steps_done} step(s)"
            self._update_routine_status(active=False, current_id="", current_name="", status="stopped" if stopped else "ready", last_message=msg, last_completed=name, elapsed_seconds=round(elapsed,1), steps_done=steps_done)
