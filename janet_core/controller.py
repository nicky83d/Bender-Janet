import time
from .state import JanetState
from .motors import MotorManager
from .sonars import SonarManager
from .speech import SpeechManager
from .faces import FaceManager
from .objects import ObjectMemory
from .vision import VisionManager
from .voice import VoiceManager
from .hermes import HermesClient
from .boot import BootManager
from .utils import format_cm
from . import config


class JanetController:
    def __init__(self):
        self.state = JanetState()
        self.motors = MotorManager(self.state)
        self.sonars = SonarManager(self.state)
        self.speech = SpeechManager(self.state)
        self.hermes = HermesClient(self.state)
        self.speech.set_translation_provider(self.hermes.translate_to_chinese)
        self.faces = FaceManager(self.state, speech=self.speech, motors=self.motors)
        self.objects = ObjectMemory(self.state, speech=self.speech, motors=self.motors)
        self.vision = VisionManager(self.state, faces=self.faces, objects=self.objects)
        self.voice = VoiceManager(self.state, motors=self.motors, hermes=self.hermes, speech=self.speech, service_report_fn=self.build_service_report)
        self.boot = BootManager(self.state, self.speech, self.motors, self.sonars, self.vision, self.hermes, voice=self.voice)

    def start(self):
        # Start the OAK-D camera first so it can warm up while Janet completes
        # the rest of her boot routine. Face/object acknowledgement stays
        # suppressed by the boot state until boot.complete=True.
        self.state.update(status="Janet V14 booting")
        self.vision.allow_start(True)
        self.vision.start(force=True)

        # Start the other low-level systems. The voice listener still waits
        # until after the boot routine, so Janet does not hear herself talking.
        self.speech.start()
        self.motors.start()
        self.faces.start()
        self.objects.start()
        self.sonars.start()
        if getattr(config, "BOOT_ROUTINE_ENABLED", True):
            self.boot.start()
        else:
            self.voice.start()
            self.state.update(status="Janet V14 running")

    def stop(self):
        self.state.shutdown_event.set()
        try: self.motors.execute_move('stop')
        except Exception: pass

    def build_service_report(self):
        data = self.state.get()
        sonar = data.get('sonar', {})
        detections = data.get('detections', [])
        face = data.get('face', {})
        voice = data.get('voice', {})
        objects = data.get('object', {})
        boot = data.get('boot', {})
        obj_line = ', '.join(f"{d.get('label')} ({d.get('confidence')}%)" for d in detections[:6]) or 'nothing specific right now'
        face_names = ', '.join(f.get('name', 'Unknown') for f in face.get('last_seen', [])[:6]) or 'no faces currently'
        return '\n'.join([
            'Janet status report',
            f"Boot: {boot.get('step', 'unknown')} | complete: {boot.get('complete', False)}",
            f"FPS: {data.get('fps', 0)}",
            f"Objects in camera: {obj_line}",
            f"Faces in camera: {face_names}",
            f"Known objects: {objects.get('known_count', 0)} labels, {objects.get('samples_count', 0)} samples",
            f"Front sonar: {format_cm(sonar.get('front'))}",
            f"Back sonar: {format_cm(sonar.get('back'))}",
            f"Left sonar: {format_cm(sonar.get('left'))}",
            f"Right sonar: {format_cm(sonar.get('right'))}",
            f"Last heard: {voice.get('last_heard') or 'nothing recognised yet'}",
        ])
