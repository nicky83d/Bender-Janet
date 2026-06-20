import os
import threading
import time
import numpy as np
try:
    import cv2
except ImportError:
    cv2=None
try:
    import depthai as dai
except ImportError:
    dai=None
from . import config


class VisionManager:
    def __init__(self, state, faces=None, objects=None):
        self.state=state; self.faces=faces; self.objects=objects
        self.thread=None; self.start_lock=threading.Lock(); self.ready=threading.Event(); self.restart_event=threading.Event()
        self.condition=threading.Condition(); self.latest_jpeg=None; self.frame_id=0; self.frame_time=0.0; self.latest_frame=None
        self.model_name=config.DETECTION_MODEL_NAME
        # V14.11 starts the camera immediately at boot so the OAK-D can warm
        # up while Janet runs the rest of her boot checks. Object/face
        # acknowledgements are still suppressed until boot.complete=True.
        self.start_allowed=True

    def allow_start(self, allowed=True):
        self.start_allowed=bool(allowed)
        if self.start_allowed:
            self.state.update(status='front camera allowed to start')

    def start(self, force=False):
        if force:
            self.allow_start(True)
        if not self.start_allowed:
            self.placeholder('Front Camera Starting', 'OAK-D pipeline warming up')
            return
        with self.start_lock:
            if self.thread and self.thread.is_alive(): return
            self.thread=threading.Thread(target=self._front_worker, daemon=True); self.thread.start()

    def set_model(self, model_name):
        self.model_name=str(model_name or config.DETECTION_MODEL_NAME)
        self.restart_event.set()
        self.state.update('detection_model', name=self.model_name, last_scan_message=f'Restarting camera with {self.model_name}')
        return True, f"Camera restart queued with {self.model_name}"

    def placeholder(self, title='Front Camera Starting', subtitle='OAK-D pipeline warming up'):
        if cv2 is None: return b''
        frame=np.zeros((config.FRONT_CAMERA_PLACEHOLDER_HEIGHT, config.FRONT_CAMERA_PLACEHOLDER_WIDTH,3),dtype=np.uint8)
        cv2.putText(frame,title,(30,190),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(frame,subtitle[:70],(30,240),cv2.FONT_HERSHEY_SIMPLEX,0.55,(160,170,190),1)
        self.set_latest_frame(frame)
        return frame

    def set_latest_frame(self, frame):
        if cv2 is None: return False
        ok,jpeg=cv2.imencode('.jpg', frame)
        if not ok: return False
        with self.condition:
            self.latest_jpeg=jpeg.tobytes(); self.frame_id+=1; self.frame_time=time.time(); self.latest_frame=frame.copy(); self.condition.notify_all()
        self.state.update(front_camera_ready=self.ready.is_set(), front_frame_age_seconds=0)
        return True

    def get_latest_jpeg(self):
        with self.condition: return self.latest_jpeg

    def generate_stream(self):
        self.start(); last_id=-1
        while not self.state.shutdown_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.state.shutdown_event.is_set() or (self.latest_jpeg is not None and self.frame_id!=last_id), timeout=1.0)
                jpeg=self.latest_jpeg; fid=self.frame_id
            if jpeg is None:
                self.placeholder(); continue
            last_id=fid
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'

    def frame_norm(self, frame, bbox):
        norm_vals=np.full(len(bbox), frame.shape[0]); norm_vals[::2]=frame.shape[1]
        return (np.clip(np.array(bbox),0,1)*norm_vals).astype(int)

    def draw_voice_overlay(self, frame):
        voice=self.state.section('voice')
        if not (voice.get('wake_word_active') or voice.get('sound_active')): return
        text=(voice.get('last_heard') or voice.get('last_sound_message') or 'audio detected').strip()
        prefix='JANET' if voice.get('wake_word_active') else 'SOUND'
        color=(0,255,0) if prefix=='JANET' else (0,255,255)
        line=f'{prefix}: {text[:90]}'
        cv2.putText(frame,line,(10,frame.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,line,(10,frame.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,0.32,color,1,cv2.LINE_AA)

    def _front_worker(self):
        if cv2 is None or dai is None:
            self.placeholder('Front Cam Unavailable', 'Install opencv/depthai on the Pi')
            self.state.update(status='front camera unavailable', front_camera_error='OpenCV or DepthAI missing')
            return
        self.placeholder()
        while not self.state.shutdown_event.is_set():
            self.restart_event.clear()
            try:
                self.state.update(status='front camera starting')
                with dai.Pipeline() as pipeline:
                    camera=pipeline.create(dai.node.Camera).build()
                    detection=pipeline.create(dai.node.DetectionNetwork).build(camera, dai.NNModelDescription(self.model_name))
                    detection.setConfidenceThreshold(config.DETECTION_CONFIDENCE)
                    label_map=detection.getClasses()
                    self.state.update('detection_model', name=self.model_name, confidence=config.DETECTION_CONFIDENCE, labels_count=len(label_map or []), labels_preview=list(label_map[:12]) if label_map else [])
                    q_rgb=detection.passthrough.createOutputQueue(); q_det=detection.out.createOutputQueue()
                    pipeline.start(); self.ready.set(); self.state.update(status='front camera active', front_camera_ready=True, front_camera_error='')
                    fps_counter=0; fps_timer=time.time(); current_fps=0.0; face_counter=0; last_faces=[]
                    print(f'Front camera worker started with model {self.model_name}')
                    while pipeline.isRunning() and not self.state.shutdown_event.is_set() and not self.restart_event.is_set():
                        in_rgb=q_rgb.get(); in_det=q_det.get(); frame=in_rgb.getCvFrame(); detections=in_det.detections
                        readings=[]
                        for det in detections:
                            bbox=self.frame_norm(frame,(det.xmin,det.ymin,det.xmax,det.ymax))
                            label=label_map[det.label] if label_map and det.label < len(label_map) else str(det.label)
                            conf=int(det.confidence*100)
                            readings.append({'label':label,'confidence':conf,'xmin':det.xmin,'ymin':det.ymin,'xmax':det.xmax,'ymax':det.ymax})
                            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),2)
                            cv2.putText(frame,f'{label} {conf}%',(bbox[0]+10,bbox[1]+25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                        boot = self.state.section('boot')
                        boot_complete = bool(boot.get('complete', False)) or not getattr(config, 'BOOT_SUPPRESS_RECOGNITION_UNTIL_COMPLETE', True)
                        if self.objects and boot_complete:
                            self.objects.handle_detections(readings, frame.shape[1], frame)
                        else:
                            self.state.update('object', last_seen=[], last_message='Object acknowledgement paused during boot routine')
                        face_counter += 1
                        if self.faces and boot_complete and face_counter % max(1, config.FACE_DETECTION_EVERY_N_FRAMES)==0:
                            last_faces=self.faces.process_frame(frame)
                        elif not boot_complete:
                            last_faces=[]
                            self.state.update('face', last_seen=[], last_message='Face acknowledgement paused during boot routine')
                        if self.faces and boot_complete: self.faces.draw_overlays(frame, last_faces)
                        fps_counter+=1; now=time.time(); elapsed=now-fps_timer
                        if elapsed>=1.0:
                            current_fps=fps_counter/elapsed; fps_counter=0; fps_timer=now
                        self.state.update(fps=round(current_fps,1), detections=readings, front_camera_ready=True, front_frame_age_seconds=round(time.time()-self.frame_time,2) if self.frame_time else None)
                        self.draw_voice_overlay(frame); self.set_latest_frame(frame)
            except Exception as exc:
                self.ready.clear(); self.state.update(status='front camera error', front_camera_ready=False, front_camera_error=str(exc)); print(f'Front camera error: {exc}')
                self.placeholder('Front Cam Error', str(exc)[:70]); time.sleep(config.FRONT_CAMERA_RESTART_DELAY)

    def generate_rear_stream(self):
        if cv2 is None:
            return
        while not self.state.shutdown_event.is_set():
            frame=np.zeros((360,640,3),dtype=np.uint8)
            title='Rear Camera Disabled' if not config.REAR_CAMERA_ENABLED else 'Rear Camera Unavailable'
            cv2.putText(frame,title,(30,150),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            ok,jpeg=cv2.imencode('.jpg',frame)
            if ok: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            time.sleep(1)
