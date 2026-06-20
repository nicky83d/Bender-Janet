import os
import re
import time
import threading
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from . import config
from .utils import safe_name, normalise_key, ensure_dir


class FaceManager:
    def __init__(self, state, speech=None, motors=None):
        self.state = state
        self.speech = speech
        self.motors = motors
        self.lock = threading.RLock()
        self.detector = None
        self.db = {"names": [], "templates": np.empty((0, config.FACE_TEMPLATE_SIZE[0]*config.FACE_TEMPLATE_SIZE[1]), dtype=np.float32), "images": []}
        self.enabled = config.FACE_RECOGNITION_ENABLED
        self.threshold = config.FACE_RECOGNITION_THRESHOLD
        self.last_greeted = {}
        self.last_center_turn = 0.0

    def start(self):
        ensure_dir(config.FACE_DIR); ensure_dir(config.FACE_IMAGE_DIR)
        if cv2 is None:
            self.update(status="OpenCV missing", available=False)
            return
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            self.update(status="cascade unavailable", available=False, last_message="OpenCV Haar cascade could not be loaded")
            return
        self.load()
        self.update(status="ready", available=True, enabled=self.enabled, threshold=self.threshold, last_message="Face recognition ready")

    def update(self, **values):
        current=self.state.section("face"); current.update(values)
        current.setdefault("known_faces", self.summary()); current["known_count"] = len(self.db.get("names", [])); current["threshold"] = self.threshold; current["enabled"] = self.enabled
        self.state.update("face", **current)

    def load(self):
        with self.lock:
            if not config.FACE_DB_FILE.exists():
                self.db = {"names": [], "templates": np.empty((0,6400), dtype=np.float32), "images": []}
                self.update(known_faces=[], known_count=0, last_message="No known faces yet")
                return
            data=np.load(str(config.FACE_DB_FILE), allow_pickle=True)
            names=[str(x) for x in data.get("names", np.array([], dtype=object)).tolist()]
            templates=data.get("templates", np.empty((0,6400), dtype=np.float32)).astype(np.float32)
            if templates.ndim==1 and templates.size: templates=templates.reshape(1,-1)
            images=[str(x) for x in data.get("images", np.array([], dtype=object)).tolist()]
            while len(images)<len(names): images.append("")
            self.db={"names": names, "templates": templates, "images": images[:len(names)]}
            self.save()
            self.update(known_faces=self.summary(), known_count=len(names), last_message=f"Loaded {len(names)} known face sample(s)")

    def save(self):
        ensure_dir(config.FACE_DIR); ensure_dir(config.FACE_IMAGE_DIR)
        np.savez_compressed(str(config.FACE_DB_FILE), names=np.array(self.db["names"], dtype=object), templates=self.db["templates"].astype(np.float32), images=np.array(self.db["images"], dtype=object))

    def summary(self):
        grouped={}
        names=list(self.db.get("names", [])); images=list(self.db.get("images", []))
        for idx,name in enumerate(names):
            grouped.setdefault(name,{"name":name,"samples":0,"images":[]})
            grouped[name]["samples"] += 1
            fn=os.path.basename(images[idx]) if idx < len(images) and images[idx] else ""
            if fn:
                grouped[name]["images"].append({"sample":idx+1,"sample_index":idx,"filename":fn,"url":f"/face_image/{fn}","thumb_url":f"/face_image/{fn}"})
        return [grouped[k] for k in sorted(grouped)]

    def detect(self, frame):
        if self.detector is None or frame is None: return []
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray=cv2.equalizeHist(gray)
        faces=self.detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=config.FACE_MIN_SIZE, flags=cv2.CASCADE_SCALE_IMAGE)
        return [tuple(map(int,f)) for f in faces]

    def template_from_box(self, frame, box):
        x,y,w,h=[int(v) for v in box]
        H,W=frame.shape[:2]; pad=int(max(w,h)*0.18)
        x1=max(0,x-pad); y1=max(0,y-pad); x2=min(W,x+w+pad); y2=min(H,y+h+pad)
        if x2<=x1 or y2<=y1: return None
        roi=frame[y1:y2,x1:x2]
        gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY); gray=cv2.resize(gray, config.FACE_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA); gray=cv2.equalizeHist(gray)
        return gray.astype(np.float32).reshape(-1)

    def recognise_template(self, template):
        names=self.db.get("names", []); templates=self.db.get("templates")
        if template is None or templates is None or len(names)==0 or templates.size==0: return "Unknown", None
        diffs=np.mean(np.abs(templates - template.reshape(1,-1)), axis=1)
        idx=int(np.argmin(diffs)); score=float(diffs[idx])
        if score <= float(self.threshold): return names[idx], round(score,1)
        return "Unknown", round(score,1)

    def process_frame(self, frame):
        if not self.enabled:
            self.update(last_seen=[]); return []
        results=[]; frame_w=frame.shape[1]
        for x,y,w,h in self.detect(frame):
            name,score=self.recognise_template(self.template_from_box(frame,(x,y,w,h)))
            item={"name":name,"score":score,"x":int(x),"y":int(y),"w":int(w),"h":int(h)}
            results.append(item)
            if name and name.lower()!="unknown":
                self.greet(name)
                self.center_on_face(name, x+w/2, frame_w)
        self.update(status="enabled", last_seen=results)
        return results

    def greet(self, name):
        key=normalise_key(name); now=time.time()
        last=self.last_greeted.get(key, 0)
        if now-last < config.FACE_GREETING_COOLDOWN_SECONDS: return
        self.last_greeted[key]=now
        phrase=config.FACE_GREETING_TEMPLATE.format(name=name)
        self.update(last_greeting=phrase, last_greeting_person=name, last_greeting_time=now, last_message=f"Greeting {name}")
        if self.speech: self.speech.say_async(phrase, allow_beep_fallback=False)

    def center_on_face(self, name, center_x, frame_width):
        if not self.motors or not config.TARGET_CENTERING_ENABLED: return
        offset=(float(center_x)/float(frame_width))-0.5
        if abs(offset) <= config.TARGET_CENTERING_DEADBAND: return
        now=time.time()
        if now-self.last_center_turn < config.TARGET_CENTERING_COOLDOWN_SECONDS: return
        self.last_center_turn=now
        direction="right" if offset>0 else "left"
        self.motors.execute_move_async(direction, duration=config.TARGET_CENTERING_TURN_DURATION, acceleration=0.0)

    def draw_overlays(self, frame, results):
        for face in results or []:
            x,y,w,h = int(face["x"]), int(face["y"]), int(face["w"]), int(face["h"])
            name=face.get("name","Unknown"); score=face.get("score")
            color=(87,242,135) if name!="Unknown" else (255,204,102)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            label=name if score is None else f"{name} {score}"
            cv2.putText(frame,label,(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)

    def _save_face_image(self, frame, box, name, idx):
        x,y,w,h=[int(v) for v in box]; H,W=frame.shape[:2]; pad=int(max(w,h)*0.25)
        x1=max(0,x-pad); y1=max(0,y-pad); x2=min(W,x+w+pad); y2=min(H,y+h+pad)
        if x2<=x1 or y2<=y1: return ""
        crop=cv2.resize(frame[y1:y2,x1:x2], (140,140), interpolation=cv2.INTER_AREA)
        fn=f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(name)}_{idx:04d}.jpg"
        cv2.imwrite(str(config.FACE_IMAGE_DIR/fn), crop)
        return fn

    def add_from_frame(self, name, frame):
        name=re.sub(r"[^A-Za-z0-9 _.-]+", "", str(name or "")).strip()
        if not name: return False,"Please enter a name"
        faces=self.detect(frame)
        if not faces: return False,"No face found in current frame"
        box=max(faces, key=lambda f:f[2]*f[3])
        template=self.template_from_box(frame, box)
        if template is None: return False,"Could not create face template"
        with self.lock:
            idx=len(self.db["names"])
            fn=self._save_face_image(frame, box, name, idx)
            self.db["names"].append(name); self.db["images"].append(fn)
            self.db["templates"] = template.reshape(1,-1).astype(np.float32) if self.db["templates"].size==0 else np.vstack([self.db["templates"], template.reshape(1,-1).astype(np.float32)])
            self.save(); self.update(known_faces=self.summary(), known_count=len(self.db["names"]), last_message=f"Added face sample for {name}")
        return True, f"Added face sample for {name}"

    def add_from_uploaded_file(self, name, file_storage):
        if cv2 is None: return False,"OpenCV unavailable"
        data=np.frombuffer(file_storage.read(), dtype=np.uint8)
        frame=cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None: return False,"Could not read uploaded image"
        return self.add_from_frame(name, frame)

    def remove_sample(self, sample_index):
        try: idx=int(sample_index)
        except Exception: return False,"Face sample not found"
        with self.lock:
            if idx<0 or idx>=len(self.db["names"]): return False,"Face sample not found"
            removed=self.db["names"][idx]
            fn=self.db["images"][idx] if idx<len(self.db["images"]) else ""
            if fn:
                try: os.remove(config.FACE_IMAGE_DIR/os.path.basename(fn))
                except Exception: pass
            keep=[i for i in range(len(self.db["names"])) if i!=idx]
            self.db["names"]=[self.db["names"][i] for i in keep]
            self.db["images"]=[self.db["images"][i] for i in keep]
            self.db["templates"]=self.db["templates"][keep] if keep else np.empty((0,6400), dtype=np.float32)
            self.save(); self.update(known_faces=self.summary(), known_count=len(self.db["names"]), last_message=f"Removed one face sample for {removed}")
        return True, f"Removed one face sample for {removed}"

    def remove_name(self, name):
        with self.lock:
            indexes=[i for i,n in enumerate(self.db["names"]) if n==name]
            if not indexes: return False,f"No known face named {name}"
            for i in indexes:
                fn=self.db["images"][i] if i<len(self.db["images"]) else ""
                if fn:
                    try: os.remove(config.FACE_IMAGE_DIR/os.path.basename(fn))
                    except Exception: pass
            keep=[i for i in range(len(self.db["names"])) if i not in indexes]
            self.db["names"]=[self.db["names"][i] for i in keep]
            self.db["images"]=[self.db["images"][i] for i in keep]
            self.db["templates"]=self.db["templates"][keep] if keep else np.empty((0,6400), dtype=np.float32)
            self.save(); self.update(known_faces=self.summary(), known_count=len(self.db["names"]), last_message=f"Removed {len(indexes)} sample(s) for {name}")
        return True, f"Removed {len(indexes)} sample(s) for {name}"

    def set_settings(self, enabled=None, threshold=None):
        if enabled is not None: self.enabled=bool(enabled)
        if threshold is not None:
            try: self.threshold=float(threshold)
            except Exception: pass
        self.update(enabled=self.enabled, threshold=self.threshold, last_message="Face settings saved")
        return {"enabled": self.enabled, "threshold": self.threshold}
