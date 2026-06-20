import json
import os
import time
import threading
try:
    import cv2
except ImportError:
    cv2 = None
from . import config
from .utils import ensure_dir, normalise_key, pretty_label, safe_name


class ObjectMemory:
    def __init__(self, state, speech=None, motors=None):
        self.state = state
        self.speech = speech
        self.motors = motors
        self.lock = threading.RLock()
        self.samples = []
        self.last_ack = {}
        self.last_center_turn = 0.0

    def start(self):
        ensure_dir(config.OBJECT_DIR); ensure_dir(config.OBJECT_IMAGE_DIR)
        self.load()
        self.update(status="ready", last_message=f"Loaded {len(self.samples)} object sample(s)")

    def update(self, **values):
        current=self.state.section("object")
        current.update(values)
        current["known_objects"] = self.summary()
        current["known_count"] = len({s.get("key") for s in self.samples if s.get("key")})
        current["samples_count"] = len(self.samples)
        self.state.update("object", **current)

    def load(self):
        with self.lock:
            if not config.OBJECT_DB_FILE.exists():
                self.samples=[]; self.save(); return
            try:
                data=json.load(open(config.OBJECT_DB_FILE, encoding="utf-8"))
                raw=data.get("samples", []) if isinstance(data, dict) else []
                cleaned=[]
                for i,s in enumerate(raw):
                    if not isinstance(s, dict): continue
                    label=pretty_label(s.get("label") or s.get("name") or "object")
                    key=normalise_key(label)
                    fn=os.path.basename(str(s.get("filename") or ""))
                    cleaned.append({"id": str(s.get("id") or f"obj_{i:05d}"), "label": label, "key": key, "filename": fn, "timestamp": float(s.get("timestamp") or 0), "confidence": int(s.get("confidence") or 0)})
                self.samples=cleaned
                for s in cleaned:
                    if s.get("key"):
                        self.last_ack[s["key"]]=max(self.last_ack.get(s["key"],0), float(s.get("timestamp") or 0))
                self.save()
            except Exception as e:
                print(f"Object DB load error: {e}")
                self.samples=[]

    def save(self):
        ensure_dir(config.OBJECT_DIR)
        json.dump({"samples": self.samples}, open(config.OBJECT_DB_FILE, "w", encoding="utf-8"), indent=2)

    def summary(self):
        grouped={}
        for idx,s in enumerate(self.samples):
            label=pretty_label(s.get("label","object")); key=s.get("key") or normalise_key(label)
            grouped.setdefault(key, {"label": label, "key": key, "samples": 0, "images": []})
            grouped[key]["samples"] += 1
            fn=os.path.basename(s.get("filename") or "")
            if fn:
                grouped[key]["images"].append({"sample": idx+1, "sample_index": idx, "id": s.get("id"), "filename": fn, "url": f"/object_image/{fn}", "thumb_url": f"/object_image/{fn}", "confidence": s.get("confidence",0), "timestamp": s.get("timestamp",0)})
        return [grouped[k] for k in sorted(grouped, key=lambda x: grouped[x]["label"].lower())]

    def _crop(self, frame, det):
        if frame is None or cv2 is None: return None
        H,W=frame.shape[:2]
        try:
            x1=int(max(0,min(1,float(det.get("xmin",0))))*W); y1=int(max(0,min(1,float(det.get("ymin",0))))*H)
            x2=int(max(0,min(1,float(det.get("xmax",0))))*W); y2=int(max(0,min(1,float(det.get("ymax",0))))*H)
        except Exception: return None
        pad=int(max(8,max(x2-x1,y2-y1)*0.12)); x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(W,x2+pad); y2=min(H,y2+pad)
        if x2<=x1 or y2<=y1: return None
        crop=frame[y1:y2,x1:x2].copy()
        return crop if crop.size else None

    def remember(self, label, det, frame):
        label=pretty_label(label); key=normalise_key(label)
        if not key or key in config.OBJECT_SKIP_LABELS: return False,"Skipped object"
        crop=self._crop(frame, det)
        if crop is None: return False,"Could not crop object"
        if cv2 is not None:
            crop=cv2.resize(crop,(160,160),interpolation=cv2.INTER_AREA)
        fn=f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(label)}.jpg"
        if cv2 is not None:
            cv2.imwrite(str(config.OBJECT_IMAGE_DIR/fn), crop)
        with self.lock:
            same=[i for i,s in enumerate(self.samples) if s.get("key")==key]
            while len(same)>=config.OBJECT_MAX_SAMPLES_PER_LABEL:
                old=self.samples.pop(same.pop(0))
                try: os.remove(config.OBJECT_IMAGE_DIR/os.path.basename(old.get("filename","")))
                except Exception: pass
                same=[i for i,s in enumerate(self.samples) if s.get("key")==key]
            conf=int(float(det.get("confidence",0))) if det else 0
            sample={"id": f"{int(time.time()*1000)}_{safe_name(label)}", "label":label,"key":key,"filename":fn,"timestamp":time.time(),"confidence":conf}
            self.samples.append(sample); self.save()
        self.update(last_message=f"Saved object sample for {label}")
        return True, f"Saved object sample for {label}"

    def should_acknowledge(self, label):
        key=normalise_key(pretty_label(label)); now=time.time()
        if not key or key in config.OBJECT_SKIP_LABELS: return False
        last=self.last_ack.get(key, 0)
        if now - last < config.OBJECT_GREETING_COOLDOWN_SECONDS: return False
        self.last_ack[key]=now
        return True

    def handle_detections(self, detections, frame_width, frame=None):
        best={}; last_seen=[]
        for det in detections or []:
            label=pretty_label(det.get("label","")); key=normalise_key(label)
            if not key or key in config.OBJECT_SKIP_LABELS: continue
            try: conf=float(det.get("confidence",0))
            except Exception: conf=0
            if conf < config.OBJECT_MIN_CONFIDENCE: continue
            if key not in best or conf > best[key][0]: best[key]=(conf,label,det)
        for conf,label,det in sorted(best.values(), reverse=True, key=lambda x:x[0]):
            last_seen.append({"label":label,"confidence":int(conf),"xmin":det.get("xmin"),"ymin":det.get("ymin"),"xmax":det.get("xmax"),"ymax":det.get("ymax")})
            if self.should_acknowledge(label):
                self.remember(label, det, frame)
                if self.speech: self.speech.say_async(config.OBJECT_GREETING_TEMPLATE.format(name=label), allow_beep_fallback=False)
                self.center_on_object(label, det, frame_width)
        self.update(status="enabled", last_seen=last_seen)

    def center_on_object(self, label, det, frame_width):
        if not self.motors or not config.TARGET_CENTERING_ENABLED: return
        try: center_norm=(float(det.get("xmin",0))+float(det.get("xmax",0)))/2.0
        except Exception: return
        offset=center_norm-0.5
        if abs(offset)<=config.TARGET_CENTERING_DEADBAND: return
        now=time.time()
        if now-self.last_center_turn<config.TARGET_CENTERING_COOLDOWN_SECONDS: return
        self.last_center_turn=now
        direction="right" if offset>0 else "left"
        self.motors.execute_move_async(direction, duration=config.TARGET_CENTERING_TURN_DURATION, acceleration=0.0)

    def remove_sample(self, sample_index=None, sample_id=None):
        with self.lock:
            idx=None
            if sample_id:
                for i,s in enumerate(self.samples):
                    if str(s.get("id"))==str(sample_id): idx=i; break
            if idx is None:
                try: idx=int(sample_index)
                except Exception: return False,"Object sample not found"
            if idx<0 or idx>=len(self.samples): return False,"Object sample not found"
            sample=self.samples.pop(idx)
            fn=os.path.basename(sample.get("filename") or "")
            if fn:
                try: os.remove(config.OBJECT_IMAGE_DIR/fn)
                except Exception: pass
            self.save()
        self.update(last_message=f"Removed one {sample.get('label','object')} sample")
        return True, "Removed object sample"

    def remove_label(self, label):
        key=normalise_key(pretty_label(label))
        with self.lock:
            removed=[s for s in self.samples if s.get("key")==key]
            if not removed: return False,f"No known object named {label}"
            self.samples=[s for s in self.samples if s.get("key")!=key]
            for s in removed:
                fn=os.path.basename(s.get("filename") or "")
                if fn:
                    try: os.remove(config.OBJECT_IMAGE_DIR/fn)
                    except Exception: pass
            self.save()
        self.update(last_message=f"Removed {len(removed)} sample(s) for {label}")
        return True, f"Removed {len(removed)} sample(s) for {label}"
