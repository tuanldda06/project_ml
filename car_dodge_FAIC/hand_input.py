from __future__ import annotations
import time
import threading
import math
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve
import cv2
import mediapipe as mp
from objects import HandState
from settings import MODEL_URL

class HandTracker:
    def __init__(self, cam_w=320, cam_h=240, model_dir="models"):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.state = HandState()
        self.x_hist = deque(maxlen=6) 
        self.last_known_pos = None  # THÊM: Lưu vị trí (x, y) cuối cùng của tay người chơi
        
        self.model_path = self._ensure_model(model_dir)
        self.landmarker = self._create_landmarker()
        self.latest_frame = None

        self.lock = threading.Lock()
        self.is_running = True
        self.last_timestamp_ms = 0
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _ensure_model(self, model_dir: str) -> str:
        model_path = Path(model_dir) / "hand_landmarker.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            urlretrieve(MODEL_URL, model_path)
        return str(model_path)

    def _create_landmarker(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,  # THAY ĐỔI: Cho phép nhận 2 tay để lọc
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _process_loop(self):
        while self.is_running:
            ok, frame = self.cap.read()
            if not ok: continue
            
            frame = cv2.flip(frame, 1)
            annotated = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            ts = int(time.time() * 1000)
            if ts <= self.last_timestamp_ms: ts = self.last_timestamp_ms + 1
            self.last_timestamp_ms = ts

            result = self.landmarker.detect_for_video(mp_image, ts)
            found, sx = False, self.state.x_norm
            h, w = annotated.shape[:2]
            
            cv2.rectangle(annotated, (int(0.2*w), 0), (int(0.8*w), h), (255, 255, 255), 1)

            if result.hand_landmarks:
                best_hand = None
                
                # THÊM LOGIC: Lọc tay chính chủ
                if self.last_known_pos is None:
                    best_hand = result.hand_landmarks[0]
                else:
                    min_dist = float('inf')
                    for hand in result.hand_landmarks:
                        # Tính khoảng cách từ tay này đến vị trí tay cũ
                        dist = math.hypot(hand[9].x - self.last_known_pos[0], hand[9].y - self.last_known_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_hand = hand
                
                # Cập nhật tọa độ tay chính
                self.last_known_pos = (best_hand[9].x, best_hand[9].y)
                raw_x = float(best_hand[9].x) 
                
                self.x_hist.append(raw_x)
                sx = sum(self.x_hist) / len(self.x_hist)
                found = True
                
                for lm in best_hand:
                    cv2.circle(annotated, (int(lm.x*w), int(lm.y*h)), 3, (120, 255, 120), -1)
                cv2.line(annotated, (int(sx*w), 0), (int(sx*w), h), (0, 255, 255), 2)
            else:
                self.x_hist.clear() 
                self.last_known_pos = None # Reset vị trí cũ khi mất dấu
                cv2.putText(annotated, "No hand", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

            with self.lock:
                self.state.found, self.state.x_norm, self.latest_frame = found, sx, annotated

    def read(self):
        with self.lock:
            return HandState(found=self.state.found, x_norm=self.state.x_norm), self.latest_frame

    def close(self):
        self.is_running = False
        self.thread.join()
        self.landmarker.close()
        self.cap.release()