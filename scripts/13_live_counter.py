#!/usr/bin/env python3
"""
Live Early Prediction Counter - COMPLETE FIXED VERSION
With lowered thresholds for jab/cross detection

Usage:
    python 13_live_counter.py --debug
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import time
import threading
import socket
import json
from queue import Queue, Empty
from collections import deque
import numpy as np
import cv2
import pickle

import tensorflow as tf
from ultralytics import YOLO

MODEL_PATH = "classifier/boxing_oracle.h5"
LABEL_PATH = "classifier/labels_oracle.pkl"
NORM_PATH = MODEL_PATH.replace('.h5', '.norm.npz')

YOLO_MODEL = "yolo11n-pose.pt"

CAM_SRC = 0
FRAME_W = 640
FRAME_H = 480
TARGET_FPS = 60

EARLY_WINDOW = 5
CLASS_WINDOW = 8
FEAT_DIM = 108

# LOWERED THRESHOLDS FOR JAB/CROSS DETECTION
EARLY_CONF_THRESHOLD = 0.40   # Down from 0.55
CLASS_CONF_THRESHOLD = 0.65   # Down from 0.75
ENERGY_THRESHOLD = 0.015      # Slightly higher to avoid standing triggers
ENGAGED_SECONDS = 0.05
PERSIST_SECONDS = 0.10
KP_CONF_THRESH = 0.20

KPT_SMOOTH_ALPHA = 0.70
PROFILE_ALPHA = 0.92
HUD_FADE_SPEED = 10.0

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

COUNTER_MAP = {
    "jab": "SLIP RIGHT",
    "cross": "SLIP LEFT",
    "hook": "DUCK / ROLL",
    "uppercut": "PULL BACK",
    "overhand": "STEP BACK + JAM"
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    ang = np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return np.degrees(ang)

class CamReader(threading.Thread):
    def __init__(self, src=0, width=640, height=480, fps=60, queue_size=2):
        super().__init__(daemon=True)
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.q = Queue(maxsize=queue_size)
        self.stopped = False
    
    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.q.full():
                try: self.q.get_nowait()
                except Empty: pass
            self.q.put(frame)
    
    def read(self, timeout=0.05):
        try: return self.q.get(timeout=timeout)
        except Empty: return None
    
    def stop(self):
        self.stopped = True
        try: self.cap.release()
        except Exception: pass

class DualOracleWorker(threading.Thread):
    def __init__(self, early_model, class_model, normalizer):
        super().__init__(daemon=True)
        self.early_model = early_model
        self.class_model = class_model
        self.mean, self.std = normalizer
        self.in_q = Queue(maxsize=4)
        self.out_q = Queue(maxsize=8)
        self.stop_flag = False
    
    def submit(self, sequence, stage):
        try:
            self.in_q.put_nowait((list(sequence), stage, time.time()))
        except Exception:
            pass
    
    def try_get(self):
        try:
            return self.out_q.get_nowait()
        except Empty:
            return None
    
    def run(self):
        while not self.stop_flag:
            try:
                seq_list, stage, ts = self.in_q.get(timeout=0.2)
            except Empty:
                continue
            
            try:
                seq_np = np.array(seq_list, dtype=np.float32)
                seq_np = (seq_np - self.mean) / self.std
                seq_np = np.expand_dims(seq_np, axis=0)
                
                if stage == 'early':
                    pred = self.early_model.predict(seq_np, verbose=0)[0]
                else:
                    pred = self.class_model.predict(seq_np, verbose=0)[0]
                
                try:
                    self.out_q.put_nowait((pred, stage, ts))
                except Exception:
                    pass
            except Exception as e:
                pass
    
    def stop(self):
        self.stop_flag = True

_udp_sock = None
def init_udp_sender():
    global _udp_sock
    try:
        _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _udp_sock.setblocking(False)
    except Exception:
        _udp_sock = None

def send_prediction_udp(label, conf, counter, stage):
    if _udp_sock is None:
        return
    msg = {
        "ts": time.time(),
        "label": str(label),
        "conf": float(conf),
        "counter": str(counter),
        "stage": str(stage)
    }
    try:
        _udp_sock.sendto(json.dumps(msg).encode("utf-8"), (UDP_IP, UDP_PORT))
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default=MODEL_PATH)
    parser.add_argument("--labels", "-l", default=LABEL_PATH)
    parser.add_argument("--norm", default=NORM_PATH)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()
    
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels not found: {args.labels}")
    
    with open(args.labels, "rb") as f:
        CLASSES = pickle.load(f)
    
    early_path = args.model.replace('.h5', '_early.h5')
    class_path = args.model.replace('.h5', '_classifier.h5')
    
    if not os.path.exists(early_path):
        raise FileNotFoundError(f"Early model not found: {early_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Classifier not found: {class_path}")
    
    print(f"Loading Early Warning: {early_path}")
    early_model = tf.keras.models.load_model(early_path, compile=False)
    
    print(f"Loading Classifier: {class_path}")
    class_model = tf.keras.models.load_model(class_path, compile=False)
    
    if os.path.exists(args.norm):
        d = np.load(args.norm)
        mean = d["mean"].astype(np.float32).reshape(1, -1)
        std = d["std"].astype(np.float32).reshape(1, -1)
    else:
        raise FileNotFoundError(f"Normalization file not found: {args.norm}")
    
    pose_model = YOLO(YOLO_MODEL)
    
    cam = CamReader(src=CAM_SRC, width=FRAME_W, height=FRAME_H, fps=TARGET_FPS)
    cam.start()
    
    init_udp_sender()
    
    oracle_worker = DualOracleWorker(early_model, class_model, (mean, std))
    oracle_worker.start()
    
    sequence = deque(maxlen=CLASS_WINDOW)
    filtered_kpts = None
    prev_kpts = None
    prev_vel = None
    energy_ema = None
    
    engaged_start = None
    
    early_warning = None
    confirmed_pred = None
    
    hud_alpha = 0.0
    counter_msg = ""
    detected_move = ""
    warning_level = ""
    fade_target = 0.0
    display_end_time = 0.0
    
    frame_idx = 0
    fps_ema = 0.0
    t_cam_ema = t_pose_ema = t_oracle_ema = t_draw_ema = 0.0
    
    print("\n" + "="*60)
    print("🥊 EARLY PREDICTION SYSTEM ACTIVE")
    print("="*60)
    print(f"Stage 1 (Early Warning): {EARLY_WINDOW} frames ({EARLY_WINDOW/60*1000:.0f}ms)")
    print(f"Stage 2 (Classifier):    {CLASS_WINDOW} frames ({CLASS_WINDOW/60*1000:.0f}ms)")
    print(f"Classes: {CLASSES}")
    print(f"Thresholds: Early={EARLY_CONF_THRESHOLD:.0%}, Classify={CLASS_CONF_THRESHOLD:.0%}")
    print("Press 'd' for debug, ESC to quit\n")
    
    try:
        while True:
            t0_frame = time.perf_counter()
            frame = cam.read()
            t_after_cam = time.perf_counter()
            t_cam = t_after_cam - t0_frame
            t_cam_ema = t_cam if t_cam_ema == 0 else PROFILE_ALPHA * t_cam_ema + (1 - PROFILE_ALPHA) * t_cam
            
            if frame is None:
                time.sleep(0.002)
                continue
            
            frame_idx += 1
            h, w = frame.shape[:2]
            
            t0_pose = time.perf_counter()
            try:
                results = pose_model.predict(frame, verbose=False, half=True, device=0)
            except Exception:
                results = pose_model.predict(frame, verbose=False)
            
            t_after_pose = time.perf_counter()
            t_pose = t_after_pose - t0_pose
            t_pose_ema = t_pose if t_pose_ema == 0 else PROFILE_ALPHA * t_pose_ema + (1 - PROFILE_ALPHA) * t_pose
            
            kpts_valid = False
            kp = None
            mean_kp_conf = 1.0
            visible_kpts = 0
            
            if results and len(results) > 0:
                try:
                    if hasattr(results[0], "keypoints") and len(results[0].keypoints.xy) > 0:
                        kp = results[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
                        kpts_valid = True
                        try:
                            confs = results[0].keypoints.conf[0].cpu().numpy()
                            mean_kp_conf = float(np.mean(confs))
                            visible_kpts = int(np.sum(confs > KP_CONF_THRESH))
                        except Exception:
                            mean_kp_conf = 1.0
                            visible_kpts = 0
                except Exception:
                    kpts_valid = False
            
            if kpts_valid and mean_kp_conf < KP_CONF_THRESH:
                kpts_valid = False
            
            state = "SCANNING"
            
            if kpts_valid:
                kpts_norm = (kp / np.array([w, h], dtype=np.float32)).flatten()
                
                if filtered_kpts is None:
                    filtered_kpts = kpts_norm.copy()
                    prev_kpts = kpts_norm.copy()
                    prev_vel = np.zeros_like(kpts_norm)
                
                filtered_kpts = KPT_SMOOTH_ALPHA * filtered_kpts + (1.0 - KPT_SMOOTH_ALPHA) * kpts_norm
                vel = filtered_kpts - prev_kpts
                acc = vel - prev_vel
                
                wrist_vec = acc[18:22]
                wrist_acc_l2 = float(np.linalg.norm(wrist_vec))
                
                if energy_ema is None:
                    energy_ema = wrist_acc_l2
                else:
                    energy_ema = 0.6 * energy_ema + 0.4 * wrist_acc_l2
                
                angles = np.zeros(5, dtype=np.float32)
                if len(kp) >= 11:
                    try:
                        angles[0] = calculate_angle(kp[5], kp[7], kp[9])
                        angles[1] = calculate_angle(kp[6], kp[8], kp[10])
                    except Exception:
                        pass
                
                feats = np.concatenate([filtered_kpts, vel, acc, angles, np.array([energy_ema])])
                if feats.shape[0] < FEAT_DIM:
                    feats = np.concatenate([feats, np.zeros(FEAT_DIM - feats.shape[0])])
                
                sequence.append(feats.astype(np.float32))
                
                prev_kpts = filtered_kpts.copy()
                prev_vel = vel.copy()
                
                now_ts = time.time()
                if energy_ema > ENERGY_THRESHOLD:
                    if engaged_start is None:
                        engaged_start = now_ts
                    
                    if (now_ts - engaged_start) >= ENGAGED_SECONDS:
                        state = "ENGAGED"
                else:
                    engaged_start = None
                
                if state == "ENGAGED":
                    if len(sequence) >= EARLY_WINDOW and early_warning is None:
                        early_seq = list(sequence)[:EARLY_WINDOW]
                        oracle_worker.submit(early_seq, 'early')
                    
                    if len(sequence) >= CLASS_WINDOW and confirmed_pred is None:
                        class_seq = list(sequence)[:CLASS_WINDOW]
                        oracle_worker.submit(class_seq, 'classify')
                
                result = oracle_worker.try_get()
                if result is not None:
                    pred, stage, pred_ts = result
                    idx = int(np.argmax(pred))
                    label = CLASSES[idx]
                    conf = float(pred[idx])
                    
                    if args.debug:
                        pred_dict = {cls: round(float(pred[i]), 2) for i, cls in enumerate(CLASSES)}
                        print(f"[{stage.upper()}] {label}: {conf:.2%} | {pred_dict}")
                    
                    if stage == 'early':
                        if conf >= EARLY_CONF_THRESHOLD:
                            early_warning = {'label': label, 'conf': conf, 'ts': now_ts}
                            counter_msg = COUNTER_MAP.get(label, "DEFEND")
                            detected_move = label.upper()
                            warning_level = "EARLY"
                            display_end_time = now_ts + 1.2
                            
                            send_prediction_udp(label, conf, counter_msg, 'early')
                            
                            if args.debug:
                                print(f"⚡ EARLY WARNING: {label.upper()} ({conf:.0%})")
                    
                    elif stage == 'classify':
                        if conf >= CLASS_CONF_THRESHOLD:
                            confirmed_pred = {'label': label, 'conf': conf, 'ts': now_ts}
                            
                            if early_warning is None or early_warning['label'] != label:
                                counter_msg = COUNTER_MAP.get(label, "DEFEND")
                                detected_move = label.upper()
                            
                            warning_level = "CONFIRMED"
                            display_end_time = now_ts + 1.0
                            
                            send_prediction_udp(label, conf, counter_msg, 'confirmed')
                            
                            if args.debug:
                                print(f"✅ CONFIRMED: {label.upper()} ({conf:.0%})")
                            
                            sequence.clear()
                            early_warning = None
                            confirmed_pred = None
                
                fade_target = 1.0 if counter_msg else 0.0
            
            else:
                fade_target = 0.0
            
            nowf = time.time()
            if nowf > display_end_time:
                counter_msg = ""
                early_warning = None
                confirmed_pred = None
            
            alpha_delta = HUD_FADE_SPEED * (nowf - t0_frame)
            if hud_alpha < fade_target:
                hud_alpha = min(fade_target, hud_alpha + alpha_delta)
            else:
                hud_alpha = max(fade_target, hud_alpha - alpha_delta)
            
            t0_draw = time.perf_counter()
            
            if energy_ema is not None:
                bar_len = int(np.clip((energy_ema / max(1e-9, ENERGY_THRESHOLD)) * (w * 0.15), 0, w * 0.6))
                bar_color = (0, 200, 0) if state == "SCANNING" else (0, 0, 255)
                cv2.rectangle(frame, (10, h-18), (10 + bar_len, h-6), bar_color, -1)
            
            if hud_alpha > 0.01:
                panel_h = 100
                overlay = frame.copy()
                
                if warning_level == "EARLY":
                    panel_color = (0, 140, 255)
                elif warning_level == "CONFIRMED":
                    panel_color = (0, 0, 200)
                else:
                    panel_color = (20, 20, 20)
                
                cv2.rectangle(overlay, (8, 8), (w-8, 8+panel_h), panel_color, -1)
                cv2.addWeighted(overlay, hud_alpha * 0.9, frame, 1 - hud_alpha * 0.9, 0, frame)
                
                if counter_msg:
                    level_text = f"[{warning_level}]" if warning_level else ""
                    cv2.putText(frame, level_text, (30, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(frame, f"INCOMING: {detected_move}", (30, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, counter_msg, (30, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"READY... {fps_ema:.1f} FPS", (30, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"Seq: {len(sequence)}/{CLASS_WINDOW} | State: {state}", 
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            
            if energy_ema is not None:
                e_color = (0, 0, 255) if energy_ema > ENERGY_THRESHOLD else (200, 200, 0)
                cv2.putText(frame, f"E:{energy_ema:.4f}", (w-200, h-22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, e_color, 1, cv2.LINE_AA)
            
            t_draw_end = time.perf_counter()
            t_draw = t_draw_end - t0_draw
            t_draw_ema = t_draw if t_draw_ema == 0 else PROFILE_ALPHA * t_draw_ema + (1 - PROFILE_ALPHA) * t_draw
            
            t_frame = time.perf_counter() - t0_frame
            fps_inst = 1.0 / max(1e-6, t_frame)
            fps_ema = fps_inst if fps_ema == 0 else PROFILE_ALPHA * fps_ema + (1 - PROFILE_ALPHA) * fps_inst
            
            profiler = f"cam:{t_cam_ema*1000:.0f}ms pose:{t_pose_ema*1000:.0f}ms oracle:{t_oracle_ema*1000:.0f}ms"
            cv2.putText(frame, profiler, (12, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"FPS:{fps_ema:.1f}", (w-120, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
            
            cv2.imshow("BOXING PRO - EARLY PREDICTION", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('d'):
                args.debug = not args.debug
                print(f"[DEBUG] {'ON' if args.debug else 'OFF'}")
    
    finally:
        cam.stop()
        oracle_worker.stop()
        if _udp_sock:
            try: _udp_sock.close()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()