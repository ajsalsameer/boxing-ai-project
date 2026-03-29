#!/usr/bin/env python3
"""
BOXING AI - ACTUALLY FIXED VERSION
===================================
REAL FIXES:
- Stricter energy threshold (stops false detections)
- Longer cooldown (prevents jab->jab->jab spam)
- Require movement CHANGE (not just any movement)
- Better classification confidence

Usage:
    python boxing_ai_ACTUALLY_FIXED.py
"""

import os, sys, time, json, pickle, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2, numpy as np, torch
from collections import deque
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

_YOLO = None

def _get_yolo_cls():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO

from utils import (
    FEAT_DIM, extract_upper_body, compute_features, draw_skeleton,
    PUNCH_COLOURS, DEFAULT_COLOUR
)

# CONFIG
YOLO_PATH = "yolo11n-pose.pt"
BANK_PATH = "data/feature_bank.pkl"
COMBOS_PATH = "data/combo_sequences.pkl"

# FIXED: MUCH STRICTER THRESHOLDS
ENERGY_THRESHOLD = 0.030      # RAISED from 0.010 - Only real punches
PUNCH_COOLDOWN = 45           # RAISED from 15 - 1.5 seconds between detections
CLASS_WINDOW = 10             # RAISED from 6 - More stable averaging
MIN_CONFIDENCE = 0.60         # RAISED from 0.45 - Higher confidence needed
TEMPORAL_WINDOW = 5           # RAISED from 3 - Better smoothing
ENERGY_CHANGE_THRESHOLD = 0.015  # NEW: Require energy to CHANGE (not just be high)

USE_FP16 = True
YOLO_IMG_SIZE = 416

def classify_punch(feat, feature_bank, min_confidence=MIN_CONFIDENCE):
    """Classify with STRICT confidence threshold"""
    if not feature_bank or feat is None:
        return "unknown", 0.0
    
    best_label = "unknown"
    best_similarity = 0.0
    
    for label, data in feature_bank.items():
        centroid = np.array(data["mean"], dtype=np.float32)
        std = np.array(data["std"], dtype=np.float32)
        
        # Mahalanobis distance
        diff = feat - centroid
        distance = np.sqrt(np.sum((diff / (std + 1e-6)) ** 2))
        similarity = np.exp(-distance / 10)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_label = label
    
    # STRICT: Reject if too low
    if best_similarity < min_confidence:
        return "unknown", best_similarity
    
    return best_label, best_similarity


def temporal_smooth(history, new_label, new_conf, window=TEMPORAL_WINDOW):
    """Smooth with confidence voting"""
    if new_label == "unknown":
        return new_label, history
    
    history = list(history) + [(new_label, new_conf)]
    history = history[-window:]
    
    votes = {}
    for lbl, conf in history:
        votes[lbl] = votes.get(lbl, 0) + conf
    
    smoothed = max(votes, key=votes.get)
    return smoothed, history


def predict_next_move(history, predictor):
    """Predict next move"""
    if not predictor or len(history) < 1:
        return None, 0.0
    
    for length in [3, 2, 1]:
        if len(history) >= length:
            pattern = tuple(history[-length:])
            
            if pattern in predictor:
                next_moves = predictor[pattern]
                total = sum(next_moves.values())
                best_move = max(next_moves, key=next_moves.get)
                confidence = next_moves[best_move] / total
                return best_move, confidence
    
    return None, 0.0


class BoxingAI:
    def __init__(self, root):
        self.root = root
        self.root.title("🥊 Boxing AI - FIXED")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0e0e15")
        
        self.mode = "MENU"
        self.running = False
        self.cap = None
        self.current_frame = None
        self.photo = None
        
        # Teaching
        self.teaching_label = None
        self.recording = False
        self.recorded_features = []
        self.recorded_combo = []
        
        # Game
        self.punch_history = []
        self.last_punch = None
        self.last_punch_time = 0.0
        self.last_punch_frame = -999
        self.frame_counter = 0
        self.prediction = None
        self.pred_confidence = 0.0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # YOLO
        self.prev_kpts = None
        self.prev_vel = None
        self.feat_seq = deque(maxlen=CLASS_WINDOW)
        self.smooth_history = []
        
        # NEW: Energy tracking for change detection
        self.energy_history = deque(maxlen=10)
        self.last_energy_spike = 0
        
        # Debug
        self.debug_text = "Ready"
        
        # Load
        print("="*70)
        print("  BOXING AI - LOADING (FIXED VERSION)")
        print("="*70)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.yolo_model = self.load_yolo()
        self.feature_bank = self.load_feature_bank()
        self.combo_sequences = self.load_combo_sequences()
        self.predictor = self.build_predictor()
        self.labels = ["jab", "cross", "hook", "uppercut"]
        
        print(f"\n✅ Device: {self.device}")
        print(f"✅ Thresholds: Energy={ENERGY_THRESHOLD}, Cooldown={PUNCH_COOLDOWN}, Confidence={MIN_CONFIDENCE}")
        print(f"✅ Feature bank: {len(self.feature_bank)} types")
        print(f"✅ Combos: {len(self.combo_sequences)} sequences")
        print("="*70 + "\n")
        
        self.build_menu()
        self.root.bind('<Escape>', lambda e: self.back_to_menu())
    
    def load_yolo(self):
        try:
            YOLO = _get_yolo_cls()
            model = YOLO(YOLO_PATH)
            print("✅ YOLO loaded")
            return model
        except Exception as e:
            print(f"❌ YOLO error: {e}")
            return None
    
    def load_feature_bank(self):
        if os.path.exists(BANK_PATH):
            try:
                with open(BANK_PATH, "rb") as f:
                    bank = pickle.load(f)
                return {k: v for k, v in bank.items() if k != "idle"}
            except:
                return {}
        return {}
    
    def load_combo_sequences(self):
        if os.path.exists(COMBOS_PATH):
            try:
                with open(COMBOS_PATH, "rb") as f:
                    data = pickle.load(f)
                
                # Convert old format if needed
                if data and isinstance(data, list):
                    if isinstance(data[0], dict):
                        combos = []
                        for item in data:
                            if "history" in item and "next" in item:
                                combo = item["history"] + [item["next"]]
                                if combo not in combos:
                                    combos.append(combo)
                        return combos
                    else:
                        return data
                return []
            except:
                return []
        return []
    
    def build_predictor(self):
        predictor = {}
        
        for combo in self.combo_sequences:
            if isinstance(combo, dict) or not isinstance(combo, list) or len(combo) < 2:
                continue
            
            for i in range(len(combo) - 1):
                pattern1 = (combo[i],)
                next_move = combo[i + 1]
                
                if pattern1 not in predictor:
                    predictor[pattern1] = {}
                predictor[pattern1][next_move] = predictor[pattern1].get(next_move, 0) + 1
                
                if i > 0:
                    pattern2 = (combo[i-1], combo[i])
                    if pattern2 not in predictor:
                        predictor[pattern2] = {}
                    predictor[pattern2][next_move] = predictor[pattern2].get(next_move, 0) + 1
                
                if i > 1:
                    pattern3 = (combo[i-2], combo[i-1], combo[i])
                    if pattern3 not in predictor:
                        predictor[pattern3] = {}
                    predictor[pattern3][next_move] = predictor[pattern3].get(next_move, 0) + 1
        
        return predictor
    
    def save_data(self):
        try:
            os.makedirs("data", exist_ok=True)
            
            with open(BANK_PATH, "wb") as f:
                pickle.dump(self.feature_bank, f)
            
            with open(COMBOS_PATH, "wb") as f:
                pickle.dump(self.combo_sequences, f)
            
            print("✅ Data saved")
            return True
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def build_menu(self):
        self.clear_window()
        self.mode = "MENU"
        
        tk.Label(self.root, text="🥊 BOXING AI (FIXED)", 
                font=("Arial", 36, "bold"), bg="#0e0e15", fg="#c8102e").pack(pady=40)
        
        tk.Label(self.root, text="Stricter Detection - No False Triggers",
                font=("Arial", 14), bg="#0e0e15", fg="#888").pack(pady=10)
        
        btn_frame = tk.Frame(self.root, bg="#0e0e15")
        btn_frame.pack(expand=True)
        
        # STEP 1
        tk.Button(btn_frame, 
                 text="📚 STEP 1: TEACH PUNCHES\n\nRecord 10-15 examples per punch\nThrow HARD for best results",
                 font=("Arial", 13, "bold"), bg="#4caf50", fg="white", relief=tk.FLAT,
                 width=45, height=6, cursor="hand2",
                 command=self.start_teach_punches).pack(pady=10)
        
        status1 = f"✅ {len(self.feature_bank)}/4 types" if self.feature_bank else "❌ Not started"
        tk.Label(btn_frame, text=status1, font=("Arial", 10), bg="#0e0e15", 
                fg="#4caf50" if self.feature_bank else "#ff9800").pack()
        
        # STEP 2
        step2_enabled = len(self.feature_bank) >= 4
        tk.Button(btn_frame,
                 text="🎯 STEP 2: TEACH COMBOS\n\nRecord 5-10 different combos\n(3-5 punches each)",
                 font=("Arial", 13, "bold"),
                 bg="#2196F3" if step2_enabled else "#555", fg="white", relief=tk.FLAT,
                 width=45, height=6, cursor="hand2" if step2_enabled else "arrow",
                 command=self.start_teach_combos if step2_enabled else None,
                 state=tk.NORMAL if step2_enabled else tk.DISABLED).pack(pady=10)
        
        status2 = f"✅ {len(self.combo_sequences)} combos" if self.combo_sequences else "❌ Not started"
        tk.Label(btn_frame, text=status2, font=("Arial", 10), bg="#0e0e15",
                fg="#4caf50" if self.combo_sequences else "#ff9800").pack()
        
        # STEP 3
        step3_enabled = len(self.feature_bank) >= 4 and len(self.combo_sequences) >= 3
        tk.Button(btn_frame,
                 text="🎮 STEP 3: PLAY GAME\n\nLive prediction!",
                 font=("Arial", 13, "bold"),
                 bg="#9c27b0" if step3_enabled else "#555", fg="white", relief=tk.FLAT,
                 width=45, height=6, cursor="hand2" if step3_enabled else "arrow",
                 command=self.start_game if step3_enabled else None,
                 state=tk.NORMAL if step3_enabled else tk.DISABLED).pack(pady=10)
        
        status3 = "✅ Ready!" if step3_enabled else "❌ Complete Steps 1 & 2"
        tk.Label(btn_frame, text=status3, font=("Arial", 10), bg="#0e0e15",
                fg="#4caf50" if step3_enabled else "#e53935").pack()
        
        tk.Button(btn_frame, text="❌ QUIT", font=("Arial", 11),
                 bg="#e53935", fg="white", width=20, relief=tk.FLAT,
                 command=self.root.destroy).pack(pady=25)
    
    def build_teach_punches_ui(self):
        self.clear_window()
        
        header = tk.Frame(self.root, bg="#4caf50", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="📚 STEP 1: TEACH PUNCHES", 
                font=("Arial", 17, "bold"), bg="#4caf50", fg="white").pack(pady=15)
        
        main = tk.Frame(self.root, bg="#0e0e15")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        left = tk.Frame(main, bg="#0e0e15")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack()
        
        tk.Label(left, text="⚠️ IMPORTANT: Throw HARD punches!\n\n1. Select punch type\n2. Click RECORD\n3. Throw 10-15 HARD examples\n4. Click STOP", 
                font=("Arial", 11), bg="#0e0e15", fg="#ff9800", justify=tk.LEFT).pack(pady=10)
        
        right = tk.Frame(main, bg="#0e0e15", width=350)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        
        tk.Label(right, text="SELECT PUNCH:", font=("Arial", 10, "bold"), 
                bg="#0e0e15", fg="white").pack(pady=10)
        
        self.punch_var = tk.StringVar(value="jab")
        for label in self.labels:
            tk.Radiobutton(right, text=label.upper(), variable=self.punch_var,
                          value=label, font=("Arial", 11), bg="#0e0e15", fg="white",
                          selectcolor="#1a1a2e").pack(anchor="w", padx=20, pady=3)
        
        self.record_btn = tk.Button(right, text="🔴 START RECORDING", 
                                    font=("Arial", 11, "bold"), bg="#e53935", fg="white",
                                    height=2, relief=tk.FLAT, command=self.toggle_recording)
        self.record_btn.pack(fill=tk.X, pady=15)
        
        self.teach_status = tk.Label(right, text="Ready", font=("Arial", 9), 
                                     bg="#0e0e15", fg="#888")
        self.teach_status.pack(pady=10)
        
        bank_frame = tk.Frame(right, bg="#1a1a2e", bd=2, relief=tk.RIDGE)
        bank_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(bank_frame, text="FEATURE BANK", font=("Arial", 9, "bold"),
                bg="#1a1a2e", fg="#4caf50").pack(pady=5)
        
        self.bank_label = tk.Label(bank_frame, text=self.get_bank_status(),
                                   font=("Arial", 8), bg="#1a1a2e", fg="white", justify=tk.LEFT)
        self.bank_label.pack(pady=10)
        
        tk.Button(right, text="✅ FINISH & SAVE", font=("Arial", 10, "bold"),
                 bg="#4caf50", fg="white", relief=tk.FLAT,
                 command=self.finish_teaching).pack(fill=tk.X, pady=10)
        
        tk.Button(right, text="← BACK", font=("Arial", 9),
                 bg="#666", fg="white", relief=tk.FLAT,
                 command=self.back_to_menu).pack(fill=tk.X)
    
    def build_teach_combos_ui(self):
        self.clear_window()
        
        header = tk.Frame(self.root, bg="#2196F3", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="🎯 STEP 2: TEACH COMBOS", 
                font=("Arial", 17, "bold"), bg="#2196F3", fg="white").pack(pady=15)
        
        main = tk.Frame(self.root, bg="#0e0e15")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        left = tk.Frame(main, bg="#0e0e15")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack()
        
        tk.Label(left, text="⚠️ IMPORTANT: Throw HARD combos!\n\n1. START RECORDING\n2. Throw combo (3-5 HARD punches)\n3. STOP\n4. Repeat 5-10 times",
                font=("Arial", 11), bg="#0e0e15", fg="#ff9800", justify=tk.LEFT).pack(pady=10)
        
        right = tk.Frame(main, bg="#0e0e15", width=350)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        
        self.combo_record_btn = tk.Button(right, text="🔴 START RECORDING COMBO", 
                                          font=("Arial", 11, "bold"), bg="#e53935", fg="white",
                                          height=2, relief=tk.FLAT, 
                                          command=self.toggle_combo_recording)
        self.combo_record_btn.pack(fill=tk.X, pady=15)
        
        self.combo_status = tk.Label(right, text="Ready", font=("Arial", 9), 
                                     bg="#0e0e15", fg="#888")
        self.combo_status.pack(pady=10)
        
        combo_frame = tk.Frame(right, bg="#1a1a2e", bd=2, relief=tk.RIDGE)
        combo_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(combo_frame, text="CURRENT COMBO", font=("Arial", 9, "bold"),
                bg="#1a1a2e", fg="#2196F3").pack(pady=5)
        
        self.current_combo_label = tk.Label(combo_frame, text="—", 
                                            font=("Arial", 10, "bold"), bg="#1a1a2e", fg="white")
        self.current_combo_label.pack(pady=10)
        
        recorded_frame = tk.Frame(right, bg="#1a1a2e", bd=2, relief=tk.RIDGE)
        recorded_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(recorded_frame, text="RECORDED COMBOS", font=("Arial", 9, "bold"),
                bg="#1a1a2e", fg="#2196F3").pack(pady=5)
        
        self.recorded_combos_label = tk.Label(recorded_frame, text=f"{len(self.combo_sequences)} combos", 
                                              font=("Arial", 9), bg="#1a1a2e", fg="white")
        self.recorded_combos_label.pack(pady=10)
        
        tk.Button(right, text="✅ FINISH & BUILD PREDICTOR", 
                 font=("Arial", 10, "bold"), bg="#4caf50", fg="white",
                 relief=tk.FLAT, command=self.finish_combo_teaching).pack(fill=tk.X, pady=10)
        
        tk.Button(right, text="← BACK", font=("Arial", 9),
                 bg="#666", fg="white", relief=tk.FLAT,
                 command=self.back_to_menu).pack(fill=tk.X)
    
    def build_game_ui(self):
        self.clear_window()
        
        header = tk.Frame(self.root, bg="#9c27b0", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="🎮 LIVE GAME - NEXT-MOVE PREDICTION", 
                font=("Arial", 17, "bold"), bg="#9c27b0", fg="white").pack(pady=15)
        
        main = tk.Frame(self.root, bg="#0e0e15")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        left = tk.Frame(main, bg="#0e0e15")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack()
        
        debug_frame = tk.Frame(left, bg="#1a1a2e", bd=2, relief=tk.RIDGE)
        debug_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(debug_frame, text="🔍 DEBUG", font=("Arial", 8, "bold"),
                bg="#1a1a2e", fg="#9c27b0").pack()
        
        self.debug_label = tk.Label(debug_frame, text="", font=("Courier", 8), 
                                    bg="#1a1a2e", fg="#4caf50", justify=tk.LEFT)
        self.debug_label.pack(padx=10, pady=5)
        
        right = tk.Frame(main, bg="#0e0e15", width=380)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        
        pred_frame = tk.Frame(right, bg="#9c27b0", bd=3, relief=tk.RIDGE)
        pred_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(pred_frame, text="🤖 AI PREDICTS NEXT", 
                font=("Arial", 11, "bold"), bg="#9c27b0", fg="white").pack(pady=(10, 5))
        
        self.pred_label = tk.Label(pred_frame, text="…", font=("Arial", 32, "bold"), 
                                   bg="#9c27b0", fg="white")
        self.pred_label.pack(pady=10)
        
        self.pred_conf_label = tk.Label(pred_frame, text="0%", font=("Arial", 10), 
                                        bg="#9c27b0", fg="white")
        self.pred_conf_label.pack(pady=(0, 10))
        
        score_frame = tk.Frame(right, bg="#1a1a2e", bd=2, relief=tk.RIDGE)
        score_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(score_frame, text="SCORE", font=("Arial", 9, "bold"),
                bg="#1a1a2e", fg="#666").pack(pady=5)
        
        self.score_label = tk.Label(score_frame, text="0/0 (0%)", font=("Arial", 14, "bold"),
                                    bg="#1a1a2e", fg="white")
        self.score_label.pack(pady=10)
        
        trail_frame = tk.Frame(right, bg="#1a1a2e", bd=2, relief=tk.RIDGE)
        trail_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(trail_frame, text="COMBO TRAIL", font=("Arial", 9, "bold"),
                bg="#1a1a2e", fg="#666").pack(pady=5)
        
        self.trail_label = tk.Label(trail_frame, text="—", font=("Arial", 10, "bold"),
                                    bg="#1a1a2e", fg="white", wraplength=360)
        self.trail_label.pack(pady=5)
        
        self.flash_label = tk.Label(right, text="", font=("Arial", 11, "bold"), 
                                    bg="#0e0e15", fg="#4caf50", wraplength=360)
        self.flash_label.pack(pady=10)
        
        self.game_btn = tk.Button(right, text="▶️ START GAME", 
                                  font=("Arial", 11, "bold"), bg="#4caf50", fg="white",
                                  height=2, relief=tk.FLAT, command=self.toggle_game)
        self.game_btn.pack(fill=tk.X, pady=15)
        
        tk.Button(right, text="🔄 RESET SCORE", font=("Arial", 9),
                 bg="#ff9800", fg="white", relief=tk.FLAT,
                 command=self.reset_score).pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(right, text="← BACK", font=("Arial", 9),
                 bg="#666", fg="white", relief=tk.FLAT,
                 command=self.back_to_menu).pack(fill=tk.X)
    
    def start_teach_punches(self):
        self.mode = "TEACH_PUNCHES"
        self.build_teach_punches_ui()
        self.start_camera_thread()
    
    def start_teach_combos(self):
        self.mode = "TEACH_COMBOS"
        self.build_teach_combos_ui()
        self.start_camera_thread()
    
    def start_game(self):
        self.mode = "GAME"
        self.build_game_ui()
    
    def toggle_recording(self):
        if not self.running:
            messagebox.showerror("Error", "Start camera first!")
            return
        
        if not self.recording:
            self.recording = True
            self.teaching_label = self.punch_var.get()
            self.recorded_features = []
            self.record_btn.config(text="⏹️ STOP RECORDING", bg="#4caf50")
            self.teach_status.config(text=f"Recording {self.teaching_label.upper()}... Throw HARD!")
        else:
            self.recording = False
            
            if len(self.recorded_features) < 5:
                messagebox.showwarning("Too Few", f"Only {len(self.recorded_features)} recorded. Need 5+!\n\nThrow HARDER punches!")
                self.record_btn.config(text="🔴 START RECORDING", bg="#e53935")
                self.teach_status.config(text="Ready")
                return
            
            self.add_to_bank(self.teaching_label, self.recorded_features)
            self.record_btn.config(text="🔴 START RECORDING", bg="#e53935")
            self.teach_status.config(text=f"✅ Added {len(self.recorded_features)} examples!")
            self.bank_label.config(text=self.get_bank_status())
    
    def toggle_combo_recording(self):
        if not self.running:
            messagebox.showerror("Error", "Start camera first!")
            return
        
        if not self.recording:
            self.recording = True
            self.recorded_combo = []
            self.combo_record_btn.config(text="⏹️ STOP RECORDING", bg="#4caf50")
            self.combo_status.config(text="Recording combo... Throw HARD!")
            self.current_combo_label.config(text="—")
        else:
            self.recording = False
            
            if len(self.recorded_combo) < 2:
                messagebox.showwarning("Too Short", f"Only {len(self.recorded_combo)} punches. Need 2+!\n\nThrow HARDER!")
                self.combo_record_btn.config(text="🔴 START RECORDING COMBO", bg="#e53935")
                self.combo_status.config(text="Ready")
                return
            
            self.combo_sequences.append(self.recorded_combo.copy())
            self.combo_record_btn.config(text="🔴 START RECORDING COMBO", bg="#e53935")
            self.combo_status.config(text=f"✅ Saved!")
            self.current_combo_label.config(text=" → ".join([p.upper() for p in self.recorded_combo]))
            self.recorded_combos_label.config(text=f"{len(self.combo_sequences)} combos")
    
    def toggle_game(self):
        if not self.running:
            self.start_camera_thread()
            self.game_btn.config(text="⏹️ STOP GAME", bg="#e53935")
        else:
            self.stop_camera()
            self.game_btn.config(text="▶️ START GAME", bg="#4caf50")
    
    def add_to_bank(self, label, features):
        features_arr = np.array(features)
        
        if label in self.feature_bank:
            old = self.feature_bank[label]
            old_mean = np.array(old["mean"])
            old_std = np.array(old["std"])
            old_count = old["count"]
            
            new_mean = features_arr.mean(axis=0)
            new_std = features_arr.std(axis=0)
            new_count = len(features)
            
            total = old_count + new_count
            merged_mean = (old_mean * old_count + new_mean * new_count) / total
            merged_std = np.sqrt((old_std**2 * old_count + new_std**2 * new_count) / total)
            
            self.feature_bank[label] = {
                "mean": merged_mean.tolist(),
                "std": merged_std.tolist(),
                "count": total
            }
        else:
            self.feature_bank[label] = {
                "mean": features_arr.mean(axis=0).tolist(),
                "std": features_arr.std(axis=0).tolist(),
                "count": len(features)
            }
        
        print(f"✅ Added {len(features)} examples of {label}")
    
    def get_bank_status(self):
        if not self.feature_bank:
            return "Empty"
        
        text = ""
        for label in self.labels:
            if label in self.feature_bank:
                count = self.feature_bank[label]["count"]
                text += f"✅ {label}: {count}\n"
            else:
                text += f"❌ {label}: 0\n"
        return text.strip()
    
    def finish_teaching(self):
        if not self.feature_bank:
            messagebox.showwarning("No Data", "Record examples first!")
            return
        
        missing = [l for l in self.labels if l not in self.feature_bank]
        if missing:
            result = messagebox.askyesno("Incomplete",
                                        f"Missing: {', '.join(missing)}\n\nSave anyway?")
            if not result:
                return
        
        if self.save_data():
            messagebox.showinfo("Success", "✅ Saved!\n\nNow do STEP 2: TEACH COMBOS")
            self.back_to_menu()
        else:
            messagebox.showerror("Error", "Save failed!")
    
    def finish_combo_teaching(self):
        if len(self.combo_sequences) < 3:
            result = messagebox.askyesno("Few Combos",
                                        f"Only {len(self.combo_sequences)} combos.\n\nRecommend: 5+\n\nContinue?")
            if not result:
                return
        
        if self.save_data():
            self.predictor = self.build_predictor()
            messagebox.showinfo("Success",
                                f"✅ Saved {len(self.combo_sequences)} combos\n✅ Built predictor with {len(self.predictor)} patterns\n\nReady for STEP 3!")
            self.back_to_menu()
        else:
            messagebox.showerror("Error", "Save failed!")
    
    def reset_score(self):
        self.correct_predictions = 0
        self.total_predictions = 0
        self.punch_history = []
        self.prediction = None
        self.update_game_ui()
    
    def update_game_ui(self):
        if self.total_predictions > 0:
            acc = int(self.correct_predictions / self.total_predictions * 100)
            self.score_label.config(text=f"{self.correct_predictions}/{self.total_predictions} ({acc}%)")
        else:
            self.score_label.config(text="0/0 (0%)")
        
        if self.punch_history:
            trail = " → ".join([p.upper() for p in self.punch_history[-5:]])
            self.trail_label.config(text=trail)
        else:
            self.trail_label.config(text="—")
    
    def start_camera_thread(self):
        if not self.yolo_model:
            messagebox.showerror("Error", "YOLO not loaded!")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            
            # Reset energy tracking
            self.energy_history.clear()
            self.last_energy_spike = 0
            
            threading.Thread(target=self.process_frames, daemon=True).start()
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
    
    def stop_camera(self):
        self.running = False
        time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def process_frames(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_counter += 1
                fidx = self.frame_counter
                h, w = frame.shape[:2]
                frame = cv2.flip(frame, 1)
                
                results = self.yolo_model.predict(
                    frame, verbose=False, device=self.device,
                    half=USE_FP16, imgsz=YOLO_IMG_SIZE
                )
                
                kp_upper = None
                try:
                    kts = results[0].keypoints
                    if kts and len(kts.xy) > 0:
                        all_kp = kts.xy[0].cpu().numpy().astype(np.float32)
                        if all_kp.shape[0] >= 13:
                            kp_upper = extract_upper_body(all_kp)
                except:
                    pass
                
                if kp_upper is not None:
                    draw_skeleton(frame, kp_upper)
                    
                    feats, self.prev_kpts, self.prev_vel, energy = \
                        compute_features(kp_upper, w, h, self.prev_kpts, self.prev_vel)
                    
                    self.feat_seq.append(feats)
                    self.energy_history.append(energy)
                    
                    # NEW: Calculate energy CHANGE (prevents static high energy from triggering)
                    energy_change = 0.0
                    if len(self.energy_history) >= 5:
                        recent_avg = np.mean(list(self.energy_history)[-5:])
                        older_avg = np.mean(list(self.energy_history)[-10:-5]) if len(self.energy_history) >= 10 else 0
                        energy_change = recent_avg - older_avg
                    
                    debug_lines = [
                        f"Energy: {energy:.4f} (need >{ENERGY_THRESHOLD})",
                        f"Change: {energy_change:.4f} (need >{ENERGY_CHANGE_THRESHOLD})",
                        f"Cooldown: {max(0, PUNCH_COOLDOWN - (fidx - self.last_punch_frame))} frames",
                        f"Buffer: {len(self.feat_seq)}/{CLASS_WINDOW}"
                    ]
                    
                    # TEACH PUNCHES
                    if self.mode == "TEACH_PUNCHES" and self.recording:
                        # FIXED: Require BOTH high energy AND energy change
                        if (energy > ENERGY_THRESHOLD and 
                            energy_change > ENERGY_CHANGE_THRESHOLD and
                            len(self.feat_seq) == CLASS_WINDOW):
                            
                            avg_feat = np.mean(list(self.feat_seq), axis=0)
                            self.recorded_features.append(avg_feat)
                            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
                            cv2.putText(frame, f"REC: {len(self.recorded_features)}", 
                                       (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # DETECT & GAME
                    elif self.mode in ["TEACH_COMBOS", "GAME"]:
                        # FIXED: Require BOTH high energy AND energy change AND cooldown
                        if (len(self.feat_seq) == CLASS_WINDOW and 
                            energy > ENERGY_THRESHOLD and 
                            energy_change > ENERGY_CHANGE_THRESHOLD and
                            (fidx - self.last_punch_frame) > PUNCH_COOLDOWN):
                            
                            avg_feat = np.mean(list(self.feat_seq), axis=0)
                            label, conf = classify_punch(avg_feat, self.feature_bank)
                            
                            debug_lines.append(f"Detected: {label} ({conf:.0%})")
                            
                            if label != "unknown":
                                smoothed, self.smooth_history = \
                                    temporal_smooth(self.smooth_history, label, conf, TEMPORAL_WINDOW)
                                
                                # FIXED: Only trigger if smoothed matches AND high confidence
                                if smoothed == label and conf >= MIN_CONFIDENCE:
                                    self.last_punch_frame = fidx
                                    self.last_punch = label
                                    
                                    # TEACH COMBOS
                                    if self.mode == "TEACH_COMBOS" and self.recording:
                                        self.recorded_combo.append(label)
                                    
                                    # GAME
                                    elif self.mode == "GAME":
                                        if self.prediction:
                                            self.total_predictions += 1
                                            if self.prediction == label:
                                                self.correct_predictions += 1
                                        
                                        self.punch_history.append(label)
                                        
                                        next_move, next_conf = predict_next_move(self.punch_history, self.predictor)
                                        self.prediction = next_move
                                        self.pred_confidence = next_conf
                                    
                                    # Draw
                                    colour = PUNCH_COLOURS.get(label, DEFAULT_COLOUR)
                                    cv2.putText(frame, label.upper(), (20, 80), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 2.5, colour, 5)
                                    cv2.putText(frame, f"{conf:.0%}", (20, 140), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour, 3)
                    
                    self.debug_text = "\n".join(debug_lines)
                
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            except Exception as e:
                print(f"Process error: {e}")
                self.debug_text = f"Error: {e}"
            
            time.sleep(0.03)
    
    def update_display(self):
        if not self.running:
            return
        
        try:
            if self.current_frame is not None:
                img = Image.fromarray(self.current_frame)
                img = img.resize((800, 600), Image.BILINEAR)
                self.photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=self.photo)
            
            if self.mode == "GAME" and hasattr(self, 'debug_label'):
                self.debug_label.config(text=self.debug_text)
            
            if self.mode == "TEACH_COMBOS" and hasattr(self, 'current_combo_label'):
                combo_text = " → ".join([p.upper() for p in self.recorded_combo])
                self.current_combo_label.config(text=combo_text if combo_text else "—")
            
            if self.mode == "GAME" and hasattr(self, 'pred_label'):
                if self.prediction:
                    self.pred_label.config(text=self.prediction.upper())
                    self.pred_conf_label.config(text=f"{self.pred_confidence:.0%}")
                else:
                    self.pred_label.config(text="…")
                
                self.update_game_ui()
        
        except Exception as e:
            print(f"Display error: {e}")
        
        self.root.after(30, self.update_display)
    
    def back_to_menu(self):
        self.stop_camera()
        self.feature_bank = self.load_feature_bank()
        self.combo_sequences = self.load_combo_sequences()
        self.predictor = self.build_predictor()
        self.build_menu()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = BoxingAI(root)
        root.mainloop()
    except Exception as e:
        print(f"\n❌ FATAL: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter...")