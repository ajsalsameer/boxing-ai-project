"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — FASTAPI SERVER v4                              ║
║   TCN recognition + 2nd-order Markov next-punch prediction   ║
║   + User profiles + idle class                               ║
╚══════════════════════════════════════════════════════════════╝

Run:   python server.py
React: http://localhost:3000
"""

import base64, collections, json, time, warnings
warnings.filterwarnings("ignore")

import cv2, numpy as np, torch
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel

from models_def import (
    CLASSES, NUM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX, IDLE_IDX,
    LANDMARKS_USED, SEQ_LEN, INPUT_SIZE,
    extract_keypoints_from_results, SmoothPredictor,
    load_tcn, infer_tcn, load_markov, infer_markov,
)
from markov_predictor import MarkovPredictor
from user_profiles import (
    UserProfile, list_users,
    train_personal_gru_async,   # kept for backward compat — can be removed later
)

MODEL_DIR = Path("models")

# ─────────────────────────────────────────────────
#  COACH COMBOS
# ─────────────────────────────────────────────────
COACH_COMBOS = [
    ["jab"], ["cross"], ["hook"], ["uppercut"],
    ["jab","cross"], ["jab","cross"],
    ["jab","cross","hook"], ["jab","cross","hook"],
    ["jab","cross","hook","cross"],
    ["jab","cross","hook","uppercut"],
    ["hook","uppercut"], ["cross","hook","uppercut"],
    ["jab","jab","cross"],
]


# ─────────────────────────────────────────────────
#  COACH SESSION
# ─────────────────────────────────────────────────
class CoachSession:
    def __init__(self):
        self.combo_queue=[]; self.current_cmd=None; self.cmd_time=0.0
        self.reps_done=0; self.reps_correct=0; self.mistakes=0
        self.reaction_times=[]; self.feedback=""; self.feedback_color="#4caf50"
        self.feedback_time=0.0; self.active=False; self.phase="idle"
        self.current_anim="idle"; self.total_cmds=0

    def start(self):
        self.combo_queue=[m for c in COACH_COMBOS for m in c]
        self.total_cmds=len(self.combo_queue); self.reps_done=0
        self.reps_correct=0; self.mistakes=0; self.reaction_times=[]
        self.active=True; self.phase="waiting"; self.next_command()

    def next_command(self):
        if not self.combo_queue:
            self.phase="done"; self.current_cmd=None; self.current_anim="idle"; return
        self.current_cmd=self.combo_queue.pop(0); self.cmd_time=time.time()
        self.current_anim=self.current_cmd; self.phase="waiting"

    def judge(self, detected):
        if self.phase!="waiting" or not self.current_cmd: return
        correct = detected==self.current_cmd
        if self.current_cmd in ("jab","cross") and detected in ("jab","cross"):
            correct=True
        elapsed=time.time()-self.cmd_time; self.reps_done+=1
        if correct:
            self.reps_correct+=1; self.reaction_times.append(elapsed)
            if elapsed<1.0: self.feedback=f"⚡ LIGHTNING! {elapsed:.2f}s"; self.feedback_color="#00ff88"
            elif elapsed<2.0: self.feedback=f"✅ GREAT! {elapsed:.2f}s"; self.feedback_color="#4caf50"
            else: self.feedback=f"👍 CORRECT — faster! {elapsed:.2f}s"; self.feedback_color="#ffd700"
            self.next_command()
        else:
            self.mistakes+=1
            self.feedback=f"❌ WRONG! Wanted {self.current_cmd.upper()}, got {detected.upper()}"
            self.feedback_color="#ff4444"
        self.feedback_time=time.time()

    def stats(self):
        total=self.reps_correct+self.mistakes
        avg=float(np.mean(self.reaction_times)) if self.reaction_times else 0.0
        return {"reps":self.reps_done,"correct":self.reps_correct,"mistakes":self.mistakes,
                "avg_time":round(avg,2),
                "accuracy":round(self.reps_correct/total*100,1) if total else 0.0,
                "total":self.total_cmds}


# ─────────────────────────────────────────────────
#  COMBO RECORDING SESSION
# ─────────────────────────────────────────────────
class ComboRecordSession:
    PAUSE_SEC = 2.0
    MIN_COMBO = 2

    def __init__(self):
        self.active=False; self.current_combo=[]
        self.saved_combos=[]; self.last_punch_t=0.0
        self.status="idle"; self.status_msg=""

    def start(self):
        self.active=True; self.current_combo=[]
        self.saved_combos=[]; self.last_punch_t=time.time()
        self.status="recording"; self.status_msg="Start throwing combos!"

    def stop(self):
        self._flush(); self.active=False; self.status="idle"
        self.status_msg=f"Session ended — {len(self.saved_combos)} combos saved"

    def add_punch(self, move):
        if not self.active or move=="idle": return
        self.current_combo.append(move); self.last_punch_t=time.time()
        self.status="recording"
        self.status_msg=f"Recording: {' → '.join(self.current_combo)}"

    def tick(self):
        if not self.active or not self.current_combo: return
        if time.time()-self.last_punch_t > self.PAUSE_SEC: self._flush()

    def _flush(self):
        if len(self.current_combo) >= self.MIN_COMBO:
            self.saved_combos.append(list(self.current_combo))
            self.status_msg=f"✅ Combo saved: {' → '.join(self.current_combo)}"
        self.current_combo=[]

    def summary(self):
        return {"active":self.active,"status":self.status,"status_msg":self.status_msg,
                "current_combo":self.current_combo,"saved_combos":self.saved_combos,
                "count":len(self.saved_combos)}


# ─────────────────────────────────────────────────
#  AI ENGINE
# ─────────────────────────────────────────────────
class BoxingEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tcn=None; self.tcn_ckpt={}
        print(f"  Device: {self.device}")
        if (MODEL_DIR/"tcn_boxing.pth").exists():
            try:
                self.tcn, self.tcn_ckpt = load_tcn(MODEL_DIR/"tcn_boxing.pth", self.device)
                print(f"  ✅ TCN  val_acc={self.tcn_ckpt.get('val_acc',0):.1%}  "
                      f"classes={self.tcn_ckpt.get('num_classes',4)}")
            except Exception as e: print(f"  ❌ TCN: {e}")
        else:
            print("  ⚠️  TCN not found → run stage2_train_tcn.py")


# ─────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────
app    = FastAPI(title="Boxing AI v4")
engine = BoxingEngine()

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return HTMLResponse("""<html><body style="background:#050508;color:#00dcff;font-family:monospace;padding:40px">
    <h2>⚡ Boxing AI v4</h2>
    <p>Predictor: 2nd-order Markov chain</p>
    <p>React → <a href="http://localhost:3000" style="color:#00ff88">http://localhost:3000</a></p>
    </body></html>""")

@app.get("/status")
async def status():
    return {"tcn_loaded": engine.tcn is not None,
            "predictor":  "markov",
            "device":     engine.device,
            "classes":    CLASSES,
            "val_acc":    engine.tcn_ckpt.get("val_acc")}

@app.get("/users")
async def get_users():
    return {"users": list_users()}

@app.get("/users/{username}")
async def get_user(username: str):
    p = UserProfile(username)
    return p.summary()

@app.get("/users/{username}/markov")
async def get_markov(username: str):
    """Return the Markov transition table for this user."""
    m = MarkovPredictor(username)
    m.load()
    return {"username": username, "stats": m.stats(),
            "first_order": m.first_order,
            "total_transitions": m.total_transitions}

class CreateUserBody(BaseModel):
    username: str

@app.post("/users")
async def create_user(body: CreateUserBody):
    p = UserProfile(body.username)
    p.save()
    return {"ok": True, "username": p.username, "summary": p.summary()}

@app.delete("/users/{username}/combos")
async def clear_combos(username: str):
    p = UserProfile(username)
    p.combo_history = []
    p.save()
    return {"ok": True, "message": "Combo history cleared"}


# ─────────────────────────────────────────────────
#  WEBSOCKET
# ─────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    print("🔌 Client connected")

    mp_pose_sol = mp.solutions.pose
    mp_draw     = mp.solutions.drawing_utils
    pose_model  = mp_pose_sol.Pose(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)

    frame_buffer = collections.deque(maxlen=SEQ_LEN)
    smoother     = SmoothPredictor(window=4, conf_thresh=0.60, idle_thresh=0.50)
    coach        = CoachSession()
    recorder     = ComboRecordSession()
    mode         = "freeplay"

    # User + Markov state
    current_user = None
    markov       = load_markov("global")
    markov_label = "global"

    fps_deque = collections.deque(maxlen=30)
    t_last    = time.time()

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)

            # ── SET USER ──────────────────────────────────
            if msg["type"] == "set_user":
                uname = msg.get("username","").strip()
                if uname:
                    current_user = UserProfile(uname)
                    # Load user-specific Markov (falls back to global if sparse)
                    markov       = load_markov(uname)
                    markov_label = uname if markov.username == uname else "global"

                    # Feed historical combo data into Markov
                    if current_user.combo_history:
                        for combo in current_user.combo_history:
                            markov.add_session_log(combo)
                        print(f"  👤 {uname}: loaded {markov.total_transitions} Markov transitions "
                              f"from {len(current_user.combo_history)} combos")

                    await ws.send_text(json.dumps({
                        "type":         "user_loaded",
                        "username":     current_user.username,
                        "summary":      current_user.summary(),
                        "markov_label": markov_label,
                        "markov_stats": markov.stats(),
                    }))
                continue

            # ── SET MODE ──────────────────────────────────
            if msg["type"] == "set_mode":
                mode = msg["mode"]
                if mode=="coach" and not coach.active:  coach.start()
                elif mode=="freeplay":                  coach=CoachSession()
                elif mode=="record":
                    recorder=ComboRecordSession(); recorder.start()
                await ws.send_text(json.dumps({"type":"mode_ack","mode":mode})); continue

            # ── STOP RECORDING ────────────────────────────
            if msg["type"] == "record_stop":
                recorder.stop()
                combos = recorder.saved_combos
                if current_user and combos:
                    for c in combos:
                        current_user.add_combo(c)
                        # Feed new combos into Markov immediately
                        markov.add_session_log(c)
                    markov.save()
                    await ws.send_text(json.dumps({
                        "type":         "record_done",
                        "combos":       combos,
                        "total":        len(current_user.combo_history),
                        "summary":      current_user.summary(),
                        "markov_stats": markov.stats(),
                    }))
                else:
                    await ws.send_text(json.dumps({"type":"record_done","combos":combos,"total":0}))
                mode="freeplay"; continue

            # ── COACH STOP ────────────────────────────────
            if msg["type"] == "coach_stop":
                if current_user and coach.active:
                    current_user.add_session(coach.stats())
                coach=CoachSession(); mode="freeplay"
                await ws.send_text(json.dumps({"type":"mode_ack","mode":"freeplay"})); continue

            if msg["type"] != "frame": continue

            # ── FRAME ─────────────────────────────────────
            img_data = base64.b64decode(msg["data"].split(",")[-1])
            frame    = cv2.imdecode(np.frombuffer(img_data,np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            frame = cv2.flip(frame,1)

            results = pose_model.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame,results.pose_landmarks,mp_pose_sol.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,120),thickness=2,circle_radius=4),
                    mp_draw.DrawingSpec(color=(0,200,255),thickness=2))

            kpts = extract_keypoints_from_results(results, LANDMARKS_USED)
            if kpts is not None: frame_buffer.append(kpts)

            current_move="idle"; probs=[0.0]*NUM_CLASSES
            next_move=None; next_conf=0.0; is_new=False

            if engine.tcn is not None and len(frame_buffer)==SEQ_LEN:
                logits = infer_tcn(engine.tcn, list(frame_buffer), engine.device)
                current_move, prob_arr, is_new = smoother.update(logits, frame_buffer=frame_buffer)
                probs = prob_arr.tolist()

                if is_new and current_move != "idle":
                    # Feed into Markov — this is how it learns your rhythm live
                    markov.add_punch(current_move)
                    next_move, next_conf = infer_markov(markov, smoother.combo)

                    if mode=="coach":   coach.judge(current_move)
                    if mode=="record":  recorder.add_punch(current_move)

            if mode=="record": recorder.tick()

            now=time.time()
            fps_deque.append(1.0/max(now-t_last,0.001))
            t_last=now

            _, jpeg   = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,72])
            frame_b64 = "data:image/jpeg;base64,"+base64.b64encode(jpeg).decode()

            feedback = coach.feedback if coach.active and (time.time()-coach.feedback_time)<4.5 else ""
            fb_color = coach.feedback_color if feedback else "#4caf50"

            await ws.send_text(json.dumps({
                "type":         "inference",
                "frame":        frame_b64,
                "move":         current_move,
                "probs":        probs,
                "combo":        smoother.combo[-6:],
                "next_move":    next_move,
                "next_conf":    round(next_conf,2),
                "fps":          round(float(np.mean(fps_deque)),1),
                "pose_detected":results.pose_landmarks is not None,
                "mode":         mode,
                "markov_label": markov_label,
                "markov_stats": markov.stats(),
                "user":         current_user.username if current_user else None,
                "coach": {
                    "active":   coach.active, "phase": coach.phase,
                    "command":  coach.current_cmd, "anim": coach.current_anim,
                    "feedback": feedback, "fb_color": fb_color,
                    "stats":    coach.stats() if coach.active else {},
                    "cmd_elapsed": round(time.time()-coach.cmd_time,1) if coach.active and coach.cmd_time else 0,
                },
                "record": recorder.summary() if mode=="record" else None,
            }))

    except WebSocketDisconnect:
        print("🔌 Disconnected")
    except Exception as e:
        print(f"⚠️  WS error: {e}")
        import traceback; traceback.print_exc()
    finally:
        # Save Markov on disconnect
        if markov.total_transitions > 0:
            markov.save()
            print(f"  💾 Markov saved ({markov.total_transitions} transitions)")
        pose_model.close()


if __name__=="__main__":
    import uvicorn
    print("╔═════════════════════════════════════════════╗")
    print("║  BOXING AI v4 — server on :8000             ║")
    print("║  Predictor: 2nd-order Markov chain           ║")
    print("╚═════════════════════════════════════════════╝\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")