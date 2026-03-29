"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI - STAGE 4: REAL-TIME                            ║
║   TCN recognition + 2nd-order Markov next-punch prediction  ║
╚══════════════════════════════════════════════════════════════╝

Keys: Q=quit  C=clear  S=screenshot  T=print Markov table
"""

import cv2, numpy as np, time, collections, mediapipe as mp
import torch
from pathlib import Path

from models_def import (
    CLASSES, NUM_CLASSES, IDX_TO_CLASS, IDLE_IDX,
    LANDMARKS_USED, SEQ_LEN,
    extract_keypoints_from_results, SmoothPredictor,
    load_tcn, infer_tcn, load_markov, infer_markov,
)

CONFIRM_THRESHOLD = 0.68
HOLD_SEC          = 1.8

MOVE_COLORS = {
    "jab":      (50,  200, 255),
    "cross":    (255, 200, 50),
    "hook":     (50,  255, 120),
    "uppercut": (255, 80,  180),
    "idle":     (90,  90,  110),
}
DIM = (45, 45, 45)


class ConfirmedDisplay:
    def __init__(self):
        self.move = None; self.probs = [0.0]*NUM_CLASSES
        self.conf = 0.0;  self.confirm_t = 0.0
        self.next_move = None; self.next_conf = 0.0

    def update(self, move, probs, conf, next_move, next_conf):
        if move == "idle": return
        self.move = move; self.probs = list(probs)
        self.conf = conf; self.confirm_t = time.time()
        if next_move and next_move != "idle":
            self.next_move = next_move; self.next_conf = next_conf

    def label_visible(self):
        return self.move is not None and (time.time()-self.confirm_t) < HOLD_SEC


def draw_hud(frame, confirmed, combo, fps, current_state, markov_stats):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,60), (8,8,12), -1)
    cv2.rectangle(ov, (0,h-180), (w,h), (8,8,12), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

    if confirmed.label_visible():
        c = MOVE_COLORS[confirmed.move]
        (tw,_),_ = cv2.getTextSize(confirmed.move.upper(), cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)
        cv2.putText(frame, confirmed.move.upper(), (14,42), cv2.FONT_HERSHEY_DUPLEX, 1.3, c, 2)
        cv2.putText(frame, f"{confirmed.conf:.0%}", (14+tw+12,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 1)
    else:
        cv2.putText(frame, "---", (14,42), cv2.FONT_HERSHEY_DUPLEX, 1.3, (55,55,55), 2)
        if current_state == "idle":
            cv2.putText(frame, "IDLE", (90,42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70,70,85), 1)

    # Markov data quality indicator
    transitions = markov_stats.get("total_transitions", 0)
    mkv_col = (0,200,100) if transitions >= 30 else ((0,180,255) if transitions >= 10 else (80,80,80))
    cv2.putText(frame, f"MKV:{transitions}", (w-180,34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mkv_col, 1)
    cv2.putText(frame, f"FPS {fps:.0f}", (w-95,34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70,70,70), 1)

    # Prob bars — all 5 classes
    bx, by = 14, h-175
    top_idx = int(np.argmax(confirmed.probs)) if any(p>0 for p in confirmed.probs) else -1
    header = "LAST CONFIRMED" if confirmed.move else "WAITING..."
    cv2.putText(frame, header, (bx,by-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (75,75,75), 1)

    for i,(cls,p) in enumerate(zip(CLASSES, confirmed.probs)):
        is_top = (i==top_idx and confirmed.move is not None and i!=IDLE_IDX)
        cc  = MOVE_COLORS[cls] if is_top else (DIM if i!=IDLE_IDX else (60,60,75))
        bw  = int(p*200)
        cv2.rectangle(frame,(bx,by+i*28),(bx+200,by+i*28+18),(18,18,18),-1)
        if bw > 0:
            cv2.rectangle(frame,(bx,by+i*28),(bx+bw,by+i*28+18),cc,-1)
        lbl = f"{'IDLE' if cls=='idle' else cls[:3].upper()}  {p:.0%}"
        cv2.putText(frame, lbl, (bx+208,by+i*28+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, cc if is_top else (55,55,65), 1)

    if confirmed.next_move and confirmed.next_move != "idle":
        nc = MOVE_COLORS.get(confirmed.next_move, (180,180,180))
        cv2.putText(frame,
                    f"NEXT: {confirmed.next_move.upper()}  ({confirmed.next_conf:.0%})",
                    (w//2-140, h-10), cv2.FONT_HERSHEY_DUPLEX, 0.75, nc, 2)

    if combo:
        cv2.putText(frame, " → ".join(combo[-6:]),
                    (14,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170,170,170), 1)

    return frame


def run():
    print("╔══════════════════════════════════════════╗")
    print("║  BOXING AI — Stage 4: Real-Time          ║")
    print(f"║  Predictor: 2nd-order Markov chain        ║")
    print(f"║  Confirm  : {CONFIRM_THRESHOLD:.0%}  Hold: {HOLD_SEC}s          ║")
    print("╚══════════════════════════════════════════╝\n")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    tcn_path = Path("models/tcn_boxing.pth")

    if not tcn_path.exists():
        print("❌ TCN not found — run stage2_train_tcn.py first"); return

    tcn, ckpt = load_tcn(tcn_path, device)
    print(f"✅ TCN  val_acc={ckpt.get('val_acc',0):.1%}  classes={ckpt.get('num_classes',4)}")

    if ckpt.get("num_classes", 4) != NUM_CLASSES:
        print(f"⚠️  Model has {ckpt.get('num_classes')} classes, expected {NUM_CLASSES}")
        print(f"   Collect idle samples and retrain: python stage2_train_tcn.py")
        return

    # Load Markov — builds automatically from punches thrown this session
    markov = load_markov("global")
    if markov.total_transitions > 0:
        print(f"✅ Markov loaded ({markov.total_transitions} transitions)")
    else:
        print("ℹ️  Markov has no data yet — predictions start after ~10 confirmed punches")

    pose    = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    buf       = collections.deque(maxlen=SEQ_LEN)
    smoother  = SmoothPredictor(window=4, conf_thresh=0.60, idle_thresh=0.30)
    fps_q     = collections.deque(maxlen=30)
    confirmed = ConfirmedDisplay()
    shot_idx  = 0

    print(f"\n  Q=quit  C=clear  S=screenshot  T=print Markov table\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,100), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255,255,0), thickness=2))

        kpts = extract_keypoints_from_results(results, LANDMARKS_USED)
        if kpts is not None:
            buf.append(kpts)

        current_state = "idle"
        if len(buf) == SEQ_LEN:
            logits                 = infer_tcn(tcn, list(buf), device)
            move, prob_arr, is_new = smoother.update(logits, frame_buffer=buf)
            top_conf               = float(prob_arr.max())
            current_state          = move

            if is_new and move != "idle" and top_conf >= CONFIRM_THRESHOLD:
                # Feed confirmed punch into Markov (this is how it learns your rhythm)
                markov.add_punch(move)

                next_move, next_conf = infer_markov(markov, smoother.combo)
                confirmed.update(move, prob_arr, top_conf, next_move, next_conf)

                pred_str = f"   next→ {next_move} ({next_conf:.0%})" if next_move else ""
                print(f"  ✅ {move.upper():<10} {top_conf:.0%}{pred_str}")

        fps_q.append(1.0/(time.time()-t0+1e-6))
        frame = draw_hud(frame, confirmed, smoother.combo,
                         np.mean(fps_q), current_state, markov.stats())
        cv2.imshow("🥊 Boxing AI — Stage 4", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            smoother.combo.clear(); smoother.last_logged = "idle"
            confirmed.__init__()
            print("  🔄 Cleared")
        elif k == ord('s'):
            fn = f"screenshot_{shot_idx:03d}.png"
            cv2.imwrite(fn, frame)
            print(f"  📸 {fn}"); shot_idx += 1
        elif k == ord('t'):
            markov.print_table()

    # Save Markov at end of session
    if markov.total_transitions > 0:
        markov.save()
        print(f"\n  💾 Markov saved ({markov.total_transitions} transitions)")
        markov.print_table()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("\n👋 Done.")


if __name__ == "__main__":
    run()