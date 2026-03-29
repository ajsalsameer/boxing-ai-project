"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — USER PROFILE MANAGER                          ║
║   Per-user combo history, session stats, Markov data         ║
╚══════════════════════════════════════════════════════════════╝

CHANGE LOG:
  v2 — Removed personal GRU training entirely.
       Replaced by 2nd-order Markov predictor (markov_predictor.py).
       UserProfile now feeds combo_history into MarkovPredictor
       via server.py on user login — no separate training step needed.

       train_personal_gru_async kept as a no-op stub because
       server.py still imports it. Safe to call — does nothing.

       has_model in summary now reflects Markov data sufficiency
       (10+ transitions) instead of GRU file existence.
"""

import json
import time
import threading
from pathlib import Path

from models_def import CLASSES, NUM_CLASSES, CLASS_TO_IDX

USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
#  USER PROFILE
# ─────────────────────────────────────────────
class UserProfile:
    """
    Stores per-user:
      - combo_history: list of recorded punch sequences [[jab,cross,hook], ...]
      - session_stats: past coach session performance
      - Markov data is derived from combo_history at runtime (no separate file)
    """
    def __init__(self, username: str):
        self.username  = username.strip().lower().replace(" ", "_")
        self.user_dir  = USER_DATA_DIR / self.username
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.user_dir / "profile.json"
        # Markov model file (written by MarkovPredictor.save())
        self.markov_path = Path("models") / f"markov_{self.username}.json"
        self._load()

    def _load(self):
        if self.data_path.exists():
            raw = json.loads(self.data_path.read_text())
        else:
            raw = {}
        self.combo_history  = raw.get("combo_history",  [])
        self.session_stats  = raw.get("session_stats",  [])
        self.created_at     = raw.get("created_at", time.strftime("%Y-%m-%d"))
        self.total_sessions = raw.get("total_sessions", 0)

    def save(self):
        self.data_path.write_text(json.dumps({
            "username":       self.username,
            "combo_history":  self.combo_history,
            "session_stats":  self.session_stats,
            "created_at":     self.created_at,
            "total_sessions": self.total_sessions,
        }, indent=2))

    def add_combo(self, combo: list):
        """Add one recorded combo sequence (min 2 punches)."""
        if len(combo) >= 2:
            self.combo_history.append(combo)
            self.save()

    def add_session(self, stats: dict):
        self.session_stats.append({**stats, "date": time.strftime("%Y-%m-%d %H:%M")})
        self.total_sessions += 1
        if len(self.session_stats) > 50:
            self.session_stats = self.session_stats[-50:]
        self.save()

    def has_enough_data(self) -> bool:
        """True if enough combos recorded to be useful for Markov."""
        total_punches = sum(len(c) for c in self.combo_history)
        return total_punches >= 10

    def has_markov_data(self) -> bool:
        """True if a Markov file exists for this user."""
        return self.markov_path.exists()

    def summary(self) -> dict:
        total_punches = sum(len(c) for c in self.combo_history)

        # Per-move counts from combo history
        move_counts = {m: 0 for m in CLASSES}
        for combo in self.combo_history:
            for m in combo:
                if m in move_counts:
                    move_counts[m] += 1

        # Top transitions
        transitions = {}
        for combo in self.combo_history:
            for i in range(len(combo) - 1):
                key = f"{combo[i]}→{combo[i+1]}"
                transitions[key] = transitions.get(key, 0) + 1
        top_transitions = sorted(transitions.items(), key=lambda x: -x[1])[:5]

        # Markov transition count from saved file
        markov_transitions = 0
        if self.markov_path.exists():
            try:
                mdata = json.loads(self.markov_path.read_text())
                markov_transitions = mdata.get("total_transitions", 0)
            except Exception:
                pass

        return {
            "username":           self.username,
            "created_at":         self.created_at,
            "total_combos":       len(self.combo_history),
            "total_punches":      total_punches,
            "move_counts":        move_counts,
            "top_transitions":    [{"pair": k, "count": v} for k, v in top_transitions],
            "total_sessions":     self.total_sessions,
            # has_model: True when Markov has enough data to give useful predictions
            "has_model":          markov_transitions >= 10,
            "markov_transitions": markov_transitions,
            "session_stats":      self.session_stats[-10:],
        }


# ─────────────────────────────────────────────
#  STUB — kept so server.py import doesn't break
#  GRU training is replaced by Markov predictor.
#  Markov needs no training — it builds from live punches.
# ─────────────────────────────────────────────
def train_personal_gru_async(profile, device, progress_cb=None):
    """
    No-op stub. GRU replaced by Markov predictor.
    Markov builds automatically from confirmed punches during stage4/server.
    This function is kept only so existing imports don't break.
    """
    if progress_cb:
        progress_cb(100, "Markov predictor active — no training needed.")
    return threading.Thread(target=lambda: None, daemon=True)


# ─────────────────────────────────────────────
#  LIST ALL USERS
# ─────────────────────────────────────────────
def list_users() -> list:
    users = []
    for d in USER_DATA_DIR.iterdir():
        if d.is_dir() and (d / "profile.json").exists():
            p = UserProfile(d.name)
            s = p.summary()
            users.append({
                "username":           p.username,
                "combos":             len(p.combo_history),
                "sessions":           p.total_sessions,
                "has_model":          s["has_model"],
                "markov_transitions": s["markov_transitions"],
                "created_at":         p.created_at,
            })
    return sorted(users, key=lambda x: x["username"])