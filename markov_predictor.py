"""
╔══════════════════════════════════════════════════════════════╗
║   BOXING AI — 2ND-ORDER MARKOV PREDICTOR                     ║
║   Replaces GRU for next-punch prediction                     ║
╚══════════════════════════════════════════════════════════════╝

WHY THIS BEATS GRU FOR YOUR SITUATION:
  GRU was trained on COMBO_SEQUENCES — a hardcoded list you wrote.
  It predicted "hook" after "jab→cross" because you wrote that combo,
  not because YOU actually throw hook after jab→cross 70% of the time.

  This Markov chain learns from YOUR actual confirmed punches in real-time.
  After one 10-minute session it already knows your personal rhythm.

2ND-ORDER means it looks at the last TWO punches:
  "jab → cross → ?"  is different from  "hook → cross → ?"
  Same last punch (cross), completely different next prediction.
  This matches how boxing combos actually work.

RECENCY WEIGHTING:
  Recent transitions count more than old ones.
  If you've been throwing jab→cross→uppercut all session,
  that pattern rises above your older jab→cross→hook history.
  Weight of transition i (from the end): DECAY^(n-1-i)
  Default DECAY=0.92 → 10 punches ago counts 43% as much as the last one.

STORAGE:
  models/markov_{username}.json  — per-user transition matrix
  models/markov_global.json      — fallback if no user or sparse data

FORMAT:
  {
    "transitions": {
      "jab|cross": {"hook": 14.2, "cross": 3.1, "jab": 2.0, "uppercut": 1.5},
      "cross|hook": {"cross": 8.5, "jab": 3.2, ...},
      ...
    },
    "first_order": {          ← fallback when 2nd-order has no data
      "jab":   {"cross": 12.1, "hook": 3.2, ...},
      ...
    },
    "total_transitions": 247,
    "last_updated": "2026-03-19T12:34:56"
  }
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────
PUNCH_CLASSES = ["jab", "cross", "hook", "uppercut"]
DECAY         = 0.92    # recency weight decay per punch
MIN_COUNT     = 3       # minimum weighted count to trust a transition
MODEL_DIR     = Path("models")


class MarkovPredictor:
    """
    2nd-order Markov chain with recency weighting.

    Usage:
        mp = MarkovPredictor("ajsal")
        mp.load()

        # After every confirmed punch in real-time:
        mp.add_punch("jab")
        mp.add_punch("cross")

        # Predict next:
        next_punch, confidence = mp.predict(["jab", "cross"])
        # → ("hook", 0.71)

        # Save at end of session:
        mp.save()

        # Print the table:
        mp.print_table()
    """

    def __init__(self, username: str = "global"):
        self.username   = username
        self.path       = MODEL_DIR / f"markov_{username}.json"
        # transitions["A|B"]["C"] = weighted count of A→B→C
        self.transitions: dict[str, dict[str, float]] = {}
        # first_order["A"]["B"] = weighted count of A→B
        self.first_order: dict[str, dict[str, float]] = {}
        self.total_transitions = 0
        # Rolling history of punches seen this session (for recency)
        self._session_history: list[str] = []

    # ── Load / Save ───────────────────────────────────────────
    def load(self) -> bool:
        """Load from JSON. Returns True if file existed."""
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text())
            self.transitions       = data.get("transitions", {})
            self.first_order       = data.get("first_order", {})
            self.total_transitions = data.get("total_transitions", 0)
            return True
        except Exception as e:
            print(f"  ⚠️  Markov load error: {e}")
            return False

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        data = {
            "transitions":       self.transitions,
            "first_order":       self.first_order,
            "total_transitions": self.total_transitions,
            "last_updated":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "username":          self.username,
        }
        self.path.write_text(json.dumps(data, indent=2))

    # ── Core update ───────────────────────────────────────────
    def add_punch(self, punch: str, weight: float = 1.0):
        """
        Record one confirmed punch.
        Call this every time stage4/server confirms a punch.
        Weight defaults to 1.0 for real-time; can be set lower
        for bulk-loading old session data.
        """
        if punch not in PUNCH_CLASSES:
            return

        h = self._session_history
        h.append(punch)

        # First-order: prev → current
        if len(h) >= 2:
            prev = h[-2]
            if prev not in self.first_order:
                self.first_order[prev] = {}
            self.first_order[prev][punch] = \
                self.first_order[prev].get(punch, 0.0) + weight

        # Second-order: prev_prev|prev → current
        if len(h) >= 3:
            key = f"{h[-3]}|{h[-2]}"
            if key not in self.transitions:
                self.transitions[key] = {}
            self.transitions[key][punch] = \
                self.transitions[key].get(punch, 0.0) + weight

        self.total_transitions += 1

    def add_session_log(self, punch_sequence: list[str]):
        """
        Bulk-add a sequence of punches from a saved session log.
        Uses recency weighting: punches at the end of the list
        get weight 1.0, older ones decay by DECAY per step.

        Call this when loading historical session data.
        """
        n = len(punch_sequence)
        for i, punch in enumerate(punch_sequence):
            # weight: 1.0 for most recent, DECAY^k for k steps back
            w = DECAY ** (n - 1 - i)
            self.add_punch(punch, weight=w)

    # ── Prediction ────────────────────────────────────────────
    def predict(self, recent_punches: list[str]) -> tuple[Optional[str], float]:
        """
        Predict next punch given recent combo history.

        Args:
            recent_punches: last 2+ confirmed punches (punch-only, no idle)

        Returns:
            (predicted_punch, confidence) or (None, 0.0) if no data
        """
        if not recent_punches:
            return None, 0.0

        # Try 2nd-order first (most specific)
        if len(recent_punches) >= 2:
            key = f"{recent_punches[-2]}|{recent_punches[-1]}"
            result = self._lookup(self.transitions.get(key, {}))
            if result[0] is not None:
                return result

        # Fall back to 1st-order
        last = recent_punches[-1]
        result = self._lookup(self.first_order.get(last, {}))
        if result[0] is not None:
            return result

        # No data at all — uniform over punch classes
        return None, 0.0

    def _lookup(self, counts: dict[str, float]) -> tuple[Optional[str], float]:
        """
        Convert a raw count dict → (best_punch, confidence).
        Only uses entries above MIN_COUNT threshold.
        """
        if not counts:
            return None, 0.0

        # Filter to valid punch classes above threshold
        valid = {k: v for k, v in counts.items()
                 if k in PUNCH_CLASSES and v >= MIN_COUNT}
        if not valid:
            return None, 0.0

        total = sum(valid.values())
        best  = max(valid, key=valid.get)
        conf  = valid[best] / total
        return best, round(conf, 3)

    # ── Diagnostics ──────────────────────────────────────────
    def print_table(self):
        """Print the transition table — shows your personal patterns."""
        print(f"\n{'═'*60}")
        print(f"  MARKOV TRANSITION TABLE — {self.username}")
        print(f"  Total transitions logged: {self.total_transitions}")
        print(f"{'═'*60}\n")

        if not self.first_order:
            print("  No data yet. Throw some punches in stage4!\n")
            return

        # First-order table
        print("  FIRST-ORDER (last punch → next):")
        print(f"  {'':12s}", end="")
        for cls in PUNCH_CLASSES:
            print(f"  {cls[:5]:<6}", end="")
        print()
        print("  " + "─" * 40)

        for from_cls in PUNCH_CLASSES:
            counts = self.first_order.get(from_cls, {})
            total  = sum(counts.values()) if counts else 0
            print(f"  {from_cls:<12}", end="")
            for to_cls in PUNCH_CLASSES:
                if total > 0 and to_cls in counts:
                    pct = counts[to_cls] / total * 100
                    print(f"  {pct:5.1f}%", end="")
                else:
                    print(f"  {'—':>6}", end="")
            print()

        # 2nd-order highlights (only show entries with enough data)
        if self.transitions:
            print(f"\n  2ND-ORDER HIGHLIGHTS (context → next):")
            for key, counts in sorted(self.transitions.items()):
                total = sum(counts.values())
                if total < MIN_COUNT * 2:
                    continue
                best     = max(counts, key=counts.get)
                best_pct = counts[best] / total * 100
                if best_pct > 50:  # only show strong tendencies
                    print(f"    [{key.replace('|',' → ')}]  →  "
                          f"{best.upper()} ({best_pct:.0f}%)")

        print()

    def stats(self) -> dict:
        """Return summary stats for the frontend."""
        return {
            "total_transitions": self.total_transitions,
            "has_data":          self.total_transitions >= 10,
            "username":          self.username,
            "top_transitions":   self._top_transitions(5),
        }

    def _top_transitions(self, n: int) -> list[dict]:
        """Return top N most frequent transitions."""
        out = []
        for from_cls in PUNCH_CLASSES:
            counts = self.first_order.get(from_cls, {})
            total  = sum(counts.values())
            if total < MIN_COUNT:
                continue
            for to_cls, cnt in counts.items():
                if to_cls in PUNCH_CLASSES:
                    out.append({
                        "from": from_cls, "to": to_cls,
                        "pct":  round(cnt / total * 100, 1),
                        "count": round(cnt, 1),
                    })
        out.sort(key=lambda x: -x["pct"])
        return out[:n]


# ── Global fallback ───────────────────────────────────────────
def load_markov(username: str = "global") -> MarkovPredictor:
    """
    Load Markov for a user. If their file doesn't exist or has
    too little data, falls back to global.
    """
    mp = MarkovPredictor(username)
    loaded = mp.load()

    if loaded and mp.total_transitions >= 10:
        return mp

    # Fall back to global
    if username != "global":
        gm = MarkovPredictor("global")
        gm.load()
        if gm.total_transitions >= 10:
            print(f"  ℹ️  {username}: sparse data ({mp.total_transitions} transitions) "
                  f"→ using global Markov")
            return gm

    # Return empty (will give no predictions until data builds up)
    return mp


def infer_markov(markov: MarkovPredictor,
                 combo_history: list[str]) -> tuple[Optional[str], float]:
    """
    Drop-in replacement for infer_gru().
    combo_history is punch-only (no idle) — same as before.
    """
    if not combo_history or markov is None:
        return None, 0.0
    return markov.predict(combo_history)