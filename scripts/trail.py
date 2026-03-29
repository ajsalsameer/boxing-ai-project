import os
import pickle
import glob

# Paths
BANK_PATH = "data/feature_bank.pkl"
SEQ_PATH = "data/combo_sequences.pkl"
MODELS_DIR = "models"

TARGET_CLASSES = {'cross', 'hook', 'idle', 'jab', 'uppercut'}

def clean_models():
    print("🧹 Cleaning models folder...")
    files = glob.glob(os.path.join(MODELS_DIR, "*"))
    for f in files:
        try:
            os.remove(f)
            print(f"   Deleted: {f}")
        except Exception as e:
            print(f"   Error deleting {f}: {e}")

def clean_feature_bank():
    if not os.path.exists(BANK_PATH):
        print("⚠️ No feature bank found.")
        return

    print("\n🧹 Cleaning Feature Bank...")
    with open(BANK_PATH, "rb") as f:
        bank = pickle.load(f)
    
    # Identify keys to remove
    keys_to_remove = [k for k in bank.keys() if k not in TARGET_CLASSES]
    
    for k in keys_to_remove:
        del bank[k]
        print(f"   ❌ Removed class: '{k}'")
        
    # Verify we have the right classes
    print(f"   ✅ Remaining classes: {list(bank.keys())}")
    
    with open(BANK_PATH, "wb") as f:
        pickle.dump(bank, f)

def clean_sequences():
    if not os.path.exists(SEQ_PATH):
        print("⚠️ No sequence data found.")
        return

    print("\n🧹 Cleaning Sequence Data...")
    with open(SEQ_PATH, "rb") as f:
        seqs = pickle.load(f)
    
    original_count = len(seqs)
    cleaned_seqs = []
    
    for s in seqs:
        # Check 'next' label
        if s["next"] not in TARGET_CLASSES:
            continue
        
        # Check history labels
        if any(h not in TARGET_CLASSES for h in s["history"]):
            continue
            
        cleaned_seqs.append(s)
        
    print(f"   Original sequences: {original_count}")
    print(f"   Cleaned sequences:  {len(cleaned_seqs)}")
    print(f"   ❌ Removed {original_count - len(cleaned_seqs)} entries containing 'combos'")
    
    with open(SEQ_PATH, "wb") as f:
        pickle.dump(cleaned_seqs, f)

if __name__ == "__main__":
    print("=== BOXING AI RESET TOOL ===")
    clean_models()
    clean_feature_bank()
    clean_sequences()
    print("\n✅ DONE! You are ready to retrain.")