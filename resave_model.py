# resave_model.py

import pickle
import numpy as np
from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func

# Step 1: Load the old pickle (trained on NumPy < 2.0)
with open("final_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Step 2: Fix the random_state in the classifier if it's using MT19937
classifier = pipeline.named_steps.get("classifier")
if classifier and hasattr(classifier, "random_state") and hasattr(classifier.random_state, "bit_generator"):
    print("⚙️ Replacing MT19937 with PCG64 to fix NumPy 2.0 incompatibility")
    classifier.random_state = np.random.default_rng()

# Step 3: Save to a new cleaned pickle
with open("final_pipeline_clean.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model re-saved without MT19937 — ready for Render!")
