# resave_model.py

from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func
import pickle
import numpy as np

# Load the model (trained using numpy 1.26.x)
with open("final_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Remove MT19937 dependency (if stored in classifier's random_state)
classifier = pipeline.named_steps["classifier"]

# If using MLPClassifier with a random_state that references MT19937, replace it
if hasattr(classifier, "random_state") and hasattr(classifier.random_state, "bit_generator"):
    print("Replacing MT19937 with new default BitGenerator (PCG64)")
    classifier.random_state = np.random.default_rng()  # compatible with NumPy 2.0

# Save it to a new pickle file
with open("final_pipeline_clean.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model re-saved safely with a compatible random generator.")
