# resave_model.py

from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func
import pickle
import numpy as np

# Load pipeline
with open("final_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Remove numpy.random generator if any
if hasattr(pipeline, 'random_state'):
    pipeline.random_state = None
if hasattr(pipeline, 'set_params'):
    pipeline.set_params(random_state=None)

# Save pipeline again with legacy-compatible NumPy
with open("final_pipeline_clean.pkl", "wb") as f:
    pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model re-saved without numpy BitGenerator issue.")
