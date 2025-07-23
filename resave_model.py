from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func
import pickle

# Load the model
with open("final_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Fix for random state issue: set random_state only for the classifier
pipeline.named_steps["classifier"].random_state = None

# Save it again under a new filename
with open("final_pipeline_clean.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model re-saved successfully with cleaned random state.")
