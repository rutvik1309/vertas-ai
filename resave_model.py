# resave_model.py

from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func
import pickle

# Step 1: Load the model
with open("final_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Step 2: Clean up any random state to avoid NumPy MT19937 error
def remove_random_state(obj):
    if hasattr(obj, "random_state"):
        obj.random_state = None
    if hasattr(obj, "estimators_"):
        for est in obj.estimators_:
            remove_random_state(est)
    return obj

model = remove_random_state(model)

# Step 3: Save cleaned model
with open("final_pipelinegit.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model cleaned and re-saved successfully for deployment.")
