from joblib import load, dump

# Load the old model trained in scikit-learn 1.7.0
pipeline = load("final_pipeline_clean.pkl")

# Re-save it using your current environment (safe for Render)
dump(pipeline, "model.pkl")

print("✅ Model re-saved successfully as model.pkl")
