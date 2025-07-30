#!/usr/bin/env python3

import joblib
import warnings

def fix_model_version():
    """Re-save the model with current scikit-learn version for deployment compatibility"""
    try:
        print("🔄 Loading original model...")
        # Suppress version warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = joblib.load("final_pipeline_clean.pkl")
        
        print(f"✅ Original model loaded successfully")
        print(f"📊 Model type: {type(pipeline)}")
        print(f"🔧 Has predict method: {hasattr(pipeline, 'predict')}")
        print(f"🔧 Has predict_proba method: {hasattr(pipeline, 'predict_proba')}")
        
        # Test the model
        test_text = "This is a test article for fake news detection."
        prediction = pipeline.predict([test_text])
        print(f"✅ Test prediction successful: {prediction}")
        
        # Re-save with current scikit-learn version
        print("🔄 Re-saving model with current scikit-learn version...")
        joblib.dump(pipeline, "final_pipeline_clean_fixed.pkl")
        
        print("✅ Model re-saved successfully as final_pipeline_clean_fixed.pkl")
        
        # Verify the re-saved model
        print("🔄 Verifying re-saved model...")
        test_pipeline = joblib.load("final_pipeline_clean_fixed.pkl")
        test_prediction = test_pipeline.predict([test_text])
        print(f"✅ Re-saved model test successful: {test_prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing model version: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing model version compatibility...")
    success = fix_model_version()
    if success:
        print("🎉 Model version fix completed successfully!")
    else:
        print("❌ Model version fix failed!") 