#!/usr/bin/env python3

import joblib
import sys

def test_model_loading():
    """Test if the model can be loaded and used"""
    try:
        print("🔄 Testing model loading...")
        model = joblib.load("final_pipeline_clean_fixed.pkl")
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model type: {type(model)}")
        print(f"🔧 Has predict method: {hasattr(model, 'predict')}")
        print(f"🔧 Has predict_proba method: {hasattr(model, 'predict_proba')}")
        
        # Test with a simple input
        test_text = "This is a test article for fake news detection."
        try:
            prediction = model.predict([test_text])
            print(f"✅ Prediction test successful: {prediction}")
            return True
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print(f"🐍 Python version: {sys.version}")
    success = test_model_loading()
    if success:
        print("🎉 Model compatibility test PASSED")
    else:
        print("❌ Model compatibility test FAILED") 