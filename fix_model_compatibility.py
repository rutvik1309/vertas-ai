#!/usr/bin/env python3
"""
Fix NumPy compatibility for existing model files
This script creates a compatible version of the model for Python 3.11
"""

import pickle
import numpy as np
import os

def fix_numpy_compatibility():
    """Fix NumPy compatibility issues"""
    print("🔧 Fixing NumPy compatibility...")
    
    # Add missing NumPy attributes for compatibility
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        print("✅ Added np.float_ = np.float64")
    
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
        print("✅ Added np.int_ = np.int64")
    
    # Add other compatibility fixes
    if not hasattr(np, '_core'):
        # Create a mock _core module if needed
        class MockCore:
            pass
        np._core = MockCore()
        print("✅ Added np._core mock")

def load_and_resave_model():
    """Load the original model and resave it with current NumPy version"""
    try:
        print("📂 Loading original model...")
        
        # Fix compatibility first
        fix_numpy_compatibility()
        
        # Try to load the original model
        with open("final_pipeline_clean.pkl", "rb") as f:
            model = pickle.load(f)
        
        print("✅ Original model loaded successfully!")
        
        # Resave with current NumPy version
        print("💾 Resaving model with current NumPy version...")
        with open("final_pipeline_clean_fixed.pkl", "wb") as f:
            pickle.dump(model, f)
        
        print("✅ Model resaved as 'final_pipeline_clean_fixed.pkl'")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_loading():
    """Test if the fixed model can be loaded"""
    try:
        print("🧪 Testing fixed model loading...")
        
        fix_numpy_compatibility()
        
        with open("final_pipeline_clean_fixed.pkl", "rb") as f:
            model = pickle.load(f)
        
        print("✅ Fixed model loads successfully!")
        
        # Test prediction
        test_text = ["This is a test article for prediction."]
        prediction = model.predict(test_text)
        print(f"✅ Test prediction works: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting model compatibility fix...")
    print(f"Current NumPy version: {np.__version__}")
    
    # Check if original model exists
    if not os.path.exists("final_pipeline_clean.pkl"):
        print("❌ Original model file 'final_pipeline_clean.pkl' not found!")
        return
    
    # Load and resave the model
    if load_and_resave_model():
        # Test the fixed model
        if test_model_loading():
            print("🎉 Model compatibility fix completed successfully!")
            print("You can now use 'final_pipeline_clean_fixed.pkl' in your app.")
        else:
            print("❌ Model fix failed during testing.")
    else:
        print("❌ Failed to load and resave the model.")

if __name__ == "__main__":
    main() 