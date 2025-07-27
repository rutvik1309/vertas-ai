#!/usr/bin/env python3
"""
Fix NumPy compatibility for deployment environment
This script patches NumPy to be compatible with older model files
"""

import numpy as np
import sys

def fix_numpy_for_deployment():
    """Fix NumPy compatibility issues for deployment"""
    print("🔧 Fixing NumPy compatibility for deployment...")
    
    # Add missing NumPy attributes
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        print("✅ Added np.float_ = np.float64")
    
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
        print("✅ Added np.int_ = np.int64")
    
    # Create a mock _core module
    if not hasattr(np, '_core'):
        class MockCore:
            def __init__(self):
                pass
        np._core = MockCore()
        print("✅ Added np._core mock")
    
    # Add other compatibility fixes
    if not hasattr(np, 'object_'):
        np.object_ = object
        print("✅ Added np.object_ = object")
    
    if not hasattr(np, 'bool_'):
        np.bool_ = bool
        print("✅ Added np.bool_ = bool")
    
    print("✅ NumPy compatibility fixes applied")

if __name__ == "__main__":
    fix_numpy_for_deployment() 