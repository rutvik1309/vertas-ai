#!/usr/bin/env python3
"""
Setup environment variables for the 8 Gemini API keys
"""

import os

def setup_env_keys():
    """Set up the environment variables for API keys"""
    
    # The 8 API keys provided by the user
    api_keys = [
        "AIzaSyA8OwqMsr7yPQJsJLqCemJ9J1WgW1h-N-g",
        "AIzaSyDxuojuzuDy_RMU2zvMxDXAeTL8dXgF9Eo", 
        "AIzaSyCH4piagASkdT4laExwsXZMIDMAbi4j5Js",
        "AIzaSyCJ16lvreTvm8VlTXoyhrCHUcm8Zzdutxo",
        "AIzaSyBjt3A8RQD68OM9Um33bX29HCmbLiAFkKc",
        "AIzaSyAWsj9zTGBXHuDY6ZFGUVkyCdyRPHD74D0",
        "AIzaSyBECr9jMJ-QkyIdv76kPntp23Dl4NRI0pI",
        "AIzaSyA7APWpWr4LizACI9OBsJyunrVSnYkFNaA"
    ]
    
    print("🔑 Setting up 8 Gemini API keys for automatic rotation...")
    
    # Set environment variables
    for i, key in enumerate(api_keys, 1):
        env_var = f"GEMINI_API_KEY_{i}"
        os.environ[env_var] = key
        print(f"✅ Set {env_var}")
    
    # Set the legacy single key for compatibility
    os.environ["GEMINI_API_KEY"] = api_keys[0]
    print(f"✅ Set GEMINI_API_KEY (legacy)")
    
    print(f"\n🎉 Successfully configured {len(api_keys)} API keys!")
    print("🔧 Configuration Summary:")
    print(f"   • Total API keys: {len(api_keys)}")
    print(f"   • Automatic rotation: Enabled")
    print(f"   • Performance tracking: Enabled")
    print(f"   • Daily limit per key: 10,000 requests")
    print(f"   • Minute limit per key: 60 requests")
    
    return api_keys

if __name__ == "__main__":
    setup_env_keys() 