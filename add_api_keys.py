#!/usr/bin/env python3
"""
Add New API Keys Script
Adds new API keys to your existing GEMINI_API_KEYS environment variable
"""

import os
import requests

def get_current_keys():
    """Get current API keys from environment"""
    current_keys = os.environ.get("GEMINI_API_KEYS", "")
    if current_keys:
        return [key.strip() for key in current_keys.split(",") if key.strip()]
    return []

def add_new_keys():
    """Add new API keys to the existing setup"""
    print("🔑 Adding New API Keys")
    print("=" * 50)
    
    # Get current keys
    current_keys = get_current_keys()
    print(f"Current keys: {len(current_keys)}")
    
    # Get new keys from user
    print("\nEnter your 4 new API keys (one per line):")
    new_keys = []
    
    for i in range(4):
        while True:
            key = input(f"API Key {i+1}: ").strip()
            if key:
                new_keys.append(key)
                break
            else:
                print("Please enter a valid API key")
    
    # Combine all keys
    all_keys = current_keys + new_keys
    print(f"\nTotal keys: {len(all_keys)}")
    
    # Test the new keys
    print("\n🔍 Testing new API keys...")
    valid_new_keys = []
    
    for i, key in enumerate(new_keys, 1):
        print(f"Testing new key {i}: {key[:10]}...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hello")
            print(f"✅ New key {i}: Valid")
            valid_new_keys.append(key)
        except Exception as e:
            print(f"❌ New key {i}: Invalid - {str(e)[:50]}...")
    
    if not valid_new_keys:
        print("❌ No valid new API keys found")
        return
    
    # Update environment variable
    all_valid_keys = current_keys + valid_new_keys
    keys_string = ",".join(all_valid_keys)
    
    # Save to .env file
    env_content = f"GEMINI_API_KEYS={keys_string}\n"
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Successfully added {len(valid_new_keys)} new API keys")
    print(f"✅ Total API keys: {len(all_valid_keys)}")
    print(f"✅ Daily capacity: {len(all_valid_keys) * 50} requests/day")
    
    # Instructions
    print("\n📋 Next Steps:")
    print("1. Restart your Flask server")
    print("2. Test the new setup:")
    print("   curl http://localhost:5005/api/status")
    print("3. The system will automatically rotate through all keys")
    
    # Check if server is running
    print("\n🔍 Checking server status...")
    try:
        response = requests.get('http://localhost:5005/api/status', timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Server is running")
            print(f"   Total keys detected: {status['total_keys']}")
            print(f"   Available keys: {status['available_keys']}")
        else:
            print("⚠️  Server responded but status endpoint failed")
    except Exception as e:
        print(f"❌ Server not running or not accessible: {e}")
        print("   Start your server with: python app.py")

if __name__ == "__main__":
    add_new_keys() 