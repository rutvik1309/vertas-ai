#!/usr/bin/env python3
"""
Gemini API Key Setup Script
Helps you manage multiple API keys for load balancing
"""

import os
import json
import requests
from datetime import datetime

def check_api_key(key):
    """Test if an API key is valid"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def test_keys(keys):
    """Test all API keys and show their status"""
    print("🔍 Testing API keys...")
    print("-" * 50)
    
    valid_keys = []
    for i, key in enumerate(keys, 1):
        print(f"Testing key {i}: {key[:10]}...")
        is_valid, message = check_api_key(key)
        
        if is_valid:
            print(f"✅ Key {i}: Valid")
            valid_keys.append(key)
        else:
            print(f"❌ Key {i}: Invalid - {message}")
        print()
    
    return valid_keys

def save_keys_to_env(keys):
    """Save API keys to .env file"""
    env_content = f"GEMINI_API_KEYS={','.join(keys)}\n"
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"✅ Saved {len(keys)} API keys to .env file")

def check_api_status():
    """Check the API status endpoint"""
    try:
        response = requests.get('http://localhost:5005/api/status')
        if response.status_code == 200:
            status = response.json()
            print("📊 API Status:")
            print(f"   Total keys: {status['total_keys']}")
            print(f"   Available keys: {status['available_keys']}")
            print(f"   Quota exceeded: {status['quota_exceeded_keys']}")
            
            print("\n🔑 Key Details:")
            for i, key_info in enumerate(status['key_details'], 1):
                print(f"   Key {i}: {key_info['key_preview']}")
                print(f"      Requests today: {key_info['requests_today']}/50")
                print(f"      Quota exceeded: {key_info['quota_exceeded']}")
        else:
            print("❌ Could not connect to API status endpoint")
    except Exception as e:
        print(f"❌ Error checking API status: {e}")

def main():
    print("🚀 Gemini API Key Setup")
    print("=" * 50)
    
    # Get API keys from user
    print("Enter your Gemini API keys (one per line, press Enter twice when done):")
    keys = []
    
    while True:
        key = input("API Key: ").strip()
        if not key:
            break
        keys.append(key)
    
    if not keys:
        print("❌ No API keys provided")
        return
    
    print(f"\n📝 Testing {len(keys)} API keys...")
    valid_keys = test_keys(keys)
    
    if not valid_keys:
        print("❌ No valid API keys found")
        return
    
    print(f"✅ Found {len(valid_keys)} valid API keys")
    
    # Save to .env file
    save_keys_to_env(valid_keys)
    
    # Instructions
    print("\n📋 Next Steps:")
    print("1. Restart your Flask server")
    print("2. Test the load balancing:")
    print("   curl http://localhost:5005/api/status")
    print("3. Try making some predictions to see key rotation in action")
    
    # Check if server is running
    print("\n🔍 Checking if server is running...")
    check_api_status()

if __name__ == "__main__":
    main() 