#!/usr/bin/env python3
"""
Setup script for multiple Gemini API keys
This script helps you configure 7-8 API keys for automatic rotation
"""

import os
import sys

def setup_api_keys():
    """Interactive setup for multiple API keys"""
    print("🔑 Veritas AI - Multiple API Key Setup")
    print("=" * 50)
    print("This script will help you set up 7-8 Gemini API keys for automatic rotation.")
    print("Each key will be used intelligently based on availability and performance.")
    print()
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📁 Found existing .env file")
        with open(env_file, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""
        print(f"📁 Creating new .env file")
    
    # Collect API keys
    api_keys = []
    print("\n🔑 Please enter your Gemini API keys (up to 8 keys):")
    print("   Leave empty to finish adding keys")
    print()
    
    for i in range(1, 9):
        while True:
            key = input(f"API Key {i} (or press Enter to finish): ").strip()
            
            if not key:
                if i == 1:
                    print("❌ You must provide at least one API key!")
                    continue
                else:
                    break
            
            # Basic validation
            if len(key) < 20:
                print("❌ API key seems too short. Please check and try again.")
                continue
            
            if key in api_keys:
                print("❌ This API key has already been added. Please use a different key.")
                continue
            
            api_keys.append(key)
            print(f"✅ API Key {i} added successfully")
            break
    
    if not api_keys:
        print("❌ No API keys provided. Setup cancelled.")
        return
    
    # Build new .env content
    new_content = existing_content
    
    # Remove existing API key entries
    lines = new_content.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.startswith('GEMINI_API_KEY'):
            filtered_lines.append(line)
    
    # Add new API key entries
    for i, key in enumerate(api_keys, 1):
        filtered_lines.append(f"GEMINI_API_KEY_{i}={key}")
    
    # Add legacy single key for compatibility
    if api_keys:
        filtered_lines.append(f"GEMINI_API_KEY={api_keys[0]}")
    
    new_content = '\n'.join(filtered_lines)
    
    # Write to .env file
    with open(env_file, 'w') as f:
        f.write(new_content)
    
    print(f"\n✅ Successfully configured {len(api_keys)} API keys!")
    print(f"📁 Updated {env_file} file")
    print()
    print("🔧 Configuration Summary:")
    print(f"   • Total API keys: {len(api_keys)}")
    print(f"   • Automatic rotation: Enabled")
    print(f"   • Performance tracking: Enabled")
    print(f"   • Daily limit per key: 10,000 requests")
    print(f"   • Minute limit per key: 60 requests")
    print()
    print("🚀 Your Veritas AI system is now ready with automatic API key rotation!")
    print("   The system will intelligently distribute requests across all keys.")
    print("   You can monitor key usage at: http://127.0.0.1:10000/api/status")

if __name__ == "__main__":
    try:
        setup_api_keys()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1) 