#!/usr/bin/env python3
"""
API Usage Monitor
Shows real-time API key usage and automatic switching
"""

import requests
import time
import json
from datetime import datetime

def monitor_api_usage():
    """Monitor API key usage in real-time"""
    print("📊 API Usage Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            # Get API status
            try:
                response = requests.get('http://localhost:5005/api/status', timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    
                    # Clear screen (works on most terminals)
                    print("\033[2J\033[H", end="")
                    
                    print(f"📊 API Status - {datetime.now().strftime('%H:%M:%S')}")
                    print("=" * 50)
                    print(f"Total Keys: {status['total_keys']}")
                    print(f"Available Keys: {status['available_keys']}")
                    print(f"Quota Exceeded: {status['quota_exceeded_keys']}")
                    print(f"Daily Capacity: {status['total_keys'] * 50} requests")
                    print()
                    
                    print("🔑 Key Details:")
                    print("-" * 30)
                    for i, key_info in enumerate(status['key_details'], 1):
                        status_icon = "✅" if not key_info['quota_exceeded'] else "❌"
                        print(f"{status_icon} Key {i}: {key_info['key_preview']}")
                        print(f"   Requests: {key_info['requests_today']}/50")
                        print(f"   Status: {'Available' if not key_info['quota_exceeded'] else 'Quota Exceeded'}")
                        print()
                    
                    # Show current active key
                    if status['available_keys'] > 0:
                        print("🔄 Automatic Switching: ACTIVE")
                        print("   System will automatically rotate through available keys")
                    else:
                        print("⚠️  All keys quota exceeded!")
                        print("   Wait for daily reset or add more keys")
                    
                else:
                    print("❌ Could not get API status")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Server not accessible: {e}")
                print("   Make sure your Flask server is running: python app.py")
                break
            
            # Wait 5 seconds before next check
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")

def test_key_rotation():
    """Test the key rotation by making multiple requests"""
    print("🔄 Testing Key Rotation")
    print("=" * 30)
    
    # Make a few test requests to see key rotation
    for i in range(5):
        try:
            print(f"Making request {i+1}...")
            response = requests.post('http://localhost:5005/', 
                                   data={'article_text': f'Test article {i+1}'},
                                   headers={'X-Requested-With': 'XMLHttpRequest'},
                                   timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Request {i+1}: Success")
            else:
                print(f"❌ Request {i+1}: Failed - {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request {i+1}: Error - {e}")
        
        time.sleep(2)  # Wait between requests
    
    print("\n🔍 Check the API status to see key rotation:")
    print("   curl http://localhost:5005/api/status")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_key_rotation()
    else:
        monitor_api_usage() 