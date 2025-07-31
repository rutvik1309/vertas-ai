#!/usr/bin/env python3
"""
API Key Rotation Monitor
Monitor the performance and usage of your multiple Gemini API keys
"""

import requests
import json
import time
import os
from datetime import datetime

def get_api_status():
    """Get API key status from the server"""
    try:
        response = requests.get('http://127.0.0.1:10000/api/status', timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the Flask app is running on port 10000")
        return None
    except Exception as e:
        print(f"❌ Error getting API status: {e}")
        return None

def display_status(status):
    """Display API key status in a formatted way"""
    if not status:
        return
    
    print("\n" + "="*80)
    print(f"🔑 VERITAS AI - API KEY ROTATION STATUS")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\n📊 OVERALL STATUS:")
    print(f"   • Total API Keys: {status['total_keys']}")
    print(f"   • Available Keys: {status['available_keys']}")
    print(f"   • Exhausted Keys: {status['quota_exceeded_keys']}")
    print(f"   • Rotation Enabled: {'✅ Yes' if status.get('rotation_enabled', False) else '❌ No'}")
    
    print(f"\n🔑 INDIVIDUAL KEY STATUS:")
    print("-" * 80)
    
    for key_info in status['key_details']:
        key_num = key_info['key_number']
        requests_today = key_info['requests_today']
        requests_per_minute = key_info['requests_per_minute']
        success_rate = key_info['success_rate']
        errors = key_info['errors']
        hours_since_reset = key_info['hours_since_reset']
        quota_exceeded = key_info['quota_exceeded']
        
        # Status indicator
        if quota_exceeded:
            status_indicator = "❌ EXHAUSTED"
        elif requests_per_minute >= 60:
            status_indicator = "⚠️  MINUTE LIMIT"
        elif requests_today >= 10000:
            status_indicator = "⚠️  DAILY LIMIT"
        elif success_rate < 0.5:
            status_indicator = "⚠️  LOW SUCCESS"
        else:
            status_indicator = "✅ AVAILABLE"
        
        print(f"Key {key_num:2d}: {status_indicator}")
        print(f"      Requests Today: {requests_today:5d} / {key_info['daily_limit']:5d}")
        print(f"      Requests/Min:   {requests_per_minute:5d} / {key_info['minute_limit']:5d}")
        print(f"      Success Rate:   {success_rate:.1%}")
        print(f"      Errors:         {errors:3d}")
        print(f"      Hours Since Reset: {hours_since_reset:5.1f}")
        print()

def monitor_continuously(interval=30):
    """Monitor API keys continuously"""
    print(f"🔍 Starting continuous monitoring (refresh every {interval} seconds)")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            status = get_api_status()
            display_status(status)
            
            if status and status['available_keys'] == 0:
                print("🚨 WARNING: All API keys are currently exhausted!")
                print("   Consider adding more API keys or waiting for quota reset.")
            
            print(f"⏰ Next update in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")

def main():
    """Main function"""
    print("🔑 Veritas AI - API Key Rotation Monitor")
    print("=" * 50)
    
    # Check if server is running
    print("🔍 Checking server connection...")
    status = get_api_status()
    
    if not status:
        print("\n❌ Cannot connect to Veritas AI server.")
        print("   Make sure the Flask app is running:")
        print("   python app.py")
        return
    
    print("✅ Server connection successful!")
    
    # Display current status
    display_status(status)
    
    # Ask user what they want to do
    print("\n📋 What would you like to do?")
    print("1. View current status (one-time)")
    print("2. Monitor continuously (refresh every 30 seconds)")
    print("3. Monitor continuously (refresh every 60 seconds)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            status = get_api_status()
            display_status(status)
            break
        elif choice == "2":
            monitor_continuously(30)
            break
        elif choice == "3":
            monitor_continuously(60)
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 