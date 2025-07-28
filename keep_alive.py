#!/usr/bin/env python3
"""
Keep Alive Script for VeritasAI
This script makes periodic requests to keep the service alive
"""

import requests
import time
import os
from datetime import datetime

def keep_alive():
    """Make periodic requests to keep the service alive"""
    # Get the service URL from environment or use default
    service_url = os.environ.get('SERVICE_URL', 'https://vertas-ai.onrender.com')
    
    print(f"🔄 Starting keep-alive for {service_url}")
    print(f"⏰ Started at: {datetime.now()}")
    
    while True:
        try:
            # Make a request to the health endpoint
            response = requests.get(f"{service_url}/health", timeout=30)
            
            if response.status_code == 200:
                print(f"✅ {datetime.now()} - Service is alive (Status: {response.status_code})")
            else:
                print(f"⚠️  {datetime.now()} - Service responded with status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {datetime.now()} - Error keeping service alive: {e}")
        
        # Wait for 10 minutes before next request
        print(f"⏳ Waiting 10 minutes before next request...")
        time.sleep(600)  # 10 minutes

if __name__ == "__main__":
    keep_alive() 