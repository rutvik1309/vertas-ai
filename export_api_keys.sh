#!/bin/bash

# Export 8 Gemini API keys for automatic rotation
echo "🔑 Setting up 8 Gemini API keys for automatic rotation..."

export GEMINI_API_KEY_1="AIzaSyA8OwqMsr7yPQJsJLqCemJ9J1WgW1h-N-g"
export GEMINI_API_KEY_2="AIzaSyDxuojuzuDy_RMU2zvMxDXAeTL8dXgF9Eo"
export GEMINI_API_KEY_3="AIzaSyCH4piagASkdT4laExwsXZMIDMAbi4j5Js"
export GEMINI_API_KEY_4="AIzaSyCJ16lvreTvm8VlTXoyhrCHUcm8Zzdutxo"
export GEMINI_API_KEY_5="AIzaSyBjt3A8RQD68OM9Um33bX29HCmbLiAFkKc"
export GEMINI_API_KEY_6="AIzaSyAWsj9zTGBXHuDY6ZFGUVkyCdyRPHD74D0"
export GEMINI_API_KEY_7="AIzaSyBECr9jMJ-QkyIdv76kPntp23Dl4NRI0pI"
export GEMINI_API_KEY_8="AIzaSyA7APWpWr4LizACI9OBsJyunrVSnYkFNaA"

# Legacy key for compatibility
export GEMINI_API_KEY="AIzaSyA8OwqMsr7yPQJsJLqCemJ9J1WgW1h-N-g"

echo "✅ Successfully exported 8 API keys!"
echo "🔧 Configuration Summary:"
echo "   • Total API keys: 8"
echo "   • Automatic rotation: Enabled"
echo "   • Performance tracking: Enabled"
echo "   • Daily limit per key: 10,000 requests"
echo "   • Minute limit per key: 60 requests"
echo ""
echo "🚀 To start the server with these keys, run:"
echo "   source export_api_keys.sh && python3 app.py"
echo ""
echo "📊 To monitor key usage, run:"
echo "   python3 monitor_api_rotation.py" 