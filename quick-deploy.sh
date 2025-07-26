#!/bin/bash

echo "🚀 Quick Deploy - Veritas AI"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed"
    exit 1
fi

echo "✅ Dependencies check passed"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file..."
    cat > .env << EOF
FLASK_SECRET_KEY=$(openssl rand -hex 32)
GEMINI_API_KEYS=your_api_key_here
FLASK_ENV=development
DATABASE_URL=sqlite:///veritas_ai.db
EOF
    echo "⚠️  Please update .env file with your actual Gemini API keys"
fi

# Create necessary directories
mkdir -p chroma_db
mkdir -p static/uploads

echo "✅ Setup complete!"
echo ""
echo "🌐 To start the web application:"
echo "   python3 app.py"
echo ""
echo "🔌 To load the Chrome extension:"
echo "   1. Open Chrome and go to chrome://extensions/"
echo "   2. Enable 'Developer mode'"
echo "   3. Click 'Load unpacked'"
echo "   4. Select the 'extension' folder"
echo ""
echo "📱 Web app will be available at: http://localhost:5005" 