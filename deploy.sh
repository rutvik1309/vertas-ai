#!/bin/bash

# VeritasAI Deployment Script
# This script helps deploy the application with proper error checking

set -e  # Exit on any error

echo "🚀 Starting VeritasAI Deployment..."

# Check if Python version is correct
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o '3\.[0-9]\+')
if [[ "$python_version" != "3.11" ]]; then
    echo "⚠️  Warning: Python version is $python_version, but 3.11 is recommended"
    echo "   Consider using: pyenv local 3.11.9"
fi

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo "❗ Please edit .env file with your actual API keys before running the application"
fi

# Download NLTK data
echo "📚 Setting up NLTK data..."
python nltk_setup.py

# Check if model files exist
echo "🧠 Checking model files..."
required_files=("final_pipeline_clean.pkl" "improved_tfidf.pkl" "tfidf_vectorizer.pkl")
missing_files=()

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "❗ Missing model files:"
    printf '   - %s\n' "${missing_files[@]}"
    echo "   Please ensure these files are present before deployment"
fi

# Test import of main modules
echo "🧪 Testing critical imports..."
python3 -c "
import sys
try:
    import flask
    import numpy
    import sklearn
    import nltk
    import google.generativeai
    import chromadb
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo "✅ Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Ensure your .env file has the correct GEMINI_API_KEY"
echo "2. Run: python app.py (for local development)"
echo "3. Or build Docker: docker build -t veritas-ai ."
echo ""
echo "For production deployment:"
echo "- Render.com: Push to your repository"
echo "- Docker: docker run -p 5005:5005 --env-file .env veritas-ai"