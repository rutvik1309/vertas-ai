# VeritasAI - News Article Authenticity Checker

A Flask-based web application that uses machine learning to detect fake news articles and provides AI-powered reasoning for the predictions.

## Features

- Article authenticity classification (Real/Fake)
- URL-based article analysis
- File upload support for text analysis
- AI-powered reasoning using Google's Gemini API
- ChromaDB integration for context retrieval
- RESTful API endpoints

## Fixed Issues

### Version Compatibility Problems
- ✅ Fixed Python version conflicts (now using Python 3.11 consistently)
- ✅ Updated NumPy to compatible version (>=1.26.0,<2.0.0)
- ✅ Updated scikit-learn to stable version (1.5.2)
- ✅ Added missing `os` import in app.py
- ✅ Fixed security issue with hardcoded API keys

### Deployment Issues
- ✅ Standardized Python version across all configuration files
- ✅ Updated Docker configuration for better compatibility
- ✅ Added environment variable support for API keys
- ✅ Fixed package version conflicts

## Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd veritas-ai
   ```

2. **Set up Python environment**
   ```bash
   # Using pyenv
   pyenv install 3.11.9
   pyenv local 3.11.9
   
   # Or using conda
   conda create -n veritas-ai python=3.11.9
   conda activate veritas-ai
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t veritas-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 5005:5005 -e GEMINI_API_KEY=your_actual_api_key_here veritas-ai
   ```

### Render.com Deployment

1. **Set up environment variables in Render dashboard**
   - Go to your service settings
   - Add environment variable: `GEMINI_API_KEY` with your actual API key

2. **Deploy**
   - Push your changes to your repository
   - Render will automatically deploy using the `render.yaml` configuration

## API Endpoints

### GET/POST /
Main web interface for article analysis

### POST /classify
RESTful API endpoint for article classification
- **Parameters**: `url`, `text`, or `file`
- **Returns**: JSON with prediction, confidence, and AI reasoning

### GET /health
Health check endpoint

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `PORT`: Port number (default: 5005)

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'os'**
   - ✅ Fixed: Added missing import statement

2. **Version compatibility errors**
   - ✅ Fixed: Updated all package versions to compatible ones

3. **Python version conflicts**
   - ✅ Fixed: Standardized to Python 3.11.9 across all configs

4. **NumPy/scikit-learn compatibility**
   - ✅ Fixed: Updated to compatible versions

### Deployment Issues

1. **Docker build failures**
   - Ensure you're using Python 3.11 base image
   - Check that all dependencies are compatible

2. **Runtime errors**
   - Verify environment variables are set
   - Check that model files are present and accessible

3. **API key issues**
   - Use environment variables instead of hardcoded keys
   - Verify the API key is valid and has proper permissions

## File Structure

```
├── app.py                 # Main Flask application
├── features.py           # Feature extraction functions
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python runtime version
├── dockerfile           # Docker configuration
├── render.yaml          # Render.com deployment config
├── nltk_setup.py        # NLTK data download script
├── templates/           # HTML templates
│   └── index.html
├── final_pipeline_clean.pkl  # Trained ML model
├── improved_tfidf.pkl       # TF-IDF vectorizer
└── chroma_db/              # ChromaDB storage
```

## Security Notes

- API keys are now managed through environment variables
- Remove any hardcoded credentials before deployment
- Use HTTPS in production
- Implement rate limiting for API endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]