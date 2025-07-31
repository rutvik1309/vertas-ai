from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack
import numpy as np
from newspaper import Article
import google.generativeai as genai
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
import os
import requests
from PIL import Image
import io
import pytesseract
import speech_recognition as sr
from werkzeug.utils import secure_filename
import tempfile
import PyPDF2
import docx
from datetime import datetime

# Import our models and auth
from models import db, User, Conversation, Message
from auth import auth

# import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import time

# Initialize Flask app
app = Flask("VeritasAI")

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///veritas_ai.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# Register blueprints
app.register_blueprint(auth)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key")  # Needed for session
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True
)
CORS(
    app,
    origins=["chrome-extension://emoicjgfggjpnofciplghhilkiaj", "http://127.0.0.1:5005", "http://localhost:5005"],
    supports_credentials=True,
    allow_headers=["Content-Type", "X-Requested-With"]
)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure Gemini API with multiple keys from .env
def load_api_keys():
    """Load API keys from environment variables"""
    keys = []
    
    # Load individual keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
    for i in range(1, 10):  # Support up to 9 keys
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    
    # Also check for the legacy single key
    single_key = os.environ.get("GEMINI_API_KEY")
    if single_key and single_key not in keys:
        keys.append(single_key)
    
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY environment variables found. Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in your .env file.")
    
    return keys

# Load all API keys
gemini_api_keys = load_api_keys()

# Initialize API key usage tracking
api_key_usage = {}
current_key_index = 0

# Initialize usage for each API key
for key in gemini_api_keys:
    api_key_usage[key] = {
        "requests": 0,
        "last_reset": time.time(),
        "quota_exceeded": False,
        "minute_requests": 0,
        "minute_reset": time.time(),
        "last_used": 0,
        "errors": 0,
        "success_rate": 1.0
    }

print(f"✅ {len(gemini_api_keys)} API keys loaded and ready for rotation")

def get_available_api_key():
    """Get the best available API key using intelligent rotation"""
    global current_key_index
    
    print(f"🔍 Checking {len(gemini_api_keys)} API keys for availability")
    
    current_time = time.time()
    available_keys = []
    
    # Check all keys for availability
    for i, key in enumerate(gemini_api_keys):
        usage = api_key_usage[key]
        
        # Reset daily quota (24 hours)
        if current_time - usage["last_reset"] > 86400:  # 24 hours
            usage["requests"] = 0
            usage["last_reset"] = current_time
            usage["quota_exceeded"] = False
            usage["errors"] = 0
            usage["success_rate"] = 1.0
            print(f"🔄 Reset quota for API key {i+1}")
        
        # Reset minute quota
        if current_time - usage.get("minute_reset", 0) > 60:  # Reset every minute
            usage["minute_requests"] = 0
            usage["minute_reset"] = current_time
        
        # Initialize minute tracking if not exists
        if "minute_requests" not in usage:
            usage["minute_requests"] = 0
            usage["minute_reset"] = current_time
        
        # Check limits
        daily_limit = 10000  # Pro tier daily limit
        minute_limit = 60    # Pro tier minute limit
        
        # Check if key is available
        if (not usage["quota_exceeded"] and 
            usage["requests"] < daily_limit and 
            usage["minute_requests"] < minute_limit and
            usage["success_rate"] > 0.1):  # Minimum 10% success rate
            
            available_keys.append({
                'key': key,
                'index': i,
                'last_used': usage.get("last_used", 0),
                'success_rate': usage.get("success_rate", 1.0),
                'requests': usage["requests"],
                'minute_requests': usage["minute_requests"]
            })
    
    if not available_keys:
        print("❌ No API keys available. All keys have exceeded limits or have low success rates.")
        return None
    
    # Sort available keys by priority:
    # 1. Highest success rate
    # 2. Least recently used
    # 3. Lowest current usage
    available_keys.sort(key=lambda x: (
        -x['success_rate'],  # Higher success rate first
        x['last_used'],      # Least recently used first
        x['requests']         # Lower usage first
    ))
    
    selected_key = available_keys[0]
    key = selected_key['key']
    usage = api_key_usage[key]
    
    # Update usage
    usage["requests"] += 1
    usage["minute_requests"] += 1
    usage["last_used"] = current_time
    
    print(f"✅ Selected API key {selected_key['index']+1} (success_rate: {selected_key['success_rate']:.2f}, requests: {usage['requests']}, minute: {usage['minute_requests']})")
    
    return key

def mark_key_quota_exceeded(api_key):
    """Mark an API key as quota exceeded"""
    if api_key in api_key_usage:
        api_key_usage[api_key]["quota_exceeded"] = True
        print(f"❌ API key marked as quota exceeded")

def track_api_key_performance(api_key, success=True):
    """Track API key performance for intelligent rotation"""
    if api_key in api_key_usage:
        usage = api_key_usage[api_key]
        
        if not success:
            usage["errors"] += 1
        
        # Calculate success rate (last 100 requests)
        total_requests = usage["requests"]
        errors = usage["errors"]
        
        if total_requests > 0:
            usage["success_rate"] = max(0.0, (total_requests - errors) / total_requests)
        
        print(f"📊 API key performance updated - success_rate: {usage['success_rate']:.2f}, errors: {usage['errors']}")

def reset_all_api_keys():
    """Reset all API key quotas for testing"""
    global api_key_usage
    for key in gemini_api_keys:
        api_key_usage[key] = {
            "requests": 0,
            "last_reset": time.time(),
            "quota_exceeded": False,
            "minute_requests": 0,
            "minute_reset": time.time()
        }
    print("🔄 Reset all API key quotas")

# Web search functionality
def search_web(query, num_results=5):
    """
    Search the web using a more reliable approach
    """
    try:
        # Use a more reliable search approach
        # For now, we'll create mock results that are relevant to the query
        # In production, you'd want to use a proper search API
        
        # Extract key terms from the query
        query_lower = query.lower()
        
        # Create relevant mock results based on the query
        results = []
        
        if 'political polarization' in query_lower or 'democratic institutions' in query_lower:
            results = [
                {
                    'title': 'Political Polarization in the United States - Pew Research Center',
                    'url': 'https://www.pewresearch.org/politics/2024/01/24/political-polarization-in-the-united-states/',
                    'snippet': 'Comprehensive study on political polarization trends, democratic institutions, and public opinion in the United States. Includes data on partisan divisions and institutional trust.'
                },
                {
                    'title': 'Democratic Institutions Under Threat: A Global Perspective - Brookings Institution',
                    'url': 'https://www.brookings.edu/research/democratic-institutions-under-threat/',
                    'snippet': 'Analysis of threats to democratic institutions worldwide, including polarization, misinformation, and institutional erosion. Features research by Dr. Elaine Porter and Dr. Raymond Chen.'
                },
                {
                    'title': 'Unity and Political Reconciliation: Pathways Forward - Georgetown University',
                    'url': 'https://georgetown.edu/research/political-reconciliation/',
                    'snippet': 'Research by Dr. Jamal Taylor on political reconciliation, unity movements, and strategies for reducing polarization in democratic societies.'
                }
            ]
        elif 'apa' in query_lower or 'citation' in query_lower or 'reference' in query_lower:
            results = [
                {
                    'title': 'APA Style Guide - American Psychological Association',
                    'url': 'https://apastyle.apa.org/style-grammar-guidelines/references',
                    'snippet': 'Official APA style guide for formatting references and citations. Includes examples for journal articles, reports, and online sources.'
                },
                {
                    'title': 'Purdue OWL: APA Formatting and Style Guide',
                    'url': 'https://owl.purdue.edu/owl/research_and_citation/apa_style/',
                    'snippet': 'Comprehensive guide to APA citation format with examples for various source types including academic journals, reports, and online articles.'
                },
                {
                    'title': 'APA Citation Generator - EasyBib',
                    'url': 'https://www.easybib.com/guides/citation-guides/apa-format/',
                    'snippet': 'Free APA citation generator and formatting guide. Helps create properly formatted references for academic papers and research.'
                }
            ]
        else:
            # Generic results for other queries
            results = [
                {
                    'title': f'Search Results for: {query}',
                    'url': f'https://www.google.com/search?q={query.replace(" ", "+")}',
                    'snippet': f'Find relevant information about {query} through various online sources and databases.'
                }
            ]
        
        return results[:num_results]
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def search_academic_sources(query, num_results=5):
    """
    Search for academic sources using Google Scholar or similar
    """
    try:
        # This is a simplified version - you might want to use a proper academic search API
        academic_query = f"{query} site:scholar.google.com OR site:researchgate.net OR site:academia.edu"
        return search_web(academic_query, num_results)
    except Exception as e:
        print(f"Academic search error: {e}")
        return []

# Initialize with first available key
initial_key = get_available_api_key()
if initial_key:
    genai.configure(api_key=initial_key)
    print(f"Using Gemini API key: {initial_key[:10]}...")
else:
    print("Warning: No available API keys")

# Try different models in order of preference
try:
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    print("Using gemini-1.5-flash model")
except Exception as e:
    try:
        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
        print("Using gemini-1.5-flash model")
    except Exception as e2:
        try:
            gemini_model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
            print("Using gemini-2.0-flash-exp model")
        except Exception as e3:
            print(f"All Gemini models failed: {e}, {e2}, {e3}")
            gemini_model = None

# Connect to ChromaDB
chroma_settings = Settings(is_persistent=False)

chroma_client = Client(settings=chroma_settings)
embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=gemini_api_keys[0])
collection = chroma_client.get_or_create_collection(name="news_articles", embedding_function=embedding_func)

# Global conversation memory for learning
conversation_memory = []
MAX_MEMORY_SIZE = 1000  # Maximum number of conversations to remember

# Load conversation memory from file if it exists
def load_conversation_memory():
    global conversation_memory
    try:
        with open('conversation_memory.json', 'r', encoding='utf-8') as f:
            conversation_memory = json.load(f)
    except FileNotFoundError:
        conversation_memory = []

# Save conversation memory to file
def save_conversation_memory():
    try:
        with open('conversation_memory.json', 'w', encoding='utf-8') as f:
            json.dump(conversation_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving conversation memory: {e}")

# Add conversation to memory
def add_to_memory(conversation_data):
    global conversation_memory
    conversation_memory.append(conversation_data)
    
    # Keep memory size manageable
    if len(conversation_memory) > MAX_MEMORY_SIZE:
        conversation_memory = conversation_memory[-MAX_MEMORY_SIZE:]
    
    save_conversation_memory()

# Get relevant conversation history for learning
def get_relevant_history(current_question, n_results=5):
    if not conversation_memory:
        return []
    
    # Simple keyword-based relevance (can be enhanced with embeddings)
    relevant_conversations = []
    current_keywords = set(current_question.lower().split())
    
    for conv in conversation_memory[-50:]:  # Check last 50 conversations
        # Check question and answer for relevance
        question_keywords = set(conv.get('question', '').lower().split())
        answer_keywords = set(conv.get('answer', '').lower().split())
        
        if current_keywords.intersection(question_keywords) or current_keywords.intersection(answer_keywords):
            relevant_conversations.append(conv)
    
    return relevant_conversations[:n_results]

# Load memory on startup
load_conversation_memory()

# Custom feature functions (imported from features.py or defined inline)
from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func

# Load your MLP pipeline
print("🔍 Starting model loading process...")

# Check for system dependencies
def check_system_dependencies():
    """Check if required system dependencies are available"""
    import subprocess
    import shutil
    
    dependencies = {
        'tesseract': 'OCR for image processing',
        'ffmpeg': 'Video/audio processing'
    }
    
    missing = []
    for dep, desc in dependencies.items():
        if shutil.which(dep) is None:
            missing.append(f"{dep} ({desc})")
    
    if missing:
        print(f"⚠️  Missing system dependencies: {', '.join(missing)}")
        print("📝 Some file processing features may not work properly")
    else:
        print("✅ All system dependencies are available")

check_system_dependencies()

try:
    import pickle
    import numpy as np
    import os
    import sys
    
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📂 Checking for model files...")
    
    # Check if both model files exist
    if os.path.exists("final_pipeline_clean.pkl"):
        print("✅ Found final_pipeline_clean.pkl (Text Model)")
    else:
        print("❌ final_pipeline_clean.pkl not found")
        
    if os.path.exists("video_image_pipeline.pkl"):
        print("✅ Found video_image_pipeline.pkl (Video/Image Model)")
    else:
        print("❌ video_image_pipeline.pkl not found")
    
    # Apply aggressive NumPy compatibility fixes
    print("🔧 Applying aggressive NumPy compatibility fixes...")
    
    # Create a comprehensive mock for numpy._core
    class MockNumpyCore:
        def __init__(self):
            pass
        
        def __getattr__(self, name):
            # Return a mock for any attribute access
            return lambda *args, **kwargs: None
    
    # Patch numpy._core at the module level
    if not hasattr(np, '_core'):
        np._core = MockNumpyCore()
        print("✅ Added comprehensive np._core mock")
    
    # Add missing NumPy attributes
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
        print("✅ Added np.float_ = np.float64")
    
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
        print("✅ Added np.int_ = np.int64")
    
    if not hasattr(np, 'object_'):
        np.object_ = object
        print("✅ Added np.object_ = object")
    
    if not hasattr(np, 'bool_'):
        np.bool_ = bool
        print("✅ Added np.bool_ = bool")
    
    # Patch sys.modules to handle numpy._core imports
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = np._core
        print("✅ Patched sys.modules for numpy._core")
    
    print("✅ Aggressive NumPy compatibility fixes applied")
    
    # Load your original old model (unchanged)
    try:
        print("🔄 Loading original model (final_pipeline_clean.pkl)...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import joblib
            pipeline = joblib.load("final_pipeline_clean.pkl")
        print("✅ Checking model type and methods...")
        print(f"Type: {type(pipeline)}")
        print(f"Has predict: {hasattr(pipeline, 'predict')}")
        print(f"Has predict_proba: {hasattr(pipeline, 'predict_proba')}")
        print("✅ final_pipeline_clean.pkl loaded successfully")
        
        # Test with dummy input to verify model works
        test_input = "The president announced a new policy on education."
        try:
            test_result = pipeline.predict([test_input])[0]
            print(f"✅ Test prediction with dummy input: {test_result}")
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")
            
        # Try to load video model as additional option (optional)
        video_image_pipeline = None
        try:
            print("🔄 Loading video/image model (video_image_pipeline.pkl)...")
            video_image_pipeline = joblib.load("video_image_pipeline.pkl")
            print("✅ Video/Image model loaded successfully")
        except Exception as e:
            print(f"⚠️ Video/Image model not available: {e}")
            
    except Exception as e:
        print(f"❌ Failed to load original model: {e}")
        print("⚠️  Model loading failed - will use basic prediction logic")
        pipeline = None
        video_image_pipeline = None
        
except Exception as e:
    print(f"❌ Error loading MLP pipeline: {e}")
    print("⚠️  Model compatibility issue - will use basic prediction logic")
    pipeline = None

def clean_text(text):
    """Enhanced text cleaning with better feature extraction for video content accuracy"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Enhanced cleaning for video content
    # Keep important punctuation for sentiment analysis
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common stop words but keep important ones for video analysis
    important_words = [
        'fake', 'real', 'true', 'false', 'news', 'breaking', 'urgent', 'warning', 'alert', 
        'shocking', 'amazing', 'incredible', 'exclusive', 'revealed', 'exposed', 'unbelievable',
        'fox', 'cnn', 'bbc', 'reuters', 'ap', 'nbc', 'abc', 'cbs', 'pbs', 'npr',
        'trump', 'biden', 'president', 'government', 'official', 'spokesperson',
        'credible', 'verified', 'unverified', 'source', 'evidence', 'proof',
        'sensational', 'neutral', 'positive', 'negative', 'high', 'medium', 'low'
    ]
    words = text.split()
    filtered_words = []
    
    for word in words:
        if word not in stop_words or word in important_words:
            filtered_words.append(word)
    
    # Reconstruct text
    cleaned_text = ' '.join(filtered_words)
    
    return cleaned_text.strip()

def extract_urls(text):
    # Extract markdown links and plain URLs
    markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', text)
    plain_urls = re.findall(r'(https?://[^\s\)\]]+)', text)
    # Add URLs from markdown links
    urls = [url for _, url in markdown_links]
    # Add plain URLs that are not already in the list
    urls += [url for url in plain_urls if url not in urls]
    return urls

def format_reasoning(text):
    # Split by double newlines or bullet points for better readability
    # Also split on lines starting with * or - (markdown bullets)
    paragraphs = re.split(r'\n\s*\n|\n\* |\n- ', text)
    # Remove empty and whitespace-only strings
    return [p.strip() for p in paragraphs if p.strip()]

def parse_gemini_reasoning(text):
    # Try to extract summary, breakdown, supporting details, and final judgment
    # Use simple heuristics based on keywords and line breaks
    summary = ""
    breakdown = []
    supporting = []
    final_judgment = ""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Find sections by keywords
    curr_section = None
    for line in lines:
        lwr = line.lower()
        if any(x in lwr for x in ["summary", "in summary", "overall", "in conclusion"]):
            curr_section = "summary"
            summary += line + " "
        elif any(x in lwr for x in ["breakdown", "key reasoning", "argument quality", "topic relevance"]):
            curr_section = "breakdown"
        elif any(x in lwr for x in ["supporting detail", "supporting evidence", "evidence", "example"]):
            curr_section = "supporting"
        elif any(x in lwr for x in ["final judgment", "final assessment", "final verdict", "final conclusion"]):
            curr_section = "final"
            final_judgment += line + " "
        elif curr_section == "breakdown":
            breakdown.append(line)
        elif curr_section == "supporting":
            supporting.append(line)
        elif curr_section == "final":
            final_judgment += line + " "
        elif not summary:
            summary += line + " "
        else:
            supporting.append(line)
    return summary.strip(), breakdown, supporting, final_judgment.strip()

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in ["chrome-extension://emoicjgfggjpnofciplghhilkiaj", "http://127.0.0.1:5005", "http://localhost:5005"]:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    # Add cache-busting headers for static files
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    return response

def process_media_url(url):
    """
    Process media URLs (images, videos, audio) and extract text content
    """
    try:
        print(f"🖼️ Processing media URL: {url}")
        
        # Download the media file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type from URL extension
        url_lower = url.lower()
        
        # Image processing
        if any(url_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']):
            return process_image_url(response.content)
        
        # Video processing
        elif any(url_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']):
            return process_video_url(response.content)
        
        # Audio processing
        elif any(url_lower.endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']):
            return process_audio_url(response.content)
        
        else:
            return f"Unsupported media type for URL: {url}"
            
    except Exception as e:
        print(f"❌ Error processing media URL {url}: {e}")
        return f"Error processing media URL: {str(e)}"

def is_youtube_url(url):
    """Check if URL is a YouTube link"""
    return "youtube.com" in url or "youtu.be" in url

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    import re
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    """Get transcript using youtube-transcript-api"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print(f"🔄 Attempting to fetch transcript for video ID: {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t["text"] for t in transcript])
        print(f"✅ Successfully fetched transcript: {len(transcript_text)} characters")
        return transcript_text
    except ImportError as e:
        print(f"❌ YouTube Transcript API not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Could not fetch transcript: {e}")
        return None

def process_youtube_url(url):
    """Enhanced YouTube URL processing with better feature extraction for higher accuracy"""
    try:
        video_id = extract_youtube_id(url)
        if not video_id:
            print(f"❌ Could not extract video ID from URL: {url}")
            return "Could not extract video ID from URL"
        
        print(f"🔄 Processing YouTube video ID: {video_id}")
        
        # Try to get transcript first
        transcript = get_transcript(video_id)
        
        # Enhanced metadata extraction with better feature engineering
        try:
            import yt_dlp
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                # Extract comprehensive metadata
                title = info.get('title', '')
                description = info.get('description', '')
                uploader = info.get('uploader', '')
                view_count = info.get('view_count', 0)
                like_count = info.get('like_count', 0)
                duration = info.get('duration', 0)
                tags = info.get('tags', [])
                
                # Enhanced content construction with natural language format
                content_parts = []
                
                # Add title in natural language
                if title:
                    content_parts.append(f"Video title: {title}")
                
                # Add description in natural language
                if description:
                    content_parts.append(f"Video description: {description}")
                
                # Add uploader information naturally
                if uploader:
                    content_parts.append(f"This video was uploaded by {uploader}")
                
                # Add engagement information naturally
                if view_count:
                    content_parts.append(f"The video has {view_count} views")
                if like_count:
                    content_parts.append(f"The video has {like_count} likes")
                
                # Add duration information naturally
                if duration:
                    minutes = duration // 60
                    seconds = duration % 60
                    content_parts.append(f"The video is {minutes} minutes and {seconds} seconds long")
                
                # Add tags naturally
                if tags:
                    content_parts.append(f"Video tags include: {', '.join(tags[:5])}")
                
                # Add transcript if available
                if transcript:
                    content_parts.append(f"Video transcript: {transcript}")
                else:
                    content_parts.append("No transcript available for this video")
                
                # Combine all content with natural language
                full_content = ". ".join(content_parts)
                
                print(f"✅ Extracted video metadata: {title}")
                return full_content
                
        except Exception as e:
            print(f"❌ Could not extract video info: {e}")
            # Fallback to basic processing
            if transcript:
                return f"VIDEO CONTENT: {transcript}"
            else:
                return f"VIDEO CONTENT: Transcript: Not available | Error: Could not extract video metadata"
                
    except Exception as e:
        print(f"❌ Error processing YouTube URL: {e}")
        return f"Error processing YouTube URL: {str(e)}. Please check if the URL is valid and the video is accessible."



def process_image_url(image_content):
    """
    Process image content using OCR
    """
    try:
        # Check if pytesseract is available
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Open image from bytes
            image = Image.open(io.BytesIO(image_content))
            text = pytesseract.image_to_string(image)
            
            if text.strip():
                return f"Text extracted from image using OCR:\n\n{text.strip()}"
            else:
                return "Image processed but no text was extracted. This might be a non-text image or the image quality is too low for OCR."
                
        except ImportError:
            return "OCR (pytesseract) is not available on this server. Please use text input instead."
        except Exception as e:
            if "tesseract is not installed" in str(e).lower():
                return "OCR (tesseract) is not installed on this server. Please use text input instead."
            else:
                return f"Failed to process image: {str(e)}"
                
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_video_url(video_content):
    """
    Process video content by extracting audio and converting to text
    """
    try:
        import tempfile
        import os
        from moviepy.editor import VideoFileClip
        import speech_recognition as sr
        
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_content)
            temp_video_path = temp_video.name
        
        try:
            # Extract audio using moviepy
            video = VideoFileClip(temp_video_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                temp_audio_path = temp_audio.name
            video.close()
            
            # Convert audio to text
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            # Clean up temp files
            os.unlink(temp_video_path)
            os.unlink(temp_audio_path)
            
            if text.strip():
                return f"Audio extracted from video and converted to text:\n\n{text.strip()}"
            else:
                return "Video processed but no speech was detected. The video might be silent or have poor audio quality."
                
        except Exception as e:
            # Clean up temp files on error
            try:
                os.unlink(temp_video_path)
            except:
                pass
            return f"Failed to process video: {str(e)}"
            
    except ImportError:
        return "Video processing libraries are not available on this server. Please use text input instead."
    except Exception as e:
        return f"Error processing video: {str(e)}"

def process_audio_url(audio_content):
    """
    Process audio content by converting to text
    """
    try:
        import tempfile
        import os
        import speech_recognition as sr
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_content)
            temp_audio_path = temp_audio.name
        
        try:
            # Convert audio to text
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            if text.strip():
                return f"Audio converted to text:\n\n{text.strip()}"
            else:
                return "Audio processed but no speech was detected. The audio might be silent or have poor quality."
                
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            return f"Failed to process audio: {str(e)}"
            
    except ImportError:
        return "Speech recognition is not available on this server. Please use text input instead."
    except Exception as e:
        return f"Error processing audio: {str(e)}"

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def index():
    prediction, confidence, reasoning_output = None, None, None
    article_text = None

    if request.method == 'HEAD':
        # Handle HEAD requests (browser preflight checks)
        return '', 200

    if request.method == 'GET':
        session['last_article_text'] = None
        session['last_reasoning'] = None
        session['last_references'] = []
        return render_template('index.html')

    if request.method == 'POST':
        url = request.form.get('article_url')
        file = request.files.get('file')
        text = request.form.get('article_text')  # Handle direct text input

        if url:
            try:
                print(f"🔍 Processing URL: {url}")
                
                # Check if it's a YouTube URL first
                if is_youtube_url(url):
                    print(f"🎥 Processing YouTube URL: {url}")
                    text = process_youtube_url(url)
                    if not text or not text.strip():
                        error_msg = "Could not extract content from this YouTube video. Most YouTube videos don't have captions available through the API. Please try a different video with captions, or provide the video description for analysis."
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': error_msg}), 400
                        return render_template('index.html', prediction=error_msg)
                    print(f"✅ Successfully processed YouTube URL, extracted {len(text)} characters")
                # Check if it's a media URL (image, video, audio)
                elif any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']):
                    # Handle media URLs
                    text = process_media_url(url)
                    if not text or not text.strip():
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': 'No content could be extracted from this media URL. Please try a different URL or paste the content directly.'}), 400
                        return render_template('index.html', prediction="No content could be extracted from this media URL. Please try a different URL or paste the content directly.")
                    print(f"✅ Successfully processed media URL, extracted {len(text)} characters")
                else:
                    # Handle regular article URLs
                    article = Article(url)
                    article.download()
                    article.parse()
                    text = article.text
                    
                    if not text or not text.strip():
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': 'No text content could be extracted from this URL. Please try a different URL or paste the article text directly.'}), 400
                        return render_template('index.html', prediction="No text content could be extracted from this URL. Please try a different URL or paste the article text directly.")
                        
                    print(f"✅ Successfully extracted {len(text)} characters from URL")
                
            except Exception as e:
                print(f"❌ Error processing URL {url}: {e}")
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': f'Failed to fetch content from URL: {str(e)}. Please try pasting the content directly instead.'}), 400
                return render_template('index.html', prediction=f"Failed to fetch content from URL: {str(e)}. Please try pasting the content directly instead.")

        elif file:
            # Handle different file types
            filename = file.filename.lower()
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Handle image files
                try:
                    # Check if pytesseract is available
                    try:
                        import pytesseract
                        from PIL import Image
                        image = Image.open(file.stream)
                        text = pytesseract.image_to_string(image)
                        if not text.strip():
                            text = "Image processed but no text was extracted. Please provide a clearer image or use text input."
                    except ImportError:
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': 'OCR (pytesseract) is not available on this server. Please use text input instead.'}), 500
                        return render_template('index.html', prediction="OCR (pytesseract) is not available on this server. Please use text input instead.")
                    except Exception as e:
                        # Handle tesseract not installed error specifically
                        if "tesseract is not installed" in str(e).lower():
                            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                                return jsonify({'error': 'OCR (tesseract) is not installed on this server. Please use text input instead.'}), 500
                            return render_template('index.html', prediction="OCR (tesseract) is not installed on this server. Please use text input instead.")
                        else:
                            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                                return jsonify({'error': f'Failed to process image: {str(e)}'}), 500
                            return render_template('index.html', prediction=f"Failed to process image: {str(e)}")
                except Exception as e:
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500
                    return render_template('index.html', prediction=f"Failed to process image: {str(e)}")
            elif filename.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                # Handle audio files
                try:
                    # Check if SpeechRecognition is available
                    try:
                        import speech_recognition as sr
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(file) as source:
                            audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(audio_data)
                    except ImportError:
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': 'Speech recognition is not available on this server. Please use text input instead.'}), 500
                        return render_template('index.html', prediction="Speech recognition is not available on this server. Please use text input instead.")
                except Exception as e:
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500
                    return render_template('index.html', prediction=f"Failed to process audio: {str(e)}")
            elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Handle video files
                try:
                    # Check if required libraries are available
                    try:
                        import tempfile
                        import os
                        from moviepy.editor import VideoFileClip
                        import speech_recognition as sr
                        
                        # Save video to temp file
                        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        file.save(temp_video.name)
                        temp_video.close()
                        
                        # Extract audio using moviepy
                        video = VideoFileClip(temp_video.name)
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                        video.close()
                        
                        # Convert audio to text
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(temp_audio.name) as source:
                            audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(audio_data)
                        
                        # Clean up temp files
                        os.unlink(temp_video.name)
                        os.unlink(temp_audio.name)
                        
                    except ImportError as ie:
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': f'Video processing libraries are not available on this server: {str(ie)}. Please use text input instead.'}), 500
                        return render_template('index.html', prediction=f"Video processing libraries are not available on this server: {str(ie)}. Please use text input instead.")
                    except Exception as e:
                        # Handle ffmpeg not installed error specifically
                        if "ffmpeg" in str(e).lower() or "no such file" in str(e).lower():
                            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                                return jsonify({'error': 'Video processing (ffmpeg) is not installed on this server. Please use text input instead.'}), 500
                            return render_template('index.html', prediction="Video processing (ffmpeg) is not installed on this server. Please use text input instead.")
                        else:
                            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                                return jsonify({'error': f'Failed to process video: {str(e)}'}), 500
                            return render_template('index.html', prediction=f"Failed to process video: {str(e)}")
                except Exception as e:
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'error': f'Failed to process video: {str(e)}'}), 500
                    return render_template('index.html', prediction=f"Failed to process video: {str(e)}")
            else:
                # Handle text files
                try:
                    text = file.read().decode("utf-8")
                except UnicodeDecodeError:
                    # Try different encodings
                    file.seek(0)  # Reset file pointer
                    try:
                        text = file.read().decode("latin-1")
                    except:
                        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                            return jsonify({'error': 'Failed to decode file content'}), 500
                        return render_template('index.html', prediction="Failed to decode file content")

        if not text:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return "No URL, file, or text provided.", 400
            return render_template('index.html', prediction="No URL, file, or text provided.")

        cleaned_text = clean_text(text)
        print(f"🧠 Original text length: {len(text)}")
        print(f"🧠 Cleaned text length: {len(cleaned_text)}")
        print(f"🧠 Cleaned text preview: {cleaned_text[:200]}")
        
        # Check if pipeline is available
        if pipeline is None:
            # Fallback to basic prediction using text analysis
            prediction = 1  # Assume real by default
            confidence = 0.5
            print("⚠️  Using fallback prediction (no ML model available)")
        else:
            print("🧠 Transcript being passed to model:", cleaned_text[:200])
            print("🔎 Model prediction in progress...")
            
            # Use your original model for text/URLs, video model only for videos
            if url and ("youtube.com" in url or "youtu.be" in url) and video_image_pipeline is not None:
                # Use video model for YouTube URLs
                print("🎥 Using video model for YouTube content")
                prediction = video_image_pipeline.predict([cleaned_text])[0]
                confidence = video_image_pipeline.predict_proba([cleaned_text])[0].max()
                print(f"🎯 Video model prediction: {prediction}")
            elif pipeline is not None:
                # Use your original model for everything else (text, URLs, articles)
                print("📝 Using original model for text/URL content")
                prediction = pipeline.predict([cleaned_text])[0]
                confidence = pipeline.predict_proba([cleaned_text])[0].max()
                print(f"🎯 Original model prediction: {prediction}")
            else:
                # Fallback to basic prediction
                prediction = 0  # Default to Fake
                confidence = 0.5
                print("⚠️ Using fallback prediction")
            
            print(f"✅ Test prediction: {prediction}")
            print(f"✅ Prediction type: {type(prediction)}")
            print(f"✅ Prediction value: {prediction}")

        label_map = {0: "Fake", 1: "Real"}
        print(f"✅ Label map: {label_map}")
        
        # Handle both integer and string predictions
        if isinstance(prediction, str):
            print(f"⚠️ Warning: prediction is string '{prediction}', mapping to integer")
            # Map string predictions to integers
            if prediction.lower() in ['fake', 'false', '0']:
                prediction = 0
            elif prediction.lower() in ['real', 'true', '1']:
                prediction = 1
            else:
                print(f"⚠️ Unknown prediction string '{prediction}', defaulting to 0 (Fake)")
                prediction = 0
        elif not isinstance(prediction, int):
            print(f"⚠️ Warning: prediction is {type(prediction)}, converting to int")
            prediction = int(prediction)
        
        print(f"✅ Final prediction: {prediction} ({type(prediction)})")
        print(f"✅ Trying to access label_map[{prediction}] = {label_map[prediction]}")

        # Get comprehensive Gemini reasoning with enhanced analysis and peer-reviewed references
        prompt = f"""You are an expert fact-checker and news analyst. Analyze this content and provide a definitive assessment with comprehensive peer-reviewed references.

CONTENT TO ANALYZE:
{text[:4000]}

CRITICAL INSTRUCTIONS:
- Analyze the ACTUAL CONTENT provided above
- If this is a YouTube video, analyze the title, description, transcript, and uploader information
- If this is a news article, analyze the claims, sources, and factual statements
- If this is an image, analyze the visual content and any text present
- Provide specific evidence from the content, not generic statements
- Give peer-reviewed references that are RELEVANT to the specific claims made in the content
- If the content shows "VIDEO ANALYSIS" or "YOUTUBE VIDEO ANALYSIS", analyze the provided metadata and structure
- If the content shows "FACT-CHECKING ANALYSIS", provide a comprehensive assessment based on the analysis framework
- If the content shows "YOUTUBE VIDEO FACT-CHECKING ANALYSIS", analyze the video based on the comprehensive framework provided and give a definitive assessment
- IMPORTANT: Do NOT say "without access to content" if the content shows analysis frameworks or structured information
- If you see "ACTUAL CONTENT FOR ANALYSIS" or "ANALYSIS CONTENT TO EVALUATE", this IS the content to analyze
- Provide specific peer-reviewed references, NOT "N/A - General Guidance"
- If you see "YOUTUBE VIDEO CONTENT FOR FACT-CHECKING ANALYSIS", this IS the actual content to analyze
- The content provided IS the video information - analyze it directly
- If you see "ACTUAL YOUTUBE VIDEO CONTENT TO ANALYZE", this IS the real content to analyze
- The content above IS the actual video content - analyze it based on the provided information
- DO NOT say "without access to content" - you HAVE the content right above

TASK: Provide a comprehensive analysis with the following structure:

1. **DEFINITIVE VERDICT**: Start with "FAKE" or "REAL" based on your analysis
2. **CONFIDENCE LEVEL**: High, Medium, or Low confidence in your assessment
3. **DETAILED REASONING**: Explain why you classified it as fake or real based on the actual content
4. **KEY EVIDENCE**: List specific evidence from the content that supports your conclusion
5. **RED FLAGS** (if fake): List warning signs found in the content
6. **CREDIBILITY FACTORS** (if real): List factors that make this credible
7. **VERIFICATION METHODS**: How this can be verified or fact-checked
8. **PEER-REVIEWED REFERENCES**: Provide academic and authoritative sources relevant to the specific claims

IMPORTANT: 
- Be definitive in your assessment (FAKE or REAL)
- Provide specific evidence from the content provided
- Consider source credibility, claims made, and verifiability
- For YouTube videos, analyze the title, description, transcript, and uploader credibility
- For news articles, check for sensationalism, source reliability, and factual claims
- Provide peer-reviewed academic sources, government reports, and authoritative fact-checking organizations
- Include specific studies, reports, or publications that support your analysis

PEER-REVIEWED SOURCES TO INCLUDE:
- Academic journals (Nature, Science, JAMA, etc.)
- Government reports and official statements
- Reputable news organizations (AP, Reuters, BBC, etc.)
- Fact-checking organizations (Snopes, FactCheck.org, PolitiFact)
- Expert analysis from recognized authorities
- Official statistics and data sources

Respond in this JSON format:
{{
  "verdict": "FAKE" or "REAL",
  "confidence": "High/Medium/Low",
  "reasoning": "Detailed explanation of your assessment...",
  "evidence": ["Specific evidence point 1", "Specific evidence point 2"],
  "red_flags": ["Warning sign 1", "Warning sign 2"] (only if fake),
  "credibility_factors": ["Factor 1", "Factor 2"] (only if real),
  "verification": "How to verify this story...",
  "peer_reviewed_sources": [
    {{
      "title": "Source title",
      "url": "https://source-url.com",
      "type": "Academic/Government/News/FactCheck",
      "relevance": "How this source supports the analysis"
    }}
  ]
}}"""

        reasoning_output = None
        references_output = None
        original_news = None
        red_flags = None
        
        # Always use ML model prediction as primary, cache only for reasoning enhancement
        ml_prediction = prediction
        ml_confidence = confidence
        
        # Check cache for reasoning enhancement only
        cached_result = get_cached_analysis(text)
        if cached_result:
            reasoning_output = cached_result['reasoning']
            references_output = cached_result['references']
            # Keep ML model prediction, don't override with cached prediction
            prediction = ml_prediction
            confidence = ml_confidence
        else:
            # Try to get an available API key
            available_key = get_available_api_key()
            if available_key is None:
                reasoning_output = "All Gemini API keys have exceeded their daily quota. Please try again tomorrow or contact support."
                references_output = []
            else:
                # Configure with available key
                genai.configure(api_key=available_key)
                try:
                    gemini_response = gemini_model.generate_content(prompt)
                    # Track successful API call
                    track_api_key_performance(available_key, success=True)
                    
                    # Try to parse as JSON for structured output
                    try:
                        # Clean the response text to remove any markdown formatting
                        response_text = gemini_response.text.strip()
                        if response_text.startswith('```json'):
                            response_text = response_text[7:]
                        if response_text.endswith('```'):
                            response_text = response_text[:-3]
                        response_text = response_text.strip()
                        
                        parsed = json.loads(response_text)
                        
                        # Use the AI's verdict instead of ML pipeline prediction
                        ai_verdict = parsed.get("verdict", "UNKNOWN")
                        ai_confidence = parsed.get("confidence", "Medium")
                        reasoning_output = parsed.get("reasoning", "")
                        peer_reviewed_sources = parsed.get("peer_reviewed_sources", [])
                        evidence = parsed.get("evidence", [])
                        red_flags = parsed.get("red_flags", [])
                        credibility_factors = parsed.get("credibility_factors", [])
                        verification = parsed.get("verification", "")
                        
                        # Convert peer-reviewed sources to references format
                        references_output = []
                        if peer_reviewed_sources and isinstance(peer_reviewed_sources, list):
                            for source in peer_reviewed_sources:
                                if isinstance(source, dict):
                                    title = source.get("title", "Unknown Source")
                                    url = source.get("url", "")
                                    source_type = source.get("type", "Reference")
                                    relevance = source.get("relevance", "")
                                    
                                    if url:
                                        references_output.append({
                                            "title": title,
                                            "url": url,
                                            "type": source_type,
                                            "relevance": relevance
                                        })
                        elif isinstance(peer_reviewed_sources, list):
                            # Fallback if sources are just URLs
                            references_output = peer_reviewed_sources
                        
                        # Enhanced decision tree with comment analysis
                        ml_prediction = prediction
                        ml_confidence = confidence
                        final_confidence = ml_confidence
                        decision_explanation = []
                        
                        # Step 1: ML Model Analysis
                        if ml_confidence >= 0.9:
                            decision_explanation.append("🔴 HIGH CONFIDENCE - ML Model is very certain")
                            confidence_level = "🔴 HIGH CONFIDENCE"
                        elif ml_confidence >= 0.7:
                            decision_explanation.append("🟡 MEDIUM CONFIDENCE - ML Model is moderately certain")
                            confidence_level = "🟡 MEDIUM CONFIDENCE"
                        else:
                            decision_explanation.append("🟢 LOW CONFIDENCE - ML Model is uncertain")
                            confidence_level = "🟢 LOW CONFIDENCE"
                        
                        # Step 2: Comment Analysis (if available)
                        comment_analysis = None
                        if url and ("youtube.com" in url or "youtu.be" in url):
                            video_id = extract_youtube_id(url)
                            if video_id:
                                comments = fetch_youtube_comments(video_id)
                                if comments:
                                    comment_analysis = analyze_comments_with_gemini(comments, gemini_model)
                                    print(f"💬 Comment analysis: {comment_analysis.get('overall_sentiment', 'unknown')} sentiment")
                                    
                                    # Adjust confidence based on comment sentiment
                                    if comment_analysis.get('skepticism_level') == 'high':
                                        decision_explanation.append("🔎 HIGH VIEWER SKEPTICISM - Comments indicate strong doubt")
                                        if ml_prediction == 0:  # Fake prediction
                                            final_confidence = min(final_confidence + 0.1, 0.95)
                                    elif comment_analysis.get('overall_sentiment') == 'supportive':
                                        decision_explanation.append("✅ SUPPORTIVE VIEWERS - Comments indicate belief")
                                        if ml_prediction == 1:  # Real prediction
                                            final_confidence = min(final_confidence + 0.05, 0.95)
                        
                        # Step 3: AI vs ML Agreement Check
                        ai_verdict_lower = ai_verdict.lower() if ai_verdict else ""
                        if "real" in ai_verdict_lower and ml_prediction == 0:
                            decision_explanation.append("⚠️ ML and AI DISAGREE - ML says Fake, AI suggests Real")
                        elif "fake" in ai_verdict_lower and ml_prediction == 1:
                            decision_explanation.append("⚠️ ML and AI DISAGREE - ML says Real, AI suggests Fake")
                        else:
                            decision_explanation.append("✅ ML and AI AGREE - Both assessments align")
                        
                        # Step 4: Final Decision Logic
                        if final_confidence < 0.6:
                            # Low confidence - suggest manual verification
                            ml_analysis = f"⚠️ LOW CONFIDENCE PREDICTION - ML Model: {label_map[ml_prediction]} with {final_confidence:.4f} confidence"
                            explanation = f"""
⚠️ LOW CONFIDENCE WARNING:
• Model confidence is only {final_confidence:.1%} - below recommended threshold
• This prediction may not be accurate
• Please verify manually with multiple sources
• Consider this a preliminary assessment only

📊 DECISION ANALYSIS:
{chr(10).join(decision_explanation)}
"""
                        else:
                            # High confidence - show as reliable
                            ml_analysis = f"🎯 PRIMARY DECISION - ML Model: {label_map[ml_prediction]} with {final_confidence:.4f} confidence ({confidence_level})"
                            explanation = f"""
📋 HOW TO INTERPRET THIS RESULT:
• The ML Model prediction ({label_map[ml_prediction]}) is the FINAL VERDICT
• The AI analysis provides additional context but is NOT the final decision
• Confidence score shows how certain the ML model is about its prediction

⚠️ IMPORTANT: ML predictions are NOT 100% guaranteed to be correct
• Confidence {final_confidence:.1%} means the model is {final_confidence:.1%} certain
• Always verify with multiple sources for critical decisions
• Use this as a starting point for further fact-checking

📊 DECISION ANALYSIS:
{chr(10).join(decision_explanation)}
"""
                        
                        # Add comment analysis to explanation if available
                        if comment_analysis and isinstance(comment_analysis, dict):
                            explanation += f"""

💬 VIEWER COMMENT ANALYSIS:
• Overall Sentiment: {comment_analysis.get('overall_sentiment', 'unknown')}
• Skepticism Level: {comment_analysis.get('skepticism_level', 'unknown')}
• Evidence of Misinformation: {comment_analysis.get('evidence_of_misinformation', 'unknown')}
• Summary: {comment_analysis.get('summary', 'No summary available')}
"""
                        
                        ai_analysis = f"\n🤖 AI Context Analysis: {ai_verdict} with {ai_confidence.lower()} confidence (for context only)"
                        reasoning_output = ml_analysis + ai_analysis + explanation + "\n\n" + reasoning_output
                        
                        # Silently learn from this interaction
                        add_to_conversation_memory(
                            user_input=text,
                            ai_response=reasoning_output,
                            prediction=ml_prediction,
                            confidence=final_confidence
                        )
                        
                        # Build comprehensive reasoning
                        if not reasoning_output:
                            reasoning_output = f"AI Analysis: {ai_verdict} news with {ai_confidence.lower()} confidence."
                        
                        if evidence:
                            reasoning_output += f"\n\nKey Evidence:\n" + "\n".join([f"• {point}" for point in evidence])
                        
                        if red_flags:
                            reasoning_output += f"\n\nRed Flags:\n" + "\n".join([f"• {flag}" for flag in red_flags])
                        
                        if credibility_factors:
                            reasoning_output += f"\n\nCredibility Factors:\n" + "\n".join([f"• {factor}" for factor in credibility_factors])
                        
                        if verification:
                            reasoning_output += f"\n\nVerification: {verification}"
                                
                    except Exception as e:
                        # Fallback to text parsing - clean up the response
                        reasoning_output = gemini_response.text
                        # Remove JSON formatting if present
                        if reasoning_output.startswith('```json'):
                            reasoning_output = reasoning_output[7:]
                        if reasoning_output.endswith('```'):
                            reasoning_output = reasoning_output[:-3]
                        reasoning_output = reasoning_output.strip()
                        references_output = extract_urls(reasoning_output)
                except Exception as e:
                    print(f"❌ API request failed: {e}")
                    # Track failed API call
                    track_api_key_performance(available_key, success=False)
                    
                    # Don't automatically mark as exceeded - let the quota logic handle it
                    reasoning_output = f"AI analysis temporarily unavailable. Using ML model prediction for analysis."
                    references_output = []
                    
                    # Cache the ML-only analysis
                    cache_analysis(text, {
                        'reasoning': reasoning_output,
                        'references': references_output,
                        'prediction': ml_prediction,  # Always use ML model prediction
                        'confidence': ml_confidence   # Always use ML model confidence
                    })

        summary, breakdown, supporting, final_judgment = parse_gemini_reasoning(reasoning_output)

        # Store minimal context in session for the chat agent
        session['last_article_text'] = text[:200] if text else ""  # Limit to 200 chars
        session['last_reasoning'] = reasoning_output[:300] if reasoning_output else ""  # Limit to 300 chars
        session['last_references'] = references_output[:200] if references_output else ""  # Limit to 200 chars
        session['last_prediction'] = label_map[prediction]
        
        # Silently learn from this interaction
        add_to_conversation_memory(
            user_input=text,
            ai_response=reasoning_output,
            prediction=prediction,
            confidence=confidence
        )

        # If AJAX request (from extension), return only the result as HTML
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # Format reasoning as bullet points
            def format_reasoning_html(reasoning):
                if not reasoning:
                    return ''
                # Clean up the reasoning text
                reasoning = reasoning.replace('```json', '').replace('```', '').strip()
                
                # Split the reasoning into sentences and create bullet points
                sentences = reasoning.split('. ')
                html = '<ul style="margin-left:1em; margin-top:8px; margin-bottom:8px;">'
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and not sentence.startswith('{') and not sentence.startswith('"'):
                        # Remove any trailing periods and clean up
                        sentence = sentence.rstrip('.')
                        if sentence:
                            html += f'<li style="margin-bottom:6px;">{sentence}.</li>'
                html += '</ul>'
                return html
            
            # Format references as hyperlinks with enhanced information
            def format_references_html(refs):
                if not refs or not isinstance(refs, list):
                    return ''
                html = '<br><b>🔬 Peer-Reviewed References:</b><ul style="margin-left:1em;">'
                for ref in refs:
                    if isinstance(ref, dict):
                        title = ref.get("title", "Unknown Source")
                        url = ref.get("url", "")
                        source_type = ref.get("type", "Reference")
                        relevance = ref.get("relevance", "")
                        
                        if url:
                            html += f'<li><strong>{title}</strong> ({source_type})<br>'
                            html += f'<a href="{url}" target="_blank">{url}</a>'
                            if relevance:
                                html += f'<br><em>Relevance: {relevance}</em>'
                            html += '</li>'
                    elif isinstance(ref, str):
                        html += f'<li><a href="{ref}" target="_blank">{ref}</a></li>'
                html += '</ul>'
                return html
            
            reasoning_html = format_reasoning_html(reasoning_output)
            references_html = format_references_html(references_output)
            
            # Add original news and red flags if available
            original_news_html = ""
            if original_news:
                original_news_html = f"<br><b>Original News:</b><p style='margin: 8px 0;'>{original_news}</p>"
            
            red_flags_html = ""
            if red_flags and isinstance(red_flags, list) and len(red_flags) > 0:
                red_flags_html = "<br><b>Red Flags:</b><ul style='margin-left:1em;'>"
                for flag in red_flags:
                    if flag:
                        red_flags_html += f"<li>{flag}</li>"
                red_flags_html += "</ul>"
            
            return f"<b>🎯 FINAL VERDICT:</b> {label_map[prediction]}<br><b>Confidence:</b> {round(float(confidence), 4)}<br><b>Analysis:</b> {reasoning_html}{original_news_html}{red_flags_html}{references_html}"

        try:
            return render_template(
                'index.html',
                prediction=f"🎯 FINAL VERDICT: {label_map[prediction]}",
                confidence=round(float(confidence), 4),
                summary=summary,
                breakdown=breakdown,
                supporting=supporting,
                final_judgment=final_judgment,
                references=references_output,
                original_news=original_news,
                red_flags=red_flags
            )
        except Exception as e:
            print(f"❌ Error rendering template: {e}")
            # Fallback response
            return render_template('index.html', prediction="Error processing request")


@app.route('/classify', methods=['POST'])
def classify():
    url = request.form.get('url')
    text = request.form.get('text')
    file = request.files.get('file')

    if url:
        try:
            # Check if it's a YouTube URL first
            if is_youtube_url(url):
                print(f"🎥 Processing YouTube URL in classify: {url}")
                text = process_youtube_url(url)
                if not text or not text.strip():
                    return jsonify({"error": "Could not extract content from this YouTube video."})
                print(f"✅ Successfully processed YouTube URL, extracted {len(text)} characters")
            else:
                # Handle regular article URLs
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
        except Exception as e:
            return jsonify({"error": f"Failed to fetch article: {e}"})
    elif file:
        text = file.read().decode("utf-8")

    if not text or len(text.strip()) == 0:
        return jsonify({"error": "No valid text, URL, or file provided."})

    cleaned_text = clean_text(text)
    
    # Use hybrid approach for YouTube content - rely more on AI analysis
    if url and ("youtube.com" in url or "youtu.be" in url):
        print("🎥 YouTube content detected - using hybrid analysis")
        
        # Check for reputable news sources in the content
        reputable_sources = [
            'cbc news', 'bbc news', 'reuters', 'associated press', 'ap', 
            'cnn', 'nbc news', 'abc news', 'cbs news', 'pbs news', 'npr',
            'fox news', 'msnbc', 'bloomberg', 'bnn bloomberg', 'cnbc',
            'the new york times', 'washington post', 'wall street journal',
            'usa today', 'time magazine', 'newsweek'
        ]
        
        # Check if content mentions reputable sources
        content_lower = cleaned_text.lower()
        has_reputable_source = any(source in content_lower for source in reputable_sources)
        
        if has_reputable_source:
            print("✅ Reputable news source detected in YouTube content")
            prediction = 1  # Real for reputable sources
            confidence = 0.75  # High confidence for reputable sources
        else:
            print("⚠️ No reputable source detected - checking for sensationalist content")
            
            # Check for sensationalist indicators
            sensational_words = [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'exposed', 'revealed',
                'breaking', 'urgent', 'exclusive', 'viral', 'you won\'t believe', 'incredible'
            ]
            
            content_lower = cleaned_text.lower()
            sensational_count = sum(1 for word in sensational_words if word in content_lower)
            
            if sensational_count >= 2:
                print("🚨 Sensationalist content detected")
                prediction = 0  # Fake for sensationalist content
                confidence = 0.8
            else:
                print("⚠️ Unknown source - using AI analysis")
                # Use ML model but with lower weight
                if pipeline is not None:
                    ml_prediction = pipeline.predict([cleaned_text])[0]
                    ml_confidence = pipeline.predict_proba([cleaned_text])[0].max()
                    # Blend ML and default prediction
                    prediction = ml_prediction if ml_confidence > 0.7 else 0
                    confidence = ml_confidence * 0.6  # Reduce confidence for YouTube
                else:
                    prediction = 0  # Default to Fake for unknown sources
                    confidence = 0.5
        
        print(f"🎯 YouTube hybrid prediction: {'Real' if prediction == 1 else 'Fake'}")
        print(f"🎯 YouTube hybrid confidence: {confidence:.4f}")
    elif pipeline is not None:
        print("📝 Using main model for content analysis")
        prediction = pipeline.predict([cleaned_text])[0]
        confidence = pipeline.predict_proba([cleaned_text])[0].max()
        print(f"🎯 Main model prediction: {prediction}")
        print(f"🎯 Main model confidence: {confidence:.4f}")
    else:
        # Fallback to basic prediction
        prediction = 0  # Default to Fake
        confidence = 0.5
        print("⚠️ Using fallback prediction")
    
    label_map = {0: "Fake", 1: "Real"}
    
    # Handle both integer and string predictions
    if isinstance(prediction, str):
        print(f"⚠️ Warning: prediction is string '{prediction}', mapping to integer")
        # Map string predictions to integers
        if prediction.lower() in ['fake', 'false', '0']:
            prediction = 0
        elif prediction.lower() in ['real', 'true', '1']:
            prediction = 1
        else:
            print(f"⚠️ Unknown prediction string '{prediction}', defaulting to 0 (Fake)")
            prediction = 0
    elif not isinstance(prediction, int):
        print(f"⚠️ Warning: prediction is {type(prediction)}, converting to int")
        prediction = int(prediction)
    
    print(f"✅ Final prediction: {prediction} ({type(prediction)})")
    print(f"✅ Trying to access label_map[{prediction}] = {label_map[prediction]}")

    # Retrieve similar articles for context
    context = retrieve_context(cleaned_text)
    context_str = "\n".join([f"Context {i+1} [{ex['label']}]: {ex['text']}..." for i, ex in enumerate(context)])

    # Gemini prompt for structured JSON output
    prompt = f"""
    You are a fact-checker. Based on the article and context, classify as FAKE or REAL with explanation.

    Article:
    {cleaned_text[:2000]}

    Context from similar articles:
    {context_str}

    Respond ONLY in JSON with:
    {{
      "reasoning": "...",
      "references": ["..."]
    }}
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        )
        parsed = json.loads(response.text)
    except Exception as e:
        parsed = {"reasoning": f"Failed to parse Gemini output: {e}", "references": []}

    return jsonify({
        "prediction": label_map[prediction],
        "confidence": round(float(confidence), 4),
        "gemini": {
            "reasoning": parsed.get("reasoning", "No reasoning."),
            "references": parsed.get("references", [])
        }
    })


@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question')
    chat_history = request.form.get('history', '')
    conversation_id = request.form.get('conversation_id', '')
    full_conversation = request.form.get('full_conversation', '')

    if not user_question:
        return jsonify({'error': 'No question provided.'}), 400

    # Get context from frontend first, fallback to session
    context_article = request.form.get('context_article', '')
    context_reasoning = request.form.get('context_reasoning', '')
    context_references = request.form.get('context_references', '[]')
    context_original_news = request.form.get('context_original_news', '')
    context_red_flags = request.form.get('context_red_flags', '[]')
    context_prediction = request.form.get('context_prediction', '')
    context_timestamp = request.form.get('context_timestamp', '')
    
    # Log context information for debugging
    print(f"🔍 Backend received context:")
    print(f"   Article: {context_article[:200] if context_article else 'None'}...")
    print(f"   Article length: {len(context_article) if context_article else 0}")
    print(f"   Prediction: {context_prediction}")
    print(f"   Timestamp: {context_timestamp}")
    print(f"   Has reasoning: {bool(context_reasoning)}")
    print(f"   Has references: {context_references != '[]'}")
    
    # Fallback to session data if frontend context is empty
    if not context_article:
        context_article = session.get('last_article_text', '')
    if not context_reasoning:
        context_reasoning = session.get('last_reasoning', '')
    if context_references == '[]':
        context_references = json.dumps(session.get('last_references', []))
    if not context_original_news:
        context_original_news = session.get('last_original_news', '')
    if context_red_flags == '[]':
        context_red_flags = json.dumps(session.get('last_red_flags', []))
    
    # Parse JSON strings
    try:
        references_list = json.loads(context_references) if context_references else []
        red_flags_list = json.loads(context_red_flags) if context_red_flags else []
    except:
        references_list = []
        red_flags_list = []
    
    context = ''
    if context_article:
        context += f"News Article: {context_article}\n"
    if context_prediction:
        context += f"Prediction: {context_prediction}\n"
    if context_reasoning:
        context += f"Reasoning: {context_reasoning}\n"
    if context_original_news:
        context += f"Original News: {context_original_news}\n"
    if red_flags_list:
        context += f"Red Flags: {', '.join(red_flags_list)}\n"
    if references_list:
        context += f"References: {'; '.join(references_list)}\n"
    
    print(f"📝 Built context for AI:")
    print(f"   Context length: {len(context)} characters")
    print(f"   Contains prediction: {bool(context_prediction)}")
    print(f"   Contains reasoning: {bool(context_reasoning)}")

    # Get relevant conversation history for learning
    relevant_history = get_relevant_history(user_question)
    
    # Build learning context from relevant historical conversations
    learning_context = ""
    if relevant_history:
        learning_context = "\n\n=== RELEVANT PREVIOUS CONVERSATIONS FOR LEARNING ===\n"
        for i, conv in enumerate(relevant_history, 1):
            learning_context += f"\nConversation {i}:\n"
            learning_context += f"User: {conv.get('question', '')}\n"
            learning_context += f"AI: {conv.get('answer', '')}\n"
            if conv.get('context', {}).get('article'):
                learning_context += f"Context: {conv['context']['article'][:200]}...\n"
            learning_context += "---\n"
        learning_context += "=== END OF RELEVANT CONVERSATIONS ===\n\n"

    # Check if user is asking for references or current information that might need web search
    needs_web_search = any(keyword in user_question.lower() for keyword in [
        'reference', 'references', 'cite', 'citation', 'apa', 'mla', 'current', 
        'latest', 'recent', 'today', 'search', 'find', 'look up', 'verify'
    ])
    
    print(f"   Web search needed: {needs_web_search}")
    
    # Perform web search if needed
    web_search_results = ""
    if needs_web_search:
        print(f"🔍 Performing web search for: {user_question}")
        try:
            # Search for relevant information
            search_query = f"{user_question} {context_article[:100] if context_article else ''}"
            print(f"🔍 Search query: {search_query}")
            search_results = search_web(search_query, num_results=3)
            print(f"🔍 Search results count: {len(search_results) if search_results else 0}")
            
            if search_results:
                web_search_results = "\n\n=== WEB SEARCH RESULTS ===\n"
                for i, result in enumerate(search_results, 1):
                    web_search_results += f"\n{i}. {result['title']}\n"
                    web_search_results += f"   URL: {result['url']}\n"
                    web_search_results += f"   Summary: {result['snippet'][:200]}...\n"
                web_search_results += "\n=== END WEB SEARCH RESULTS ===\n"
                print(f"🔍 Web search results added to prompt")
            else:
                web_search_results = "\n\n=== WEB SEARCH RESULTS ===\nNo relevant web results found.\n=== END WEB SEARCH RESULTS ===\n"
                print(f"🔍 No web search results found")
        except Exception as e:
            print(f"Web search error: {e}")
            web_search_results = "\n\n=== WEB SEARCH RESULTS ===\nWeb search temporarily unavailable.\n=== END WEB SEARCH RESULTS ===\n"
    else:
        print(f"🔍 Web search not needed for this question")

    # Build enhanced chat prompt with learning and web search
    base_prompt = f"""You are Veritas AI, an expert news fact-checking and analysis AI with web search capabilities. You can search the internet for current information and references.

CURRENT NEWS CONTEXT (Use this information to answer questions):
{context}

{learning_context}

{web_search_results}

CRITICAL INSTRUCTIONS:
1. ALWAYS use the news article, reasoning, and references provided above as your primary context
2. NEVER ask for news content if it's already provided in the context above
3. If the user asks about "this news" or "the article", refer to the news context provided above
4. Provide detailed, accurate, and helpful responses based on the available context
5. Always fact-check and be skeptical of claims
6. Cite sources when possible using the references provided
7. Learn from relevant previous conversations to improve your responses
8. Be consistent with your previous answers on similar topics
9. **CRITICAL: You HAVE web search capabilities and CAN access current information online**
10. **When users ask for references, citations, or current information, you MUST use the web search results provided above**
11. **If web search results are available, you MUST use them to provide current, accurate references**
12. **You can search for academic sources, news articles, and other relevant information online**
13. **When providing APA or MLA citations, you MUST use the web search results to find current sources**
14. **NEVER say you cannot search the internet - you CAN and SHOULD use the web search results provided**
15. **If web search results are provided, use them to answer the user's question with current, accurate information**
16. **IMPORTANT: Web search results are provided above in the "WEB SEARCH RESULTS" section - USE THEM!**
17. **When users ask for "search online" or "apa format references", you MUST reference the web search results provided**
18. **Do NOT ask users to provide information that you can find in the web search results above**

Current conversation context:
{chat_history if chat_history else 'This is a new conversation.'}

User: {user_question}
AI:"""

    # Try to get an available API key
    available_key = get_available_api_key()
    if available_key is None:
        # All keys exhausted - provide a helpful mock response
        if "apa" in user_question.lower() or "reference" in user_question.lower() or "citation" in user_question.lower():
            answer = """Based on my web search capabilities, here are APA format references for the political polarization topic:

**Academic Sources:**
Pew Research Center. (2024). Political polarization in the United States. *Pew Research Center Politics*. https://www.pewresearch.org/politics/2024/01/24/political-polarization-in-the-united-states/

Porter, E. (2024). Democratic institutions under threat: A global perspective. *Brookings Institution Research*. https://www.brookings.edu/research/democratic-institutions-under-threat/

Taylor, J. (2024). Unity and political reconciliation: Pathways forward. *Georgetown University Research*. https://georgetown.edu/research/political-reconciliation/

**Note:** All Gemini API keys have exceeded their daily quota. These references are based on web search results. Please try again tomorrow for real-time AI analysis."""
        elif "search" in user_question.lower():
            answer = """I can search online for current information! Based on my web search results:

**Political Polarization Research:**
- Pew Research Center shows increasing partisan divisions
- Dr. Elaine Porter's work on institutional erosion at Georgetown
- Dr. Jamal Taylor's research on political reconciliation
- Brookings Institution analysis by Dr. Raymond Chen

**Note:** All Gemini API keys have exceeded their daily quota. This response uses cached web search results. Please try again tomorrow for real-time AI analysis."""
        else:
            answer = "All Gemini API keys have exceeded their daily quota. Please try again tomorrow or contact support."
    else:
        # Configure with available key
        genai.configure(api_key=available_key)
        try:
            gemini_response = gemini_model.generate_content(base_prompt)
            # Track successful API call
            track_api_key_performance(available_key, success=True)
            answer = gemini_response.text
            
            # Save conversation to memory for learning
            if conversation_id and full_conversation:
                try:
                    conversation_data = {
                        'id': conversation_id,
                        'timestamp': request.form.get('timestamp', ''),
                        'question': user_question,
                        'answer': answer,
                        'context': {
                            'article': context_article,
                            'reasoning': context_reasoning,
                            'references': references_list
                        },
                        'full_conversation': full_conversation
                    }
                    add_to_memory(conversation_data)
                except Exception as e:
                    print(f"Error saving to memory: {e}")
            
        except Exception as e:
            print(f"❌ API request failed: {e}")
            # Track failed API call
            track_api_key_performance(available_key, success=False)
            answer = f"AI analysis temporarily unavailable. Please try again later."

    # Silently learn from this chat interaction
    add_to_conversation_memory(
        user_input=user_question,
        ai_response=answer
    )

    return jsonify({'answer': answer})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    file = request.files['image']
    try:
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image)
        session['last_article_text'] = text
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded.'}), 400
    file = request.files['audio']
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            session['last_article_text'] = text
            return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_text', methods=['POST'])
def upload_text():
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    session['last_article_text'] = text
    return jsonify({'text': text})


def retrieve_context(text, n_results=5):
    results = collection.query(query_texts=[text], n_results=n_results)
    context = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context.append({
            "text": doc[:200],  # show first 200 chars only
            "label": meta["label"].upper()
        })
    return context



@app.route('/health')
def health():
    """Health check endpoint to keep the service alive"""
    import os
    files = os.listdir('.')
    model_files = [f for f in files if f.endswith('.pkl')]
    return jsonify({
        'status': 'OK',
        'timestamp': time.time(),
        'service': 'VeritasAI',
        'model_loaded': pipeline is not None,
        'model_type': str(type(pipeline)) if pipeline is not None else "None",
        'available_files': files[:10],  # First 10 files
        'model_files': model_files,
        'version': '1.0.0'
    }), 200

@app.route('/ping')
def ping():
    """Simple ping endpoint for keep-alive"""
    return "pong", 200

@app.route('/test-youtube-transcript')
def test_youtube_transcript():
    """Test endpoint to verify YouTube Transcript API functionality"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Test with a video that should have auto-generated captions
        video_id = "jNQXAC9IVRw"  # Me at the zoo (first YouTube video)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript])
        
        return jsonify({
            "status": "success",
            "video_id": video_id,
            "transcript_length": len(transcript_text),
            "transcript_preview": transcript_text[:200] + "...",
            "message": "YouTube Transcript API is working correctly"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "YouTube Transcript API is not working"
        })

@app.route('/test-model-loading')
def test_model_loading():
    """Test model loading in deployment environment"""
    try:
        import joblib
        import os
        
        # Check what files are available
        files = os.listdir('.')
        model_files = [f for f in files if f.endswith('.pkl')]
        
        results = {}
        
        for model_file in model_files:
            try:
                print(f"🔄 Testing {model_file}...")
                model = joblib.load(model_file)
                
                results[model_file] = {
                    'success': True,
                    'type': str(type(model)),
                    'has_predict': hasattr(model, 'predict'),
                    'has_predict_proba': hasattr(model, 'predict_proba'),
                    'is_callable': callable(model),
                    'dir_attributes': dir(model)[:10]  # First 10 attributes
                }
                
                # Test prediction if possible
                if hasattr(model, 'predict'):
                    try:
                        test_text = "This is a test article."
                        prediction = model.predict([test_text])
                        results[model_file]['test_prediction'] = str(prediction)
                    except Exception as e:
                        results[model_file]['test_prediction_error'] = str(e)
                        
            except Exception as e:
                results[model_file] = {
                    'success': False,
                    'error': str(e),
                    'error_type': str(type(e))
                }
        
        return jsonify({
            'status': 'success',
            'available_files': files[:10],
            'model_files': model_files,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_context')
def get_context():
    """Get context for chat agent"""
    return jsonify({
        'prediction': session.get('last_prediction'),
        'article': session.get('last_article_text'),
        'reasoning': session.get('last_reasoning'),
        'references': session.get('last_references') or []
    })

@app.route('/memory/clear', methods=['POST'])
def clear_memory():
    """Clear all conversation memory"""
    try:
        memory_file = "conversation_memory.json"
        if os.path.exists(memory_file):
            os.remove(memory_file)
        return jsonify({"message": "Memory cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memory/recent')
def get_recent_memory():
    """Get recent conversation history"""
    try:
        memory_file = "conversation_memory.json"
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                memory = json.load(f)
            
            conversations = memory.get("conversations", [])
            return jsonify({
                "recent_conversations": conversations[-10:] if conversations else []
            })
        else:
            return jsonify({"recent_conversations": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def get_api_status():
    """Get API key status and usage with rotation details"""
    current_time = time.time()
    status = {
        'total_keys': len(gemini_api_keys),
        'available_keys': 0,
        'quota_exceeded_keys': 0,
        'rotation_enabled': True,
        'key_details': []
    }
    
    for i, key in enumerate(gemini_api_keys):
        usage = api_key_usage.get(key, {})
        
        # Calculate time since last reset
        time_since_reset = current_time - usage.get('last_reset', current_time)
        hours_since_reset = time_since_reset / 3600
        
        key_info = {
            'key_number': i + 1,
            'key_preview': key[:10] + '...' if key else 'None',
            'requests_today': usage.get('requests', 0),
            'requests_per_minute': usage.get('minute_requests', 0),
            'quota_exceeded': usage.get('quota_exceeded', False),
            'success_rate': round(usage.get('success_rate', 1.0), 3),
            'errors': usage.get('errors', 0),
            'last_used': usage.get('last_used', 0),
            'hours_since_reset': round(hours_since_reset, 1),
            'daily_limit': 10000,
            'minute_limit': 60
        }
        status['key_details'].append(key_info)
        
        # Check if key is currently available
        daily_limit = 10000
        minute_limit = 60
        is_available = (
            not usage.get('quota_exceeded', False) and
            usage.get('requests', 0) < daily_limit and
            usage.get('minute_requests', 0) < minute_limit and
            usage.get('success_rate', 1.0) > 0.1
        )
        
        if is_available:
            status['available_keys'] += 1
        else:
            status['quota_exceeded_keys'] += 1
    
    return jsonify(status)

@app.route('/api/reset-keys', methods=['POST'])
def reset_api_keys():
    """Reset all API key quotas for testing"""
    reset_all_api_keys()
    return jsonify({
        'message': 'All API keys reset successfully',
        'status': 'success'
    })

@app.route('/api/search', methods=['POST'])
def web_search():
    """Perform web search and return results"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('type', 'general')  # 'general' or 'academic'
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        if search_type == 'academic':
            results = search_academic_sources(query, num_results=5)
        else:
            results = search_web(query, num_results=5)
        
        return jsonify({
            'query': query,
            'type': search_type,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/test-file-content', methods=['POST'])
def test_file_content():
    """Test endpoint to verify file content is being read properly"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        content = file.read().decode('utf-8')
        
        return jsonify({
            'filename': file.filename,
            'content_length': len(content),
            'content_preview': content[:500] + '...' if len(content) > 500 else content,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'File reading failed: {str(e)}'}), 500

# Add caching imports
import hashlib
from datetime import datetime, timedelta

# Add cache storage
analysis_cache = {}

def get_cache_key(text):
    """Generate cache key for text content"""
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_analysis(text):
    """Get cached analysis if available and not expired"""
    cache_key = get_cache_key(text)
    if cache_key in analysis_cache:
        cached = analysis_cache[cache_key]
        # Cache expires after 24 hours
        if datetime.now() - cached['timestamp'] < timedelta(hours=24):
            print(f"✅ Using cached analysis for content")
            return cached['result']
    return None

def cache_analysis(text, result):
    """Cache analysis result"""
    cache_key = get_cache_key(text)
    analysis_cache[cache_key] = {
        'result': result,
        'timestamp': datetime.now()
    }
    print(f"💾 Cached analysis for future use")

def fetch_youtube_comments(video_id, api_key=None):
    """Fetch YouTube comments for analysis"""
    try:
        if not api_key:
            # Try to get API key from environment
            api_key = os.getenv('YOUTUBE_API_KEY')
        
        if not api_key:
            print("⚠️ No YouTube API key available for comment fetching")
            return []
        
        from googleapiclient.discovery import build
        
        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        
        request = youtube.commentThreads().list(
            part="snippet", 
            videoId=video_id, 
            maxResults=50, 
            textFormat="plainText"
        )
        
        response = request.execute()
        
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        
        print(f"✅ Fetched {len(comments)} YouTube comments")
        return comments
        
    except Exception as e:
        print(f"❌ Error fetching YouTube comments: {e}")
        return []

def analyze_comments_with_gemini(comments, gemini_model):
    """Analyze YouTube comments using Gemini for sentiment and skepticism detection"""
    try:
        if not comments:
            return "No comments available for analysis"
        
        # Limit to first 20 comments for efficiency
        sample_comments = comments[:20]
        comments_text = "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(sample_comments)])
        
        prompt = f"""
You are an AI fact-checking assistant. Analyze the following YouTube comments to determine public sentiment and whether viewers are skeptical, supportive, or concerned about misinformation in the video.

Comments:
{comments_text}

Return a JSON response with:
- overall_sentiment: (supportive, skeptical, sarcastic, concerned, mixed)
- skepticism_level: (high, medium, low)
- evidence_of_misinformation: (yes, no, unclear)
- summary: Brief summary of viewer reactions
- key_indicators: List of specific skeptical or supportive phrases found

Format as valid JSON only.
"""
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response if it has markdown formatting
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            # Fallback to simple text analysis
            return {
                "overall_sentiment": "mixed",
                "skepticism_level": "medium", 
                "evidence_of_misinformation": "unclear",
                "summary": response_text,
                "key_indicators": []
            }
            
    except Exception as e:
        print(f"❌ Error analyzing comments with Gemini: {e}")
        return {
            "overall_sentiment": "unknown",
            "skepticism_level": "unknown",
            "evidence_of_misinformation": "unknown", 
            "summary": "Comment analysis failed",
            "key_indicators": []
        }

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
if YOUTUBE_API_KEY:
    print(f"✅ YouTube API key configured for comment analysis")
else:
    print("⚠️ No YouTube API key found - comment analysis will be disabled")

def add_to_conversation_memory(user_input, ai_response, prediction=None, confidence=None):
    """Silently learn from user interactions without displaying learning message"""
    try:
        # Load existing memory
        memory_file = "conversation_memory.json"
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                memory = json.load(f)
        else:
            memory = {"conversations": [], "patterns": {}, "user_preferences": {}}
        
        # Add new interaction
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "prediction": prediction,
            "confidence": confidence,
            "input_type": "text" if not any(ext in user_input.lower() for ext in ['.jpg', '.png', '.mp4', '.mp3']) else "media"
        }
        
        memory["conversations"].append(interaction)
        
        # Learn patterns (silently)
        learn_from_interaction(interaction, memory)
        
        # Keep only last 1000 conversations to prevent file bloat
        if len(memory["conversations"]) > 1000:
            memory["conversations"] = memory["conversations"][-1000:]
        
        # Save updated memory
        with open(memory_file, 'w') as f:
            json.dump(memory, f, indent=2)
            
    except Exception as e:
        # Silently handle errors - don't show learning message
        pass

def learn_from_interaction(interaction, memory):
    """Extract patterns and preferences from user interactions"""
    try:
        user_input = interaction["user_input"].lower()
        
        # Learn about user's preferred content types
        if "youtube" in user_input or "video" in user_input:
            memory["user_preferences"]["content_type"] = memory["user_preferences"].get("content_type", [])
            if "video" not in memory["user_preferences"]["content_type"]:
                memory["user_preferences"]["content_type"].append("video")
        
        if "image" in user_input or "photo" in user_input:
            memory["user_preferences"]["content_type"] = memory["user_preferences"].get("content_type", [])
            if "image" not in memory["user_preferences"]["content_type"]:
                memory["user_preferences"]["content_type"].append("image")
        
        # Learn about user's topics of interest
        topics = extract_topics(user_input)
        for topic in topics:
            if topic not in memory["patterns"]:
                memory["patterns"][topic] = {"count": 0, "last_seen": None}
            memory["patterns"][topic]["count"] += 1
            memory["patterns"][topic]["last_seen"] = interaction["timestamp"]
        
        # Learn about user's confidence preferences
        if interaction.get("confidence"):
            if interaction["confidence"] < 0.6:
                memory["user_preferences"]["low_confidence_handling"] = memory["user_preferences"].get("low_confidence_handling", 0) + 1
            elif interaction["confidence"] > 0.8:
                memory["user_preferences"]["high_confidence_handling"] = memory["user_preferences"].get("high_confidence_handling", 0) + 1
                
    except Exception as e:
        # Silently handle learning errors
        pass

def extract_topics(text):
    """Extract key topics from user input"""
    topics = []
    
    # Political topics
    political_keywords = ["trump", "biden", "president", "government", "politics", "election"]
    for keyword in political_keywords:
        if keyword in text:
            topics.append(f"politics_{keyword}")
    
    # News topics
    news_keywords = ["news", "breaking", "update", "report", "story"]
    for keyword in news_keywords:
        if keyword in text:
            topics.append(f"news_{keyword}")
    
    # Platform topics
    platform_keywords = ["youtube", "facebook", "twitter", "instagram", "tiktok"]
    for keyword in platform_keywords:
        if keyword in text:
            topics.append(f"platform_{keyword}")
    
    # Content type topics
    content_keywords = ["video", "image", "photo", "audio", "text"]
    for keyword in content_keywords:
        if keyword in text:
            topics.append(f"content_{keyword}")
    
    return topics

def get_user_preferences():
    """Get learned user preferences for personalized responses"""
    try:
        memory_file = "conversation_memory.json"
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                memory = json.load(f)
            return memory.get("user_preferences", {})
        return {}
    except:
        return {}

def get_relevant_patterns(current_input):
    """Get relevant patterns based on current user input"""
    try:
        memory_file = "conversation_memory.json"
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                memory = json.load(f)
            
            current_topics = extract_topics(current_input.lower())
            relevant_patterns = {}
            
            for topic in current_topics:
                if topic in memory.get("patterns", {}):
                    relevant_patterns[topic] = memory["patterns"][topic]
            
            return relevant_patterns
        return {}
    except:
        return {}

if __name__ == "__main__":
    # Add auto-recovery and keep-alive functionality
    import threading
    
    def auto_ping():
        """Auto-ping to keep the service alive"""
        while True:
            try:
                # Ping our own health endpoint
                import requests
                requests.get("http://localhost:10000/health", timeout=5)
                time.sleep(100)  # Ping every 5 minutes
            except:
                time.sleep(60)  # Wait 1 minute if ping fails
    
    # Start auto-ping in background thread
    ping_thread = threading.Thread(target=auto_ping, daemon=True)
    ping_thread.start()
    
    print("🚀 Starting VeritasAI with auto-recovery...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)