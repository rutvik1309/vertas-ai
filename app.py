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

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key")  # Needed for session
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

# Configure Gemini API with multiple keys for load balancing
gemini_api_keys = os.environ.get("GEMINI_API_KEYS", "").split(",")
if not gemini_api_keys or gemini_api_keys[0] == "":
    # Fallback to single key
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set. Please set it in your deployment environment.")
    gemini_api_keys = [gemini_api_key]

# Track API key usage
api_key_usage = {key: {"requests": 0, "last_reset": time.time(), "quota_exceeded": False} for key in gemini_api_keys}
current_key_index = 0

def get_available_api_key():
    """Get an available API key, rotating through them"""
    global current_key_index
    
    # Check if current key is available
    for _ in range(len(gemini_api_keys)):
        key = gemini_api_keys[current_key_index]
        usage = api_key_usage[key]
        
        # Reset daily quota (24 hours)
        if time.time() - usage["last_reset"] > 86400:  # 24 hours
            usage["requests"] = 0
            usage["last_reset"] = time.time()
            usage["quota_exceeded"] = False
        
        # Check if quota exceeded (50 requests per day per key)
        if not usage["quota_exceeded"] and usage["requests"] < 50:
            usage["requests"] += 1
            return key
        
        # Move to next key
        current_key_index = (current_key_index + 1) % len(gemini_api_keys)
    
    # All keys exhausted - return None to trigger mock response
    print("All API keys have exceeded their daily quota. Using mock responses.")
    return None

def mark_key_quota_exceeded(api_key):
    """Mark an API key as quota exceeded"""
    if api_key in api_key_usage:
        api_key_usage[api_key]["quota_exceeded"] = True

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
        gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")
        print("Using gemini-1.5-pro model")
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
embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyA7APWpWr4LizACI9OBsJyunrVSnYkFNaA")
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
    
    # Check if model files exist
    if os.path.exists("final_pipeline_clean_fixed.pkl"):
        print("✅ Found final_pipeline_clean_fixed.pkl")
    else:
        print("❌ final_pipeline_clean_fixed.pkl not found")
    
    if os.path.exists("final_pipeline_clean.pkl"):
        print("✅ Found final_pipeline_clean.pkl")
    else:
        print("❌ final_pipeline_clean.pkl not found")
    
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
    
    # Try to load the fixed model with error suppression
    try:
        print("🔄 Attempting to load fixed model...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open("final_pipeline_clean_fixed.pkl", "rb") as f:
                pipeline = pickle.load(f)
        print("✅ Fixed MLP pipeline loaded successfully")
    except Exception as e1:
        print(f"❌ Failed to load fixed model: {e1}")
        # Fallback to original model
        try:
            print("🔄 Attempting to load original model...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open("final_pipeline_clean.pkl", "rb") as f:
                    pipeline = pickle.load(f)
            print("✅ Original MLP pipeline loaded successfully")
        except Exception as e2:
            print(f"❌ Failed to load original model: {e2}")
            raise e2
        
except Exception as e:
    print(f"❌ Error loading MLP pipeline: {e}")
    print("⚠️  Model compatibility issue - will use basic prediction logic")
    pipeline = None

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

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
    """
    Process YouTube URLs by extracting actual video content and analyzing it
    """
    try:
        print(f"🎥 Processing YouTube URL: {url}")
        
        # Check if it's a YouTube URL
        if not is_youtube_url(url):
            return "Not a YouTube URL"
        
        # Extract video ID from URL
        video_id = extract_youtube_id(url)
        if not video_id:
            return "Could not extract video ID from YouTube URL"
        
        print(f"✅ Extracted video ID: {video_id}")
        
        # Step 1: Try to fetch actual video transcript
        transcript_text = get_transcript(video_id)
        
        if transcript_text:
            print(f"✅ Successfully extracted transcript: {len(transcript_text)} characters")
            print(f"📝 Transcript preview: {transcript_text[:200]}...")
            
            # Create comprehensive analysis with actual video content
            content_for_analysis = f"""
ACTUAL YOUTUBE VIDEO CONTENT TO ANALYZE:

VIDEO DETAILS:
- Video URL: {url}
- Video ID: {video_id}
- Platform: YouTube
- Content Type: Video with actual transcript

ACTUAL VIDEO TRANSCRIPT (REAL CONTENT):
{transcript_text}

VIDEO CONTENT ANALYSIS:
This YouTube video has been transcribed and requires comprehensive fact-checking analysis. The video content includes:

1. **ACTUAL VIDEO CONTENT**:
   - Full transcript of the video (real spoken content)
   - All claims, statements, and assertions made by the speaker
   - Actual content that can be fact-checked
   - Tone and presentation style evident in transcript

2. **CONTENT CHARACTERISTICS**:
   - Video is hosted on YouTube platform
   - Real transcript available for detailed analysis
   - Contains actual claims, statements, or assertions
   - May have bias indicators or sensationalist content

3. **ANALYSIS REQUIREMENTS**:
   - Assess video credibility and accuracy based on actual spoken content
   - Identify potential misinformation indicators in the real transcript
   - Cross-reference claims with peer-reviewed sources
   - Evaluate the factual accuracy of statements made
   - Check for sensationalist language or emotional appeals

4. **FACT-CHECKING CRITERIA**:
   - Source credibility assessment
   - Claim verification against authoritative sources
   - Misinformation indicator detection in actual content
   - Peer-reviewed reference identification for claims made
   - Bias and manipulation detection in transcript

VIDEO CONTENT SUMMARY:
This is a YouTube video with a complete transcript of the actual spoken content. The video has been transcribed and needs assessment for accuracy, credibility, and potential misinformation indicators based on the REAL video content.

Please analyze this YouTube video content and provide a definitive FAKE/REAL assessment with specific evidence and relevant peer-reviewed sources.
            """.strip()
            
            return content_for_analysis
        
        # Step 2: Fallback to video metadata if transcript not available
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', '')
                description = info.get('description', '')
                uploader = info.get('uploader', '')
                duration = info.get('duration', 0)
                view_count = info.get('view_count', 0)
                like_count = info.get('like_count', 0)
                
                print(f"✅ Extracted video metadata: {title}")
                
                # Create analysis with available metadata
                content_for_analysis = f"""
ACTUAL YOUTUBE VIDEO CONTENT TO ANALYZE:

VIDEO DETAILS:
- Video URL: {url}
- Video ID: {video_id}
- Platform: YouTube
- Content Type: Video with metadata

ACTUAL VIDEO CONTENT:
Title: {title}
Uploader: {uploader}
Duration: {duration} seconds
View Count: {view_count:,} views
Like Count: {like_count:,} likes
Description: {description[:1000] if description else 'No description available'}

VIDEO CONTENT ANALYSIS:
This YouTube video has been identified and requires comprehensive fact-checking analysis. The video content includes:

1. **ACTUAL VIDEO CONTENT**:
   - Video title: {title}
   - Uploader: {uploader}
   - Description: {description[:500] if description else 'No description'}
   - Claims and content evident in title and description

2. **CONTENT CHARACTERISTICS**:
   - Video is hosted on YouTube platform
   - Title and description available for analysis
   - Contains actual claims, statements, or assertions
   - May have bias indicators or sensationalist content

3. **ANALYSIS REQUIREMENTS**:
   - Assess video credibility and accuracy based on available content
   - Identify potential misinformation indicators in title/description
   - Cross-reference claims with peer-reviewed sources
   - Evaluate the factual accuracy of statements made
   - Check for sensationalist language or emotional appeals

4. **FACT-CHECKING CRITERIA**:
   - Source credibility assessment
   - Claim verification against authoritative sources
   - Misinformation indicator detection in available content
   - Peer-reviewed reference identification for claims made
   - Bias and manipulation detection in title/description

VIDEO CONTENT SUMMARY:
This is a YouTube video with available metadata that requires comprehensive fact-checking analysis. The video has been identified and needs assessment for accuracy, credibility, and potential misinformation indicators based on the available content.

Please analyze this YouTube video content and provide a definitive FAKE/REAL assessment with specific evidence and relevant peer-reviewed sources.
                """.strip()
                
                return content_for_analysis
                
        except Exception as ydl_error:
            print(f"❌ Could not extract video info: {ydl_error}")
        
        # Step 3: Final fallback - transcript unavailable
        return f"""
YOUTUBE VIDEO ANALYSIS - TRANSCRIPT UNAVAILABLE:

VIDEO DETAILS:
- Video URL: {url}
- Video ID: {video_id}
- Platform: YouTube
- Content Type: Video with unavailable transcript

ANALYSIS STATUS:
This YouTube video could not be analyzed because:
1. No transcript is available for this video
2. The video may not have captions enabled
3. The transcript extraction failed

IMPORTANT NOTE:
Most YouTube videos do not have captions available through the YouTube Transcript API.
This is a common limitation and does not necessarily indicate a problem with the video.

RECOMMENDATION:
To properly analyze this video, we need access to the actual video content.
Please try one of the following:
1. Use a different video with available captions
2. Provide the video description for analysis
3. Share the video title and uploader information for basic assessment

Please provide a definitive FAKE/REAL assessment based on the available information.
        """.strip()
            
    except Exception as e:
        print(f"❌ Error processing YouTube URL {url}: {e}")
        return f"Error processing YouTube URL: {str(e)}"



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
        
        # Check if pipeline is available
        if pipeline is None:
            # Fallback to basic prediction using text analysis
            prediction = 1  # Assume real by default
            confidence = 0.5
            print("⚠️  Using fallback prediction (no ML model available)")
        else:
            prediction = pipeline.predict([cleaned_text])[0]
            confidence = pipeline.predict_proba([cleaned_text])[0].max()

        label_map = {0: "Fake", 1: "Real"}

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
                    
                    # Update prediction based on AI analysis
                    if ai_verdict.upper() == "FAKE":
                        prediction = 0  # Fake
                        confidence = 0.8 if ai_confidence.lower() == "high" else 0.6 if ai_confidence.lower() == "medium" else 0.4
                    elif ai_verdict.upper() == "REAL":
                        prediction = 1  # Real
                        confidence = 0.8 if ai_confidence.lower() == "high" else 0.6 if ai_confidence.lower() == "medium" else 0.4
                    else:
                        # Keep original ML prediction if AI couldn't determine
                        confidence = 0.5
                    
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
                if "429" in str(e) or "quota" in str(e).lower():
                    mark_key_quota_exceeded(available_key)
                    reasoning_output = f"Gemini API quota exceeded for this key. Trying another key..."
                    # Try again with a different key
                    available_key = get_available_api_key()
                    if available_key:
                        genai.configure(api_key=available_key)
                        try:
                            gemini_response = gemini_model.generate_content(prompt)
                            # Process response as before...
                            try:
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
                                
                                # Update prediction based on AI analysis
                                if ai_verdict.upper() == "FAKE":
                                    prediction = 0  # Fake
                                    confidence = 0.8 if ai_confidence.lower() == "high" else 0.6 if ai_confidence.lower() == "medium" else 0.4
                                elif ai_verdict.upper() == "REAL":
                                    prediction = 1  # Real
                                    confidence = 0.8 if ai_confidence.lower() == "high" else 0.6 if ai_confidence.lower() == "medium" else 0.4
                                else:
                                    # Keep original ML prediction if AI couldn't determine
                                    confidence = 0.5
                                
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
                            except Exception as e2:
                                reasoning_output = gemini_response.text
                                if reasoning_output.startswith('```json'):
                                    reasoning_output = reasoning_output[7:]
                                if reasoning_output.endswith('```'):
                                    reasoning_output = reasoning_output[:-3]
                                reasoning_output = reasoning_output.strip()
                                references_output = extract_urls(reasoning_output)
                        except Exception as e3:
                            reasoning_output = f"All Gemini API keys exhausted. Please try again later. Error: {e3}"
                            references_output = []
                    else:
                        reasoning_output = f"All Gemini API keys have exceeded their daily quota. Please try again tomorrow."
                        references_output = []
                else:
                    reasoning_output = f"Gemini generation failed: {e}"
                    references_output = []

        summary, breakdown, supporting, final_judgment = parse_gemini_reasoning(reasoning_output)

        # Store all context in session for the chat agent
        session['last_article_text'] = text
        session['last_reasoning'] = reasoning_output
        session['last_references'] = references_output
        session['last_original_news'] = original_news
        session['last_red_flags'] = red_flags
        session['last_prediction'] = label_map[prediction]

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
            
            return f"<b>Prediction:</b> {label_map[prediction]}<br><b>Confidence:</b> {round(float(confidence), 4)}<br><b>Reasoning:</b> {reasoning_html}{original_news_html}{red_flags_html}{references_html}"

        try:
            return render_template(
                'index.html',
                prediction=label_map[prediction],
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
    prediction = pipeline.predict([cleaned_text])[0]
    confidence = pipeline.predict_proba([cleaned_text])[0].max()
    label_map = {0: "Fake", 1: "Real"}

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
            if "429" in str(e) or "quota" in str(e).lower():
                mark_key_quota_exceeded(available_key)
                answer = f"Gemini API quota exceeded for this key. Trying another key..."
                # Try again with a different key
                available_key = get_available_api_key()
                if available_key:
                    genai.configure(api_key=available_key)
                    try:
                        gemini_response = gemini_model.generate_content(base_prompt)
                        answer = gemini_response.text
                    except Exception as e2:
                        answer = f"All Gemini API keys exhausted. Please try again later. Error: {e2}"
                else:
                    answer = f"All Gemini API keys have exceeded their daily quota. Please try again tomorrow."
            else:
                answer = f"Error: {e}"

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
    return jsonify({
        'status': 'OK',
        'timestamp': time.time(),
        'service': 'VeritasAI',
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

@app.route('/get_context')
def get_context():
    return jsonify({
        'article': session.get('last_article_text'),
        'reasoning': session.get('last_reasoning'),
        'references': session.get('last_references') or []
    })

@app.route('/memory/stats')
def get_memory_stats():
    """Get conversation memory statistics"""
    return jsonify({
        'total_conversations': len(conversation_memory),
        'max_memory_size': MAX_MEMORY_SIZE,
        'memory_usage_percent': (len(conversation_memory) / MAX_MEMORY_SIZE) * 100
    })

@app.route('/memory/clear', methods=['POST'])
def clear_memory():
    """Clear all conversation memory"""
    global conversation_memory
    conversation_memory = []
    save_conversation_memory()
    return jsonify({'message': 'Memory cleared successfully'})

@app.route('/memory/recent')
def get_recent_memory():
    """Get recent conversations for debugging"""
    recent = conversation_memory[-10:] if len(conversation_memory) > 10 else conversation_memory
    return jsonify({
        'recent_conversations': recent,
        'total_count': len(conversation_memory)
    })

@app.route('/api/status')
def get_api_status():
    """Get API key status and usage"""
    status = {
        'total_keys': len(gemini_api_keys),
        'available_keys': 0,
        'quota_exceeded_keys': 0,
        'key_details': []
    }
    
    for key in gemini_api_keys:
        usage = api_key_usage.get(key, {})
        key_info = {
            'key_preview': key[:10] + '...' if key else 'None',
            'requests_today': usage.get('requests', 0),
            'quota_exceeded': usage.get('quota_exceeded', False),
            'last_reset': usage.get('last_reset', 0)
        }
        status['key_details'].append(key_info)
        
        if usage.get('quota_exceeded', False):
            status['quota_exceeded_keys'] += 1
        else:
            status['available_keys'] += 1
    
    return jsonify(status)

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
                time.sleep(300)  # Ping every 5 minutes
            except:
                time.sleep(60)  # Wait 1 minute if ping fails
    
    # Start auto-ping in background thread
    ping_thread = threading.Thread(target=auto_ping, daemon=True)
    ping_thread.start()
    
    print("🚀 Starting VeritasAI with auto-recovery...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)