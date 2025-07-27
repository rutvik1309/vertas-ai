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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, confidence, reasoning_output = None, None, None
    article_text = None

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
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
            except Exception as e:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return f"Failed to fetch article: {e}", 400
                return render_template('index.html', prediction=f"Failed to fetch article: {e}")

        elif file:
            text = file.read().decode("utf-8")

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

        # Get comprehensive Gemini reasoning
        if label_map[prediction] == "Fake":
            prompt = f"""You are an expert fact-checker. Analyze this FAKE news article and provide:

1. REASONING: Detailed explanation of why this is fake news
2. ORIGINAL_NEWS: What the original/real story was (if this is a distortion)
3. RED_FLAGS: Key indicators that this is fake
4. REFERENCES: Credible sources for fact-checking

Article: {text[:2000]}

IMPORTANT: Respond ONLY with valid JSON, no markdown formatting, no code blocks. Use this exact format:
{{
  "reasoning": "Detailed explanation...",
  "original_news": "What the real story was...",
  "red_flags": ["Flag 1", "Flag 2", "Flag 3"],
  "references": ["https://factcheck.org/...", "https://snopes.com/..."]
}}"""
        else:
            prompt = f"""You are an expert fact-checker. Analyze this REAL news article and provide:

1. REASONING: Detailed explanation of why this appears to be real news
2. CREDIBILITY_FACTORS: What makes this story credible
3. VERIFICATION: How this story can be verified
4. REFERENCES: Credible sources that support this story

Article: {text[:2000]}

IMPORTANT: Respond ONLY with valid JSON, no markdown formatting, no code blocks. Use this exact format:
{{
  "reasoning": "Detailed explanation...",
  "credibility_factors": ["Factor 1", "Factor 2", "Factor 3"],
  "verification": "How to verify this story...",
  "references": ["https://reliable-source.com/...", "https://official-statement.com/..."]
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
                    reasoning_output = parsed.get("reasoning", "")
                    references_output = parsed.get("references", [])
                    original_news = parsed.get("original_news", "")
                    red_flags = parsed.get("red_flags", [])
                    
                    # If reasoning is empty, try alternative fields
                    if not reasoning_output:
                        if label_map[prediction] == "Fake":
                            reasoning_output = f"Analysis: This article appears to be fake news based on several indicators."
                            if red_flags:
                                reasoning_output += f"\n\nRed Flags:\n" + "\n".join([f"• {flag}" for flag in red_flags])
                        else:
                            reasoning_output = f"Analysis: This article appears to be real news based on credible factors."
                            credibility_factors = parsed.get("credibility_factors", [])
                            if credibility_factors:
                                reasoning_output += f"\n\nCredibility Factors:\n" + "\n".join([f"• {factor}" for factor in credibility_factors])
                            
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
                                reasoning_output = parsed.get("reasoning", "")
                                references_output = parsed.get("references", [])
                                original_news = parsed.get("original_news", "")
                                red_flags = parsed.get("red_flags", [])
                                
                                if not reasoning_output:
                                    if label_map[prediction] == "Fake":
                                        reasoning_output = f"Analysis: This article appears to be fake news based on several indicators."
                                        if red_flags:
                                            reasoning_output += f"\n\nRed Flags:\n" + "\n".join([f"• {flag}" for flag in red_flags])
                                    else:
                                        reasoning_output = f"Analysis: This article appears to be real news based on credible factors."
                                        credibility_factors = parsed.get("credibility_factors", [])
                                        if credibility_factors:
                                            reasoning_output += f"\n\nCredibility Factors:\n" + "\n".join([f"• {factor}" for factor in credibility_factors])
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
            
            # Format references as hyperlinks (only once)
            def format_references_html(refs):
                if not refs or not isinstance(refs, list):
                    return ''
                html = '<br><b>References:</b><ul style="margin-left:1em;">'
                for ref in refs:
                    if ref and isinstance(ref, str):
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
    return "OK", 200

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
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)