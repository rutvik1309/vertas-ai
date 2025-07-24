from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
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

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask("VeritasAI")
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure Gemini API
genai.configure(api_key="AIzaSyA7APWpWr4LizACI9OBsJyunrVSnYkFNaA")
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash-exp")

# Connect to ChromaDB
chroma_settings = Settings(is_persistent=False)



chroma_client = Client(settings=chroma_settings)
embedding_func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key="AIzaSyA7APWpWr4LizACI9OBsJyunrVSnYkFNaA")
collection = chroma_client.get_or_create_collection(name="news_articles", embedding_function=embedding_func)

# Custom feature functions (imported from features.py or defined inline)
from features import text_length_func, unique_words_func, avg_word_length_func, sentence_count_func

# Load your MLP pipeline
with open("final_pipeline_clean.pkl", "rb") as f:
    pipeline = pickle.load(f)

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, confidence, reasoning_output = None, None, None

    if request.method == 'POST':
        url = request.form.get('article_url')
        file = request.files.get('file')
        text = None

        if url:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
            except Exception as e:
                return render_template('index.html', prediction=f"Failed to fetch article: {e}")

        elif file:
            text = file.read().decode("utf-8")

        if not text:
            return render_template('index.html', prediction="No URL or file provided.")

        cleaned_text = clean_text(text)
        prediction = pipeline.predict([cleaned_text])[0]
        confidence = pipeline.predict_proba([cleaned_text])[0].max()

        label_map = {0: "Fake", 1: "Real"}

        # Get Gemini reasoning
        prompt = f"Provide reasoning and credible references for why this article is predicted as {label_map[prediction]}. Here is the article text:\n\n{text}"

        try:
            gemini_response = gemini_model.generate_content(prompt)
            reasoning_output = gemini_response.text
        except Exception as e:
            reasoning_output = f"Gemini generation failed: {e}"

        return render_template(
            'index.html',
            prediction=label_map[prediction],
            confidence=round(float(confidence), 4),
            reasoning=reasoning_output
        )

    return render_template('index.html')


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)