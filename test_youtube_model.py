#!/usr/bin/env python3
"""
Test script to debug YouTube model predictions
"""

import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

def test_model_predictions():
    """Test model predictions with different types of content"""
    
    print("🔍 Testing model predictions...")
    
    # Load the main model
    try:
        pipeline = joblib.load("final_pipeline_clean.pkl")
        print("✅ Main model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load main model: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "CBC News Video (New Format)",
            "text": "Video title: CBC News - GOP senator calls for extension on US-Canada trade deal deadline. Video description: A Republican senator is calling for an extension on the deadline for a trade deal between the US and Canada. This video was uploaded by CBC News. The video has 50000 views. The video has 1200 likes. The video is 3 minutes and 45 seconds long. Video tags include: news, politics, trade, canada, us. Video transcript: The Republican senator from North Dakota is calling for an extension on the deadline for the US-Canada trade deal."
        },
        {
            "name": "BNN Bloomberg Video (New Format)", 
            "text": "Video title: BNN Bloomberg - Trade lawyer discusses Trump's potential tariffs on Canada. Video description: A trade lawyer discusses President Trump's potential unilateral tariffs on Canada. This video was uploaded by BNN Bloomberg. The video has 75000 views. The video has 1800 likes. The video is 4 minutes and 20 seconds long. Video tags include: business, trade, trump, canada, tariffs. Video transcript: The trade lawyer explains the potential impact of President Trump's proposed tariffs on Canadian goods."
        },
        {
            "name": "Sensationalist Video",
            "text": "TITLE: SHOCKING BREAKING NEWS - AMAZING REVELATION EXPOSED [SENTIMENT: SENSATIONAL] | UPLOADER: Independent Channel [TYPE: INDEPENDENT] | DESCRIPTION: You won't believe what just happened! Incredible breaking news that will shock you! [CREDIBILITY: UNVERIFIED]"
        },
        {
            "name": "Reuters Article",
            "text": "Reuters reports that the Federal Reserve announced new monetary policy measures today. The central bank's decision comes after months of economic analysis and consultation with financial experts."
        },
        {
            "name": "Fake News Example",
            "text": "BREAKING: ALIENS CONFIRMED ON EARTH! AMAZING FOOTAGE REVEALED! You won't believe what scientists just discovered! SHOCKING TRUTH EXPOSED!"
        },
        {
            "name": "CBC News Video (Old Format - For Comparison)",
            "text": "TITLE: CBC News - GOP senator calls for extension on US-Canada trade deal deadline [SENTIMENT: NEUTRAL] | UPLOADER: CBC News [TYPE: MAJOR_NEWS] | DESCRIPTION: A Republican senator is calling for an extension on the deadline for a trade deal between the US and Canada. [CREDIBILITY: CREDIBLE]"
        }
    ]
    
    print("\n📊 Model Prediction Results:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        # Clean the text
        cleaned_text = clean_text(test_case['text'])
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        print(f"Cleaned text preview: {cleaned_text[:200]}...")
        
        # Get prediction
        try:
            prediction = pipeline.predict([cleaned_text])[0]
            confidence = pipeline.predict_proba([cleaned_text])[0].max()
            
            label = "Fake" if prediction == 0 else "Real"
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.4f}")
            
            # Show probability distribution
            probas = pipeline.predict_proba([cleaned_text])[0]
            print(f"Fake probability: {probas[0]:.4f}")
            print(f"Real probability: {probas[1]:.4f}")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
        
        print()

if __name__ == "__main__":
    test_model_predictions() 