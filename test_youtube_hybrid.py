#!/usr/bin/env python3
"""
Test the YouTube hybrid analysis fix
"""

def test_youtube_hybrid_analysis():
    """Test the new YouTube hybrid analysis approach"""
    
    print("🔍 Testing YouTube Hybrid Analysis Fix")
    print("=" * 60)
    
    # Test cases for YouTube content
    test_cases = [
        {
            "name": "CBC News YouTube Video",
            "url": "https://youtube.com/watch?v=abc123",
            "text": "Video title: CBC News - GOP senator calls for extension on US-Canada trade deal deadline. Video description: A Republican senator is calling for an extension on the deadline for a trade deal between the US and Canada. This video was uploaded by CBC News. The video has 50000 views. The video has 1200 likes. The video is 3 minutes and 45 seconds long. Video tags include: news, politics, trade, canada, us. Video transcript: The Republican senator from North Dakota is calling for an extension on the deadline for the US-Canada trade deal."
        },
        {
            "name": "BNN Bloomberg YouTube Video",
            "url": "https://youtube.com/watch?v=def456", 
            "text": "Video title: BNN Bloomberg - Trade lawyer discusses Trump's potential tariffs on Canada. Video description: A trade lawyer discusses President Trump's potential unilateral tariffs on Canada. This video was uploaded by BNN Bloomberg. The video has 75000 views. The video has 1800 likes. The video is 4 minutes and 20 seconds long. Video tags include: business, trade, trump, canada, tariffs. Video transcript: The trade lawyer explains the potential impact of President Trump's proposed tariffs on Canadian goods."
        },
        {
            "name": "Independent YouTube Video",
            "url": "https://youtube.com/watch?v=ghi789",
            "text": "Video title: SHOCKING BREAKING NEWS - AMAZING REVELATION EXPOSED. Video description: You won't believe what just happened! Incredible breaking news that will shock you! This video was uploaded by Independent Channel. The video has 1000000 views. The video has 50000 likes. The video is 10 minutes and 30 seconds long. Video tags include: breaking, news, shocking, amazing, viral. Video transcript: This is absolutely incredible! You won't believe what I just discovered!"
        },
        {
            "name": "Reuters Article (Non-YouTube)",
            "url": "https://reuters.com/article/123",
            "text": "Reuters reports that the Federal Reserve announced new monetary policy measures today. The central bank's decision comes after months of economic analysis and consultation with financial experts."
        }
    ]
    
    # Reputable sources list (same as in app.py)
    reputable_sources = [
        'cbc news', 'bbc news', 'reuters', 'associated press', 'ap', 
        'cnn', 'nbc news', 'abc news', 'cbs news', 'pbs news', 'npr',
        'fox news', 'msnbc', 'bloomberg', 'bnn bloomberg', 'cnbc',
        'the new york times', 'washington post', 'wall street journal',
        'usa today', 'time magazine', 'newsweek'
    ]
    
    print("\n📊 Test Results:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        url = test_case['url']
        text = test_case['text']
        
        # Check if it's YouTube
        is_youtube = "youtube.com" in url or "youtu.be" in url
        print(f"YouTube URL: {is_youtube}")
        
        if is_youtube:
            # Check for reputable sources
            text_lower = text.lower()
            has_reputable_source = any(source in text_lower for source in reputable_sources)
            
            if has_reputable_source:
                print("✅ Reputable news source detected")
                prediction = "Real"
                confidence = 0.75
            else:
                print("⚠️ No reputable source detected - checking for sensationalist content")
                
                # Check for sensationalist indicators
                sensational_words = [
                    'shocking', 'amazing', 'incredible', 'unbelievable', 'exposed', 'revealed',
                    'breaking', 'urgent', 'exclusive', 'viral', 'you won\'t believe', 'incredible'
                ]
                
                sensational_count = sum(1 for word in sensational_words if word in text_lower)
                
                if sensational_count >= 2:
                    print("🚨 Sensationalist content detected")
                    prediction = "Fake"
                    confidence = 0.8
                else:
                    print("⚠️ Unknown source - using AI analysis")
                    prediction = "Fake"  # Default for unknown sources
                    confidence = 0.5
        else:
            # Non-YouTube content - would use regular ML model
            print("📝 Non-YouTube content - would use regular ML model")
            prediction = "Real"  # Based on previous test results
            confidence = 0.55
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")
        
        # Show which reputable sources were found
        if is_youtube:
            found_sources = [source for source in reputable_sources if source in text_lower]
            if found_sources:
                print(f"Found reputable sources: {', '.join(found_sources)}")
            else:
                print("No reputable sources found in content")

if __name__ == "__main__":
    test_youtube_hybrid_analysis() 