#!/usr/bin/env python3
"""
Test the YouTube processing fix
"""

def test_youtube_format():
    """Test the new YouTube content format"""
    
    # Old format (problematic)
    old_format = "TITLE: CBC News - GOP senator calls for extension on US-Canada trade deal deadline [SENTIMENT: NEUTRAL] | UPLOADER: CBC News [TYPE: MAJOR_NEWS] | DESCRIPTION: A Republican senator is calling for an extension on the deadline for a trade deal between the US and Canada. [CREDIBILITY: CREDIBLE]"
    
    # New format (natural language)
    new_format = "Video title: CBC News - GOP senator calls for extension on US-Canada trade deal deadline. Video description: A Republican senator is calling for an extension on the deadline for a trade deal between the US and Canada. This video was uploaded by CBC News. The video has 50000 views. The video has 1200 likes. The video is 3 minutes and 45 seconds long. Video tags include: news, politics, trade, canada, us. Video transcript: The Republican senator from North Dakota is calling for an extension on the deadline for the US-Canada trade deal."
    
    print("🔍 Testing YouTube Content Format Fix")
    print("=" * 60)
    
    print("\n❌ OLD FORMAT (Problematic):")
    print("-" * 40)
    print(old_format)
    
    print("\n✅ NEW FORMAT (Natural Language):")
    print("-" * 40)
    print(new_format)
    
    print("\n📊 Comparison:")
    print("-" * 40)
    print(f"Old format length: {len(old_format)} characters")
    print(f"New format length: {len(new_format)} characters")
    print(f"Old format has metadata tags: {'[' in old_format and ']' in old_format}")
    print(f"New format has metadata tags: {'[' in new_format and ']' in new_format}")
    
    # Check for suspicious patterns in old format
    suspicious_patterns = ['[SENTIMENT:', '[TYPE:', '[CREDIBILITY:', '[ENGAGEMENT:']
    old_suspicious = any(pattern in old_format for pattern in suspicious_patterns)
    new_suspicious = any(pattern in new_format for pattern in suspicious_patterns)
    
    print(f"Old format has suspicious patterns: {old_suspicious}")
    print(f"New format has suspicious patterns: {new_suspicious}")
    
    print("\n🎯 Expected Result:")
    print("The new format should be much more natural and less likely to be flagged as fake by the ML model.")

if __name__ == "__main__":
    test_youtube_format() 