#!/usr/bin/env python3

import sys
from youtube_transcript_api import YouTubeTranscriptApi

def test_transcript_extraction(video_id):
    """Test transcript extraction for a given video ID"""
    try:
        print(f"Testing transcript extraction for video ID: {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript])
        print(f"✅ SUCCESS: Extracted {len(transcript_text)} characters")
        print(f"Preview: {transcript_text[:200]}...")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test with a few different video IDs
    test_videos = [
        "jqVsY9DhAxA",  # Original test video
        "dQw4w9WgXcQ",  # Rick Roll
        "jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
        "9bZkp7q19f0",  # Gangnam Style (popular video with captions)
    ]
    
    print("Testing YouTube Transcript API...")
    print("=" * 50)
    
    success_count = 0
    for video_id in test_videos:
        if test_transcript_extraction(video_id):
            success_count += 1
        print("-" * 30)
    
    print(f"Results: {success_count}/{len(test_videos)} videos had available transcripts")
    
    if success_count == 0:
        print("❌ No transcripts available - this might indicate an API issue")
    elif success_count < len(test_videos):
        print("⚠️  Some videos don't have transcripts available")
    else:
        print("✅ All videos have transcripts available") 