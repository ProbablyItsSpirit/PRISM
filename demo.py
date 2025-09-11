#!/usr/bin/env python3
"""
PRISM Demo without Heavy Dependencies

This script demonstrates PRISM functionality using lightweight components
that don't require PyTorch or transformers to be installed.
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockMuRILModel:
    """Mock MuRIL model for demonstration without transformers."""
    
    def __init__(self, model_name: str = "mock-muril"):
        self.model_name = model_name
        self.device = "cpu"
        
    def encode_text(self, text: str):
        """Return mock embeddings."""
        import random
        # Return mock 768-dim embeddings
        return [[random.random() for _ in range(768)]]
    
    def encode_batch(self, texts: List[str]):
        """Return mock batch embeddings."""
        import random
        return [[random.random() for _ in range(768)] for _ in texts]
    
    def is_available(self) -> bool:
        return True


class DemoAnalyzer:
    """Lightweight demo analyzer without PyTorch dependencies."""
    
    def __init__(self):
        """Initialize demo analyzer."""
        self.model = MockMuRILModel()
        
        # Hate speech keywords for demo
        self.hate_keywords = {
            'english': ['hate', 'stupid', 'idiot', 'terrible', 'awful', 'disgusting'],
            'hindi': ['बकवास', 'घटिया', 'बुरा'],
            'mixed': ['bakwas', 'ghatiya', 'bura']
        }
        
        # Mood keywords for demo
        self.mood_keywords = {
            'happy': ['happy', 'joy', 'excited', 'wonderful', 'खुश', 'आनंद'],
            'sad': ['sad', 'cry', 'disappointed', 'उदास', 'दुख'],
            'angry': ['angry', 'mad', 'furious', 'गुस्सा', 'क्रोध'],
            'neutral': ['okay', 'fine', 'normal', 'ठीक']
        }
        
        # Sentiment keywords for demo
        self.sentiment_keywords = {
            'positive': ['good', 'great', 'amazing', 'love', 'wonderful', 'अच्छा', 'बढ़िया'],
            'negative': ['bad', 'terrible', 'hate', 'awful', 'बुरा', 'खराब'],
            'neutral': ['okay', 'fine', 'normal', 'ठीक']
        }
    
    def detect_hate_speech(self, text: str) -> Dict[str, Any]:
        """Simple hate speech detection."""
        text_lower = text.lower()
        hate_count = 0
        detected_keywords = []
        
        for lang_keywords in self.hate_keywords.values():
            for keyword in lang_keywords:
                if keyword.lower() in text_lower:
                    hate_count += 1
                    detected_keywords.append(keyword)
        
        score = min(hate_count / max(len(text.split()), 1) * 3, 1.0)
        is_hate = score > 0.3
        
        return {
            'is_hate_speech': is_hate,
            'confidence': abs(score - 0.5) * 2,
            'score': score,
            'detected_keywords': detected_keywords,
            'severity': 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
        }
    
    def analyze_mood(self, text: str) -> Dict[str, Any]:
        """Simple mood analysis."""
        text_lower = text.lower()
        mood_scores = {}
        
        for mood, keywords in self.mood_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            mood_scores[mood] = score
        
        if not any(mood_scores.values()):
            primary_mood = 'neutral'
            confidence = 0.5
        else:
            primary_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
            confidence = mood_scores[primary_mood] / max(sum(mood_scores.values()), 1)
        
        return {
            'primary_mood': primary_mood,
            'confidence': confidence,
            'mood_distribution': mood_scores,
            'intensity': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis."""
        text_lower = text.lower()
        sentiment_scores = {}
        
        for sentiment, keywords in self.sentiment_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            sentiment_scores[sentiment] = score
        
        if not any(sentiment_scores.values()):
            primary_sentiment = 'neutral'
            confidence = 0.5
        else:
            primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = sentiment_scores[primary_sentiment] / max(sum(sentiment_scores.values()), 1)
        
        # Calculate valence
        pos_score = sentiment_scores.get('positive', 0)
        neg_score = sentiment_scores.get('negative', 0)
        total = pos_score + neg_score
        valence = (pos_score - neg_score) / max(total, 1)
        
        return {
            'primary_sentiment': primary_sentiment,
            'polarity': primary_sentiment,
            'confidence': confidence,
            'sentiment_distribution': sentiment_scores,
            'valence': valence,
            'intensity': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
        }
    
    def detect_language_mix(self, text: str) -> Dict[str, Any]:
        """Detect if text is code-mixed."""
        import re
        
        # Count different scripts
        latin_chars = len(re.findall(r'[A-Za-z]', text))
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_chars = latin_chars + devanagari_chars
        
        if total_chars == 0:
            return {'is_code_mixed': False, 'languages': {}}
        
        ratios = {
            'english': latin_chars / total_chars,
            'hindi': devanagari_chars / total_chars
        }
        
        is_mixed = min(ratios.values()) > 0.1  # At least 10% of each language
        
        return {
            'is_code_mixed': is_mixed,
            'language_distribution': ratios,
            'dominant_language': max(ratios.items(), key=lambda x: x[1])[0]
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Complete analysis of text."""
        if not text or not isinstance(text, str):
            return {'error': 'Invalid text input'}
        
        results = {
            'original_text': text,
            'preprocessing': self.detect_language_mix(text),
            'hate_speech': self.detect_hate_speech(text),
            'mood': self.analyze_mood(text),
            'sentiment': self.analyze_sentiment(text)
        }
        
        return results
    
    def detect_transitions(self, texts: List[str]) -> Dict[str, Any]:
        """Detect transitions between texts."""
        if len(texts) < 2:
            return {'transitions_detected': False, 'reason': 'Need at least 2 texts'}
        
        # Analyze each text
        analyses = [self.analyze(text) for text in texts]
        
        # Check for mood transitions
        mood_transitions = []
        sentiment_transitions = []
        
        for i in range(1, len(analyses)):
            prev = analyses[i-1]
            curr = analyses[i]
            
            # Mood transition
            if prev['mood']['primary_mood'] != curr['mood']['primary_mood']:
                mood_transitions.append({
                    'position': i,
                    'from_mood': prev['mood']['primary_mood'],
                    'to_mood': curr['mood']['primary_mood'],
                    'confidence_change': abs(curr['mood']['confidence'] - prev['mood']['confidence'])
                })
            
            # Sentiment transition
            if prev['sentiment']['polarity'] != curr['sentiment']['polarity']:
                sentiment_transitions.append({
                    'position': i,
                    'from_sentiment': prev['sentiment']['polarity'],
                    'to_sentiment': curr['sentiment']['polarity'],
                    'valence_change': abs(curr['sentiment']['valence'] - prev['sentiment']['valence'])
                })
        
        return {
            'transitions_detected': len(mood_transitions) > 0 or len(sentiment_transitions) > 0,
            'mood_transitions': mood_transitions,
            'sentiment_transitions': sentiment_transitions,
            'total_transitions': len(mood_transitions) + len(sentiment_transitions)
        }


def main():
    """Demonstrate PRISM functionality."""
    print("=" * 60)
    print("PRISM Demo - Code-Mixed Text Analysis")
    print("=" * 60)
    
    analyzer = DemoAnalyzer()
    
    # Test cases
    test_cases = [
        ("Code-mixed positive", "यह movie बहुत अच्छी है, I loved it!"),
        ("Code-mixed negative", "यह film बकवास है, waste of time"),
        ("English hate speech", "This is stupid and terrible"),
        ("Hindi positive", "यह फिल्म बहुत शानदार है"),
        ("Neutral mixed", "Main kal office जाऊंगा"),
    ]
    
    print("Individual Text Analysis:")
    print("-" * 40)
    
    for test_name, text in test_cases:
        print(f"\nTest: {test_name}")
        print(f"Text: {text}")
        
        results = analyzer.analyze(text)
        
        if 'error' not in results:
            print(f"  Code-mixed: {results['preprocessing']['is_code_mixed']}")
            print(f"  Languages: {results['preprocessing']['language_distribution']}")
            print(f"  Hate speech: {results['hate_speech']['is_hate_speech']} (confidence: {results['hate_speech']['confidence']:.2f})")
            print(f"  Mood: {results['mood']['primary_mood']} (confidence: {results['mood']['confidence']:.2f})")
            print(f"  Sentiment: {results['sentiment']['polarity']} (valence: {results['sentiment']['valence']:.2f})")
        else:
            print(f"  Error: {results['error']}")
    
    # Transition detection demo
    print("\n" + "=" * 60)
    print("Transition Detection Demo:")
    print("-" * 40)
    
    transition_sequence = [
        "I was so excited about this movie!",
        "But then the story became confusing",
        "By the end, I was completely disappointed",
        "However, the music was absolutely amazing"
    ]
    
    print("Analyzing emotional transition sequence:")
    for i, text in enumerate(transition_sequence):
        print(f"  {i+1}. {text}")
    
    transition_results = analyzer.detect_transitions(transition_sequence)
    
    print(f"\nTransition Results:")
    print(f"  Transitions detected: {transition_results['transitions_detected']}")
    print(f"  Total transitions: {transition_results['total_transitions']}")
    
    if transition_results['mood_transitions']:
        print(f"  Mood transitions:")
        for trans in transition_results['mood_transitions']:
            print(f"    Position {trans['position']}: {trans['from_mood']} → {trans['to_mood']}")
    
    if transition_results['sentiment_transitions']:
        print(f"  Sentiment transitions:")
        for trans in transition_results['sentiment_transitions']:
            print(f"    Position {trans['position']}: {trans['from_sentiment']} → {trans['to_sentiment']}")
    
    # Save demo results
    demo_results = {
        'individual_analyses': [],
        'transition_analysis': transition_results
    }
    
    for test_name, text in test_cases:
        result = analyzer.analyze(text)
        result['test_name'] = test_name
        demo_results['individual_analyses'].append(result)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    with open('output/demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 60)
    print("Demo completed! Results saved to output/demo_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()