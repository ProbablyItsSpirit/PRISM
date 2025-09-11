#!/usr/bin/env python3
"""
PRISM Example Usage

This script demonstrates how to use PRISM for analyzing code-mixed text.
"""

import sys
import os

# Add the parent directory to the path so we can import prism
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism import PRISMAnalyzer
from prism.utils import create_sample_data


def main():
    """Main example function."""
    print("=" * 60)
    print("PRISM: Code-Mixed Text Analysis Example")
    print("=" * 60)
    
    # Initialize the analyzer
    print("Initializing PRISM analyzer...")
    analyzer = PRISMAnalyzer()
    
    # Check if everything is ready
    capabilities = analyzer.get_capabilities()
    print(f"MuRIL Model Available: {capabilities['muril_model_available']}")
    print(f"Device: {capabilities['device']}")
    print()
    
    # Get sample data
    sample_data = create_sample_data()
    
    # Example 1: Single text analysis
    print("Example 1: Single Text Analysis")
    print("-" * 40)
    
    text = sample_data['code_mixed'][0]
    print(f"Text: {text}")
    print()
    
    results = analyzer.analyze(text, include_transitions=True)
    
    # Display results
    if 'error' not in results:
        print("Results:")
        
        # Preprocessing info
        if results.get('preprocessing'):
            prep = results['preprocessing']
            print(f"  Code-mixed: {prep.get('is_code_mixed', False)}")
            print(f"  Languages: {prep.get('language_distribution', {})}")
        
        # Hate speech
        if results.get('hate_speech'):
            hs = results['hate_speech']
            print(f"  Hate Speech: {hs.get('is_hate_speech', False)} (confidence: {hs.get('confidence', 0):.2f})")
        
        # Mood
        if results.get('mood'):
            mood = results['mood']
            print(f"  Mood: {mood.get('primary_mood', 'unknown')} (confidence: {mood.get('confidence', 0):.2f})")
        
        # Sentiment
        if results.get('sentiment'):
            sent = results['sentiment']
            print(f"  Sentiment: {sent.get('polarity', 'unknown')} (valence: {sent.get('valence', 0):.2f})")
    else:
        print(f"Error: {results['error']}")
    
    print()
    
    # Example 2: Batch analysis with transitions
    print("Example 2: Transition Detection")
    print("-" * 40)
    
    # Use emotional transition texts
    transition_texts = sample_data['emotional_transitions']
    
    print("Analyzing emotional transition sequence:")
    for i, text in enumerate(transition_texts):
        print(f"  {i+1}. {text}")
    print()
    
    batch_results = analyzer.analyze_batch(transition_texts, include_transitions=True)
    
    # Check for batch transition analysis
    if batch_results and 'batch_metadata' in batch_results[0]:
        trans_analysis = batch_results[0]['batch_metadata']['batch_transition_analysis']
        
        print("Transition Analysis Results:")
        print(f"  Transitions detected: {trans_analysis.get('transitions_detected', False)}")
        
        if trans_analysis.get('mood_transitions'):
            print(f"  Mood transitions: {len(trans_analysis['mood_transitions'])}")
            for i, trans in enumerate(trans_analysis['mood_transitions']):
                print(f"    {trans['from_mood']} → {trans['to_mood']} (position {trans['position']})")
        
        if trans_analysis.get('sentiment_transitions'):
            print(f"  Sentiment transitions: {len(trans_analysis['sentiment_transitions'])}")
            for i, trans in enumerate(trans_analysis['sentiment_transitions']):
                print(f"    {trans['from_polarity']} → {trans['to_polarity']} (position {trans['position']})")
        
        if trans_analysis.get('emotional_volatility'):
            volatility = trans_analysis['emotional_volatility']
            print(f"  Emotional volatility: {volatility.get('overall_volatility', 0):.3f}")
    
    print()
    
    # Example 3: Conversation analysis
    print("Example 3: Conversation Analysis")
    print("-" * 40)
    
    conversation = [
        "यह movie देखने जा रहे हैं tonight",
        "Great! I'm so excited about it",
        "But I heard mixed reviews about it",
        "Well, let's see for ourselves"
    ]
    
    print("Conversation:")
    for i, msg in enumerate(conversation):
        print(f"  Person {i%2 + 1}: {msg}")
    print()
    
    conv_results = analyzer.analyze_conversation(conversation)
    
    if 'conversation_summary' in conv_results:
        summary = conv_results['conversation_summary']
        print("Conversation Summary:")
        print(f"  Messages: {summary.get('total_messages', 0)}")
        print(f"  Hate speech detected: {summary.get('hate_speech_detected', 0)}")
        print(f"  Mood distribution: {summary.get('mood_distribution', {})}")
        print(f"  Sentiment distribution: {summary.get('sentiment_distribution', {})}")
        print(f"  Transitions detected: {summary.get('transitions_detected', False)}")
    
    print()
    
    # Example 4: Generate formatted report
    print("Example 4: Formatted Report")
    print("-" * 40)
    
    sample_text = "This movie is amazing! यह फिल्म बहुत शानदार है।"
    analysis = analyzer.analyze(sample_text)
    
    if 'error' not in analysis:
        report = analyzer.get_report(analysis)
        print(report)
    
    print("\nPRISM analysis examples completed!")


if __name__ == "__main__":
    main()