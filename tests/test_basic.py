#!/usr/bin/env python3
"""
Basic tests for PRISM components.

This script runs basic functionality tests to ensure PRISM is working correctly.
"""

import sys
import os

# Add the parent directory to the path so we can import prism
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism import PRISMAnalyzer
from prism.preprocessing import TextCleaner, CodeMixedTokenizer
from prism.utils import create_sample_data


def test_text_cleaner():
    """Test text cleaning functionality."""
    print("Testing TextCleaner...")
    
    cleaner = TextCleaner()
    
    # Test basic cleaning
    dirty_text = "This is a test!!! हमें   यह   पसंद है... @user #hashtag http://example.com"
    cleaned = cleaner.clean(dirty_text, level="moderate")
    
    assert cleaned, "Cleaned text should not be empty"
    print(f"  Original: {dirty_text}")
    print(f"  Cleaned: {cleaned}")
    
    # Test language detection
    mixed_text = "This is English और यह हिंदी है"
    ratios = cleaner.detect_languages(mixed_text)
    
    assert 'english' in ratios, "Should detect English"
    assert 'hindi' in ratios, "Should detect Hindi"
    assert cleaner.is_code_mixed(mixed_text), "Should detect as code-mixed"
    
    print(f"  Language ratios: {ratios}")
    print("  ✓ TextCleaner tests passed")


def test_tokenizer():
    """Test tokenization functionality."""
    print("Testing CodeMixedTokenizer...")
    
    cleaner = TextCleaner()
    tokenizer = CodeMixedTokenizer(cleaner)
    
    # Test tokenization
    text = "यह movie बहुत अच्छी है but not perfect"
    tokens = tokenizer.tokenize(text)
    
    assert len(tokens) > 0, "Should produce tokens"
    print(f"  Text: {text}")
    print(f"  Tokens: {tokens}")
    
    # Test language tagging
    tagged_tokens = tokenizer.tokenize_with_language_tags(text)
    
    assert len(tagged_tokens) > 0, "Should produce tagged tokens"
    print(f"  Tagged tokens: {tagged_tokens[:3]}...")  # Show first 3
    
    # Test code mixing index
    mixing_index = tokenizer.get_code_mixing_index(text)
    print(f"  Code-mixing index: {mixing_index:.3f}")
    
    print("  ✓ Tokenizer tests passed")


def test_prism_analyzer():
    """Test main PRISM analyzer."""
    print("Testing PRISMAnalyzer...")
    
    # Initialize analyzer
    analyzer = PRISMAnalyzer()
    
    # Check capabilities
    capabilities = analyzer.get_capabilities()
    print(f"  Capabilities: {capabilities}")
    
    # Test single text analysis
    test_text = "This movie is good यह फिल्म अच्छी है"
    results = analyzer.analyze(test_text)
    
    assert 'error' not in results or not results['error'], f"Analysis failed: {results.get('error')}"
    assert 'hate_speech' in results, "Should include hate speech analysis"
    assert 'mood' in results, "Should include mood analysis"
    assert 'sentiment' in results, "Should include sentiment analysis"
    
    print(f"  Analyzed: {test_text}")
    print(f"  Hate speech: {results['hate_speech']['is_hate_speech']}")
    print(f"  Mood: {results['mood']['primary_mood']}")
    print(f"  Sentiment: {results['sentiment']['polarity']}")
    
    # Test batch analysis
    sample_data = create_sample_data()
    batch_texts = sample_data['code_mixed'][:2]  # Test with 2 texts
    
    batch_results = analyzer.analyze_batch(batch_texts)
    
    assert len(batch_results) == len(batch_texts), "Should return results for all texts"
    
    print(f"  Batch analysis: {len(batch_results)} results")
    
    print("  ✓ PRISMAnalyzer tests passed")


def test_integration():
    """Test complete integration."""
    print("Testing Integration...")
    
    analyzer = PRISMAnalyzer()
    
    # Test with different types of content
    test_cases = [
        ("Positive English", "This is amazing and wonderful!"),
        ("Negative English", "This is terrible and awful"),
        ("Code-mixed positive", "यह movie बहुत अच्छी है, I loved it!"),
        ("Code-mixed negative", "यह film बकवास है, waste of time"),
        ("Neutral", "The weather is okay today")
    ]
    
    for test_name, test_text in test_cases:
        results = analyzer.analyze(test_text)
        
        if 'error' not in results:
            hate_speech = results['hate_speech']['is_hate_speech']
            mood = results['mood']['primary_mood']
            sentiment = results['sentiment']['polarity']
            
            print(f"  {test_name}:")
            print(f"    Hate: {hate_speech}, Mood: {mood}, Sentiment: {sentiment}")
        else:
            print(f"  {test_name}: Error - {results['error']}")
    
    print("  ✓ Integration tests completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PRISM Basic Tests")
    print("=" * 60)
    
    try:
        test_text_cleaner()
        print()
        
        test_tokenizer()
        print()
        
        test_prism_analyzer()
        print()
        
        test_integration()
        print()
        
        print("=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)