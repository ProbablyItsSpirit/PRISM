"""
Helper Utilities

This module provides various utility functions for PRISM.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import os
from datetime import datetime


class PRISMLogger:
    """Custom logger for PRISM operations."""
    
    def __init__(self, name: str = "PRISM", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


def save_results(results: Dict[str, Any], filepath: str) -> bool:
    """
    Save analysis results to file.
    
    Args:
        results (Dict[str, Any]): Results to save
        filepath (str): Output file path
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_to_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def load_results(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load analysis results from file.
    
    Args:
        filepath (str): Input file path
        
    Returns:
        Optional[Dict[str, Any]]: Loaded results or None if failed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def format_analysis_report(results: Dict[str, Any], text: str) -> str:
    """
    Format analysis results into a readable report.
    
    Args:
        results (Dict[str, Any]): Analysis results
        text (str): Original text analyzed
        
    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 60)
    report.append("PRISM ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Original Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    report.append("")
    
    # Hate Speech Analysis
    if 'hate_speech' in results:
        hs = results['hate_speech']
        report.append("HATE SPEECH ANALYSIS:")
        report.append(f"  Classification: {'HATE SPEECH' if hs.get('is_hate_speech') else 'NOT HATE SPEECH'}")
        report.append(f"  Confidence: {hs.get('confidence', 0):.2f}")
        report.append(f"  Severity: {hs.get('severity', 'unknown').upper()}")
        if hs.get('detected_keywords'):
            report.append(f"  Detected Keywords: {', '.join(hs['detected_keywords'])}")
        report.append("")
    
    # Mood Analysis
    if 'mood' in results:
        mood = results['mood']
        report.append("MOOD ANALYSIS:")
        report.append(f"  Primary Mood: {mood.get('primary_mood', 'unknown').upper()}")
        report.append(f"  Confidence: {mood.get('confidence', 0):.2f}")
        report.append(f"  Intensity: {mood.get('intensity', 'unknown').upper()}")
        report.append(f"  Emotional Polarity: {mood.get('emotional_polarity', 'unknown').upper()}")
        report.append("")
    
    # Sentiment Analysis
    if 'sentiment' in results:
        sent = results['sentiment']
        report.append("SENTIMENT ANALYSIS:")
        report.append(f"  Primary Sentiment: {sent.get('primary_sentiment', 'unknown').upper()}")
        report.append(f"  Polarity: {sent.get('polarity', 'unknown').upper()}")
        report.append(f"  Confidence: {sent.get('confidence', 0):.2f}")
        report.append(f"  Valence: {sent.get('valence', 0):.2f}")
        report.append(f"  Intensity: {sent.get('intensity', 'unknown').upper()}")
        report.append("")
    
    # Transition Analysis (if available)
    if 'transitions' in results:
        trans = results['transitions']
        report.append("TRANSITION ANALYSIS:")
        report.append(f"  Transitions Detected: {trans.get('transitions_detected', False)}")
        if trans.get('mood_transitions'):
            report.append(f"  Mood Transitions: {len(trans['mood_transitions'])}")
        if trans.get('sentiment_transitions'):
            report.append(f"  Sentiment Transitions: {len(trans['sentiment_transitions'])}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for scores.
    
    Args:
        scores (List[float]): List of scores
        confidence (float): Confidence level
        
    Returns:
        Dict[str, float]: Mean, lower bound, upper bound
    """
    if not scores:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    
    # Simple approximation using normal distribution
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    margin = z_score * (std / np.sqrt(len(scores)))
    
    return {
        'mean': float(mean),
        'lower': float(mean - margin),
        'upper': float(mean + margin),
        'std': float(std)
    }


def merge_analysis_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple analysis results into summary statistics.
    
    Args:
        results_list (List[Dict[str, Any]]): List of individual results
        
    Returns:
        Dict[str, Any]: Merged summary results
    """
    if not results_list:
        return {}
    
    merged = {
        'total_analyzed': len(results_list),
        'timestamp': datetime.now().isoformat()
    }
    
    # Aggregate hate speech results
    hate_speech_scores = []
    hate_speech_positives = 0
    
    for result in results_list:
        if 'hate_speech' in result:
            hs = result['hate_speech']
            hate_speech_scores.append(hs.get('score', 0))
            if hs.get('is_hate_speech'):
                hate_speech_positives += 1
    
    if hate_speech_scores:
        merged['hate_speech_summary'] = {
            'detection_rate': hate_speech_positives / len(results_list),
            'score_stats': calculate_confidence_interval(hate_speech_scores)
        }
    
    # Aggregate mood results
    mood_distribution = {}
    mood_confidences = []
    
    for result in results_list:
        if 'mood' in result:
            mood = result['mood']
            primary_mood = mood.get('primary_mood', 'unknown')
            mood_distribution[primary_mood] = mood_distribution.get(primary_mood, 0) + 1
            mood_confidences.append(mood.get('confidence', 0))
    
    if mood_distribution:
        merged['mood_summary'] = {
            'distribution': mood_distribution,
            'confidence_stats': calculate_confidence_interval(mood_confidences)
        }
    
    # Aggregate sentiment results
    sentiment_distribution = {}
    sentiment_valences = []
    
    for result in results_list:
        if 'sentiment' in result:
            sent = result['sentiment']
            polarity = sent.get('polarity', 'unknown')
            sentiment_distribution[polarity] = sentiment_distribution.get(polarity, 0) + 1
            sentiment_valences.append(sent.get('valence', 0))
    
    if sentiment_distribution:
        merged['sentiment_summary'] = {
            'distribution': sentiment_distribution,
            'valence_stats': calculate_confidence_interval(sentiment_valences)
        }
    
    return merged


def validate_text_input(text: Union[str, List[str]]) -> List[str]:
    """
    Validate and normalize text input.
    
    Args:
        text (Union[str, List[str]]): Text input to validate
        
    Returns:
        List[str]: List of valid text strings
    """
    if isinstance(text, str):
        if text.strip():
            return [text.strip()]
        else:
            return []
    elif isinstance(text, list):
        valid_texts = []
        for item in text:
            if isinstance(item, str) and item.strip():
                valid_texts.append(item.strip())
        return valid_texts
    else:
        return []


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dict[str, Any]: System information
    """
    import torch
    import platform
    
    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
    
    return info


def create_sample_data() -> Dict[str, List[str]]:
    """
    Create sample data for testing and demonstration.
    
    Returns:
        Dict[str, List[str]]: Sample texts categorized by type
    """
    return {
        'code_mixed': [
            "यह movie बहुत अच्छी है but the ending was disappointing",
            "Main kal office जाऊंगा and then shopping करूंगा",
            "This song is so beautiful, मुझे बहुत पसंद आया",
            "Weather आज बहुत अच्छा है, let's go for a walk"
        ],
        'hate_speech': [
            "I hate this stupid movie",
            "यह बकवास है, waste of time",
            "This is terrible and disgusting"
        ],
        'positive_sentiment': [
            "I love this amazing movie!",
            "यह फिल्म शानदार है, बहुत पसंद आई",
            "Fantastic performance by all actors"
        ],
        'emotional_transitions': [
            "I was so excited about the movie initially",
            "But then the story became confusing", 
            "By the end, I was completely disappointed",
            "However, the music saved the experience"
        ]
    }