"""
PRISM: Profanity Recognition & Intelligent Sentiment Monitoring

A comprehensive NLP system for analyzing code-mixed text using MuRIL transformer.
Provides hate speech detection, mood analysis, sentiment analysis, and transition detection.
"""

from .analyzer import PRISMAnalyzer

__version__ = "1.0.0"
__author__ = "PRISM Team"
__email__ = "contact@prism-nlp.com"

__all__ = ["PRISMAnalyzer"]