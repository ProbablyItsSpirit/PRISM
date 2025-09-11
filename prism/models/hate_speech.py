"""
Hate Speech Detection Module

This module implements hate speech detection for code-mixed text using MuRIL embeddings
and a classification layer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
from ..models.muril_model import MuRILModel


class HateSpeechDetector:
    """
    Hate speech detection system for code-mixed text.
    
    Uses MuRIL embeddings combined with a classification layer to detect
    offensive and harmful content in multilingual text.
    """
    
    def __init__(self, muril_model: MuRILModel, threshold: float = 0.5):
        """
        Initialize the hate speech detector.
        
        Args:
            muril_model (MuRILModel): Pre-loaded MuRIL model
            threshold (float): Classification threshold
        """
        self.muril_model = muril_model
        self.threshold = threshold
        
        # Simple classifier on top of MuRIL embeddings
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # MuRIL has 768-dim embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Move to same device as MuRIL
        self.classifier.to(self.muril_model.device)
        
        # Initialize with some basic patterns for demonstration
        self._initialize_basic_patterns()
    
    def _initialize_basic_patterns(self):
        """Initialize with basic hate speech patterns for demonstration."""
        # This is a simplified approach for demonstration
        # In practice, this would be trained on labeled data
        self.hate_keywords = {
            'english': ['hate', 'stupid', 'idiot', 'kill', 'die', 'worst', 'terrible'],
            'hindi': ['गंदा', 'बुरा', 'घटिया', 'बकवास'],  # basic negative words
            'mixed': ['bakwas', 'ganda', 'bura']  # transliterated
        }
        
        # Initialize classifier weights randomly for demonstration
        # In practice, these would be trained on labeled hate speech data
        with torch.no_grad():
            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect hate speech in the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, Any]: Detection results including score and classification
        """
        # Get MuRIL embeddings
        embeddings = self.muril_model.encode_text(text)
        
        # Basic keyword-based detection (for demonstration)
        keyword_score = self._keyword_based_detection(text)
        
        # Neural classification (simplified for demonstration)
        with torch.no_grad():
            neural_score = self.classifier(embeddings).item()
        
        # Combine scores
        combined_score = 0.6 * neural_score + 0.4 * keyword_score
        
        # Classification
        is_hate_speech = combined_score > self.threshold
        
        # Confidence level
        confidence = abs(combined_score - 0.5) * 2  # Scale to 0-1
        
        return {
            'is_hate_speech': is_hate_speech,
            'confidence': confidence,
            'score': combined_score,
            'neural_score': neural_score,
            'keyword_score': keyword_score,
            'detected_keywords': self._get_detected_keywords(text),
            'severity': self._get_severity_level(combined_score)
        }
    
    def _keyword_based_detection(self, text: str) -> float:
        """
        Simple keyword-based hate speech detection.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Keyword-based hate speech score (0-1)
        """
        text_lower = text.lower()
        hate_count = 0
        total_words = len(text.split())
        
        # Check for hate keywords in different languages
        for lang_keywords in self.hate_keywords.values():
            for keyword in lang_keywords:
                if keyword.lower() in text_lower:
                    hate_count += 1
        
        # Normalize by text length
        if total_words == 0:
            return 0.0
        
        score = min(hate_count / total_words * 5, 1.0)  # Scale factor
        return score
    
    def _get_detected_keywords(self, text: str) -> List[str]:
        """Get list of detected hate speech keywords."""
        text_lower = text.lower()
        detected = []
        
        for lang_keywords in self.hate_keywords.values():
            for keyword in lang_keywords:
                if keyword.lower() in text_lower:
                    detected.append(keyword)
        
        return detected
    
    def _get_severity_level(self, score: float) -> str:
        """
        Get severity level based on hate speech score.
        
        Args:
            score (float): Hate speech score
            
        Returns:
            str: Severity level
        """
        if score < 0.3:
            return "low"
        elif score < 0.6:
            return "medium"
        elif score < 0.8:
            return "high"
        else:
            return "severe"
    
    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect hate speech in multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict[str, Any]]: List of detection results
        """
        results = []
        
        # Get batch embeddings
        embeddings = self.muril_model.encode_batch(texts)
        
        # Process each text
        for i, text in enumerate(texts):
            # Keyword detection
            keyword_score = self._keyword_based_detection(text)
            
            # Neural classification
            with torch.no_grad():
                neural_score = self.classifier(embeddings[i:i+1]).item()
            
            # Combine scores
            combined_score = 0.6 * neural_score + 0.4 * keyword_score
            
            # Classification
            is_hate_speech = combined_score > self.threshold
            confidence = abs(combined_score - 0.5) * 2
            
            results.append({
                'is_hate_speech': is_hate_speech,
                'confidence': confidence,
                'score': combined_score,
                'neural_score': neural_score,
                'keyword_score': keyword_score,
                'detected_keywords': self._get_detected_keywords(text),
                'severity': self._get_severity_level(combined_score)
            })
        
        return results