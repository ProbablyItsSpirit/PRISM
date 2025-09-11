"""
Mood Analysis Module

This module implements mood detection and analysis for code-mixed text using MuRIL embeddings.
It can detect various emotional states and moods from text.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
from ..models.muril_model import MuRILModel


class MoodAnalyzer:
    """
    Mood analysis system for code-mixed text.
    
    Detects various emotional states and moods using MuRIL embeddings
    and multi-class classification.
    """
    
    # Define mood categories
    MOOD_CATEGORIES = [
        'happy', 'sad', 'angry', 'excited', 'calm', 
        'anxious', 'confused', 'confident', 'frustrated', 'neutral'
    ]
    
    def __init__(self, muril_model: MuRILModel):
        """
        Initialize the mood analyzer.
        
        Args:
            muril_model (MuRILModel): Pre-loaded MuRIL model
        """
        self.muril_model = muril_model
        self.num_moods = len(self.MOOD_CATEGORIES)
        
        # Multi-class classifier for mood detection
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # MuRIL has 768-dim embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_moods),
            nn.Softmax(dim=-1)
        )
        
        # Move to same device as MuRIL
        self.classifier.to(self.muril_model.device)
        
        # Initialize mood indicators
        self._initialize_mood_indicators()
    
    def _initialize_mood_indicators(self):
        """Initialize mood indicator keywords for different languages."""
        self.mood_indicators = {
            'happy': {
                'english': ['happy', 'joy', 'excited', 'wonderful', 'great', 'amazing', 'love', 'smile'],
                'hindi': ['खुश', 'प्रसन्न', 'आनंद', 'अच्छा', 'बढ़िया', 'मजा'],
                'mixed': ['khush', 'accha', 'badhiya', 'maza', 'prasann']
            },
            'sad': {
                'english': ['sad', 'cry', 'depression', 'sorrow', 'upset', 'disappointed'],
                'hindi': ['उदास', 'दुख', 'रो', 'निराश', 'परेशान'],
                'mixed': ['udas', 'dukh', 'pareshaan', 'nirash']
            },
            'angry': {
                'english': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed'],
                'hindi': ['गुस्सा', 'क्रोध', 'चिढ़', 'नाराज'],
                'mixed': ['gussa', 'krodh', 'naraz', 'chidh']
            },
            'excited': {
                'english': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped'],
                'hindi': ['उत्साहित', 'रोमांचित', 'जोश'],
                'mixed': ['utsahit', 'josh', 'romanchit']
            },
            'calm': {
                'english': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil'],
                'hindi': ['शांत', 'आराम', 'सुकून'],
                'mixed': ['shant', 'aaram', 'sukoon']
            },
            'anxious': {
                'english': ['anxious', 'worried', 'nervous', 'stressed', 'tense'],
                'hindi': ['चिंतित', 'परेशान', 'तनाव'],
                'mixed': ['chintit', 'pareshaan', 'tanaav']
            }
        }
        
        # Initialize classifier weights
        with torch.no_grad():
            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze mood in the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, Any]: Mood analysis results
        """
        # Get MuRIL embeddings
        embeddings = self.muril_model.encode_text(text)
        
        # Neural classification
        with torch.no_grad():
            mood_probs = self.classifier(embeddings).squeeze().cpu().numpy()
        
        # Keyword-based analysis
        keyword_scores = self._keyword_based_analysis(text)
        
        # Combine neural and keyword-based scores
        combined_scores = 0.7 * mood_probs + 0.3 * np.array(keyword_scores)
        combined_scores = combined_scores / np.sum(combined_scores)  # Normalize
        
        # Get top moods
        top_mood_idx = np.argmax(combined_scores)
        top_mood = self.MOOD_CATEGORIES[top_mood_idx]
        confidence = combined_scores[top_mood_idx]
        
        # Get mood distribution
        mood_distribution = {
            mood: float(score) for mood, score in zip(self.MOOD_CATEGORIES, combined_scores)
        }
        
        # Determine intensity
        intensity = self._get_mood_intensity(confidence)
        
        # Get detected indicators
        detected_indicators = self._get_detected_indicators(text)
        
        return {
            'primary_mood': top_mood,
            'confidence': float(confidence),
            'intensity': intensity,
            'mood_distribution': mood_distribution,
            'detected_indicators': detected_indicators,
            'emotional_polarity': self._get_emotional_polarity(top_mood),
            'arousal_level': self._get_arousal_level(top_mood)
        }
    
    def _keyword_based_analysis(self, text: str) -> List[float]:
        """
        Perform keyword-based mood analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Mood scores for each category
        """
        text_lower = text.lower()
        scores = [0.0] * self.num_moods
        
        for mood_idx, mood in enumerate(self.MOOD_CATEGORIES):
            if mood in self.mood_indicators:
                for lang_keywords in self.mood_indicators[mood].values():
                    for keyword in lang_keywords:
                        if keyword.lower() in text_lower:
                            scores[mood_idx] += 1.0
        
        # Normalize scores
        total_score = sum(scores)
        if total_score > 0:
            scores = [score / total_score for score in scores]
        else:
            # Default to neutral if no keywords found
            neutral_idx = self.MOOD_CATEGORIES.index('neutral')
            scores[neutral_idx] = 1.0
        
        return scores
    
    def _get_detected_indicators(self, text: str) -> Dict[str, List[str]]:
        """Get detected mood indicators by category."""
        text_lower = text.lower()
        detected = {mood: [] for mood in self.MOOD_CATEGORIES}
        
        for mood in self.MOOD_CATEGORIES:
            if mood in self.mood_indicators:
                for lang_keywords in self.mood_indicators[mood].values():
                    for keyword in lang_keywords:
                        if keyword.lower() in text_lower:
                            detected[mood].append(keyword)
        
        # Remove empty categories
        detected = {mood: keywords for mood, keywords in detected.items() if keywords}
        
        return detected
    
    def _get_mood_intensity(self, confidence: float) -> str:
        """
        Get mood intensity level based on confidence.
        
        Args:
            confidence (float): Confidence score
            
        Returns:
            str: Intensity level
        """
        if confidence < 0.3:
            return "low"
        elif confidence < 0.6:
            return "moderate"
        elif confidence < 0.8:
            return "high"
        else:
            return "very_high"
    
    def _get_emotional_polarity(self, mood: str) -> str:
        """
        Get emotional polarity (positive/negative/neutral) for a mood.
        
        Args:
            mood (str): Detected mood
            
        Returns:
            str: Emotional polarity
        """
        positive_moods = ['happy', 'excited', 'calm', 'confident']
        negative_moods = ['sad', 'angry', 'anxious', 'frustrated']
        
        if mood in positive_moods:
            return "positive"
        elif mood in negative_moods:
            return "negative"
        else:
            return "neutral"
    
    def _get_arousal_level(self, mood: str) -> str:
        """
        Get arousal level (high/low) for a mood.
        
        Args:
            mood (str): Detected mood
            
        Returns:
            str: Arousal level
        """
        high_arousal = ['excited', 'angry', 'anxious', 'frustrated']
        low_arousal = ['calm', 'sad', 'neutral']
        
        if mood in high_arousal:
            return "high"
        elif mood in low_arousal:
            return "low"
        else:
            return "medium"
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze mood in multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict[str, Any]]: List of mood analysis results
        """
        results = []
        
        # Get batch embeddings
        embeddings = self.muril_model.encode_batch(texts)
        
        # Process each text
        for i, text in enumerate(texts):
            # Neural classification
            with torch.no_grad():
                mood_probs = self.classifier(embeddings[i:i+1]).squeeze().cpu().numpy()
            
            # Keyword-based analysis
            keyword_scores = self._keyword_based_analysis(text)
            
            # Combine scores
            combined_scores = 0.7 * mood_probs + 0.3 * np.array(keyword_scores)
            combined_scores = combined_scores / np.sum(combined_scores)
            
            # Get results
            top_mood_idx = np.argmax(combined_scores)
            top_mood = self.MOOD_CATEGORIES[top_mood_idx]
            confidence = combined_scores[top_mood_idx]
            
            mood_distribution = {
                mood: float(score) for mood, score in zip(self.MOOD_CATEGORIES, combined_scores)
            }
            
            results.append({
                'primary_mood': top_mood,
                'confidence': float(confidence),
                'intensity': self._get_mood_intensity(confidence),
                'mood_distribution': mood_distribution,
                'detected_indicators': self._get_detected_indicators(text),
                'emotional_polarity': self._get_emotional_polarity(top_mood),
                'arousal_level': self._get_arousal_level(top_mood)
            })
        
        return results