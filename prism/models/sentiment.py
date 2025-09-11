"""
Sentiment Analysis Module

This module implements sentiment analysis for code-mixed text using MuRIL embeddings.
It provides detailed sentiment classification beyond simple positive/negative.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
from ..models.muril_model import MuRILModel


class SentimentAnalyzer:
    """
    Sentiment analysis system for code-mixed text.
    
    Provides detailed sentiment analysis including polarity, intensity,
    and emotional dimensions using MuRIL embeddings.
    """
    
    # Define sentiment categories
    SENTIMENT_CATEGORIES = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    
    def __init__(self, muril_model: MuRILModel):
        """
        Initialize the sentiment analyzer.
        
        Args:
            muril_model (MuRILModel): Pre-loaded MuRIL model
        """
        self.muril_model = muril_model
        self.num_sentiments = len(self.SENTIMENT_CATEGORIES)
        
        # Sentiment classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # MuRIL has 768-dim embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_sentiments),
            nn.Softmax(dim=-1)
        )
        
        # Regression head for continuous sentiment score
        self.regression_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Move to same device as MuRIL
        self.classifier.to(self.muril_model.device)
        self.regression_head.to(self.muril_model.device)
        
        # Initialize sentiment indicators
        self._initialize_sentiment_indicators()
    
    def _initialize_sentiment_indicators(self):
        """Initialize sentiment indicator keywords for different languages."""
        self.sentiment_indicators = {
            'very_positive': {
                'english': ['excellent', 'amazing', 'fantastic', 'wonderful', 'outstanding', 'brilliant', 'perfect'],
                'hindi': ['शानदार', 'बेहतरीन', 'उत्कृष्ट', 'अद्भुत', 'परफेक्ट'],
                'mixed': ['shandar', 'behatreen', 'utkrisht', 'adbhut']
            },
            'positive': {
                'english': ['good', 'nice', 'great', 'happy', 'like', 'love', 'enjoy', 'pleased'],
                'hindi': ['अच्छा', 'अच्छी', 'बढ़िया', 'खुश', 'पसंद', 'प्रेम', 'मजा'],
                'mixed': ['accha', 'achhi', 'badhiya', 'khush', 'pasand', 'maza']
            },
            'neutral': {
                'english': ['okay', 'fine', 'normal', 'average', 'standard'],
                'hindi': ['ठीक', 'सामान्य', 'औसत'],
                'mixed': ['theek', 'samanya', 'ausat']
            },
            'negative': {
                'english': ['bad', 'poor', 'disappointing', 'sad', 'dislike', 'hate', 'annoying'],
                'hindi': ['बुरा', 'खराब', 'गलत', 'दुख', 'नापसंद', 'गुस्सा'],
                'mixed': ['bura', 'kharab', 'galat', 'dukh', 'napasand', 'gussa']
            },
            'very_negative': {
                'english': ['terrible', 'awful', 'horrible', 'disgusting', 'worst', 'hate', 'pathetic'],
                'hindi': ['भयानक', 'घटिया', 'बकवास', 'बेकार', 'सबसे खराब'],
                'mixed': ['bhayanak', 'ghatiya', 'bakwas', 'bekaar', 'sabse kharab']
            }
        }
        
        # Initialize weights
        with torch.no_grad():
            for module in [self.classifier, self.regression_head]:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment in the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        # Get MuRIL embeddings
        embeddings = self.muril_model.encode_text(text)
        
        # Neural classification
        with torch.no_grad():
            sentiment_probs = self.classifier(embeddings).squeeze().cpu().numpy()
            continuous_score = self.regression_head(embeddings).squeeze().cpu().item()
        
        # Keyword-based analysis
        keyword_scores = self._keyword_based_analysis(text)
        
        # Combine neural and keyword-based scores
        combined_probs = 0.7 * sentiment_probs + 0.3 * np.array(keyword_scores)
        combined_probs = combined_probs / np.sum(combined_probs)  # Normalize
        
        # Get primary sentiment
        primary_idx = np.argmax(combined_probs)
        primary_sentiment = self.SENTIMENT_CATEGORIES[primary_idx]
        confidence = combined_probs[primary_idx]
        
        # Convert to polarity
        polarity = self._get_polarity(primary_sentiment)
        
        # Calculate intensity
        intensity = self._calculate_intensity(combined_probs, continuous_score)
        
        # Get sentiment distribution
        sentiment_distribution = {
            sentiment: float(score) for sentiment, score in zip(self.SENTIMENT_CATEGORIES, combined_probs)
        }
        
        # Get detected indicators
        detected_indicators = self._get_detected_indicators(text)
        
        # Calculate emotional dimensions
        valence = self._calculate_valence(combined_probs)
        arousal = self._calculate_arousal(text, combined_probs)
        
        return {
            'primary_sentiment': primary_sentiment,
            'polarity': polarity,
            'confidence': float(confidence),
            'intensity': intensity,
            'continuous_score': continuous_score,
            'sentiment_distribution': sentiment_distribution,
            'detected_indicators': detected_indicators,
            'valence': valence,
            'arousal': arousal,
            'subjectivity': self._calculate_subjectivity(text)
        }
    
    def _keyword_based_analysis(self, text: str) -> List[float]:
        """
        Perform keyword-based sentiment analysis.
        
        Args:
            text (str): Input text
            
        Returns:
            List[float]: Sentiment scores for each category
        """
        text_lower = text.lower()
        scores = [0.0] * self.num_sentiments
        
        for sent_idx, sentiment in enumerate(self.SENTIMENT_CATEGORIES):
            if sentiment in self.sentiment_indicators:
                for lang_keywords in self.sentiment_indicators[sentiment].values():
                    for keyword in lang_keywords:
                        if keyword.lower() in text_lower:
                            scores[sent_idx] += 1.0
        
        # Normalize scores
        total_score = sum(scores)
        if total_score > 0:
            scores = [score / total_score for score in scores]
        else:
            # Default to neutral if no keywords found
            neutral_idx = self.SENTIMENT_CATEGORIES.index('neutral')
            scores[neutral_idx] = 1.0
        
        return scores
    
    def _get_detected_indicators(self, text: str) -> Dict[str, List[str]]:
        """Get detected sentiment indicators by category."""
        text_lower = text.lower()
        detected = {sentiment: [] for sentiment in self.SENTIMENT_CATEGORIES}
        
        for sentiment in self.SENTIMENT_CATEGORIES:
            if sentiment in self.sentiment_indicators:
                for lang_keywords in self.sentiment_indicators[sentiment].values():
                    for keyword in lang_keywords:
                        if keyword.lower() in text_lower:
                            detected[sentiment].append(keyword)
        
        # Remove empty categories
        detected = {sent: keywords for sent, keywords in detected.items() if keywords}
        
        return detected
    
    def _get_polarity(self, sentiment: str) -> str:
        """Convert sentiment category to simple polarity."""
        if sentiment in ['very_positive', 'positive']:
            return 'positive'
        elif sentiment in ['very_negative', 'negative']:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_intensity(self, probs: np.ndarray, continuous_score: float) -> str:
        """Calculate sentiment intensity."""
        # Use both categorical confidence and continuous score
        max_prob = np.max(probs)
        abs_continuous = abs(continuous_score)
        
        combined_intensity = (max_prob + abs_continuous) / 2
        
        if combined_intensity < 0.3:
            return "low"
        elif combined_intensity < 0.6:
            return "moderate"
        elif combined_intensity < 0.8:
            return "high"
        else:
            return "very_high"
    
    def _calculate_valence(self, probs: np.ndarray) -> float:
        """Calculate valence (pleasantness) score."""
        # Weight each sentiment category
        weights = [-2, -1, 0, 1, 2]  # very_negative to very_positive
        valence = np.sum(probs * weights) / 2  # Normalize to [-1, 1]
        return float(valence)
    
    def _calculate_arousal(self, text: str, probs: np.ndarray) -> float:
        """Calculate arousal (activation) level."""
        # High arousal words
        high_arousal_words = [
            'excited', 'angry', 'thrilled', 'furious', 'amazing', 'terrible',
            'उत्साहित', 'गुस्सा', 'शानदार', 'भयानक'
        ]
        
        text_lower = text.lower()
        arousal_count = sum(1 for word in high_arousal_words if word.lower() in text_lower)
        
        # Combine with extreme sentiment probabilities
        extreme_probs = probs[0] + probs[-1]  # very_negative + very_positive
        
        arousal = (arousal_count / max(len(text.split()), 1)) + extreme_probs
        return min(float(arousal), 1.0)
    
    def _calculate_subjectivity(self, text: str) -> float:
        """Calculate subjectivity score (objective vs subjective)."""
        # Subjective indicators
        subjective_words = [
            'think', 'feel', 'believe', 'opinion', 'personally', 'i think',
            'लगता', 'सोच', 'राय', 'विचार'
        ]
        
        text_lower = text.lower()
        subjective_count = sum(1 for word in subjective_words if word.lower() in text_lower)
        
        # Simple heuristic
        subjectivity = min(subjective_count / max(len(text.split()), 1) * 3, 1.0)
        return float(subjectivity)
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment in multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict[str, Any]]: List of sentiment analysis results
        """
        results = []
        
        # Get batch embeddings
        embeddings = self.muril_model.encode_batch(texts)
        
        # Process each text
        for i, text in enumerate(texts):
            # Neural analysis
            with torch.no_grad():
                sentiment_probs = self.classifier(embeddings[i:i+1]).squeeze().cpu().numpy()
                continuous_score = self.regression_head(embeddings[i:i+1]).squeeze().cpu().item()
            
            # Keyword analysis
            keyword_scores = self._keyword_based_analysis(text)
            
            # Combine scores
            combined_probs = 0.7 * sentiment_probs + 0.3 * np.array(keyword_scores)
            combined_probs = combined_probs / np.sum(combined_probs)
            
            # Get results
            primary_idx = np.argmax(combined_probs)
            primary_sentiment = self.SENTIMENT_CATEGORIES[primary_idx]
            confidence = combined_probs[primary_idx]
            
            sentiment_distribution = {
                sentiment: float(score) for sentiment, score in zip(self.SENTIMENT_CATEGORIES, combined_probs)
            }
            
            results.append({
                'primary_sentiment': primary_sentiment,
                'polarity': self._get_polarity(primary_sentiment),
                'confidence': float(confidence),
                'intensity': self._calculate_intensity(combined_probs, continuous_score),
                'continuous_score': continuous_score,
                'sentiment_distribution': sentiment_distribution,
                'detected_indicators': self._get_detected_indicators(text),
                'valence': self._calculate_valence(combined_probs),
                'arousal': self._calculate_arousal(text, combined_probs),
                'subjectivity': self._calculate_subjectivity(text)
            })
        
        return results