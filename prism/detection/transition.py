"""
Transition Detection Module

This module implements detection of mood and sentiment transitions in text sequences.
It can identify changes in emotional states over time or across different parts of text.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ..models.mood_analysis import MoodAnalyzer
from ..models.sentiment import SentimentAnalyzer


class TransitionDetector:
    """
    Detects transitions in mood and sentiment across text sequences.
    
    This class analyzes sequences of text (e.g., sentences, paragraphs, or temporal data)
    to identify significant changes in emotional states.
    """
    
    def __init__(self, mood_analyzer: MoodAnalyzer, sentiment_analyzer: SentimentAnalyzer,
                 transition_threshold: float = 0.3):
        """
        Initialize the transition detector.
        
        Args:
            mood_analyzer (MoodAnalyzer): Mood analysis component
            sentiment_analyzer (SentimentAnalyzer): Sentiment analysis component
            transition_threshold (float): Threshold for detecting significant transitions
        """
        self.mood_analyzer = mood_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        self.transition_threshold = transition_threshold
    
    def detect_transitions(self, texts: List[str], text_type: str = "sequence") -> Dict[str, Any]:
        """
        Detect mood and sentiment transitions in a sequence of texts.
        
        Args:
            texts (List[str]): Sequence of texts to analyze
            text_type (str): Type of text sequence ("sequence", "temporal", "conversation")
            
        Returns:
            Dict[str, Any]: Transition detection results
        """
        if len(texts) < 2:
            return {
                'transitions_detected': False,
                'reason': 'Need at least 2 texts for transition detection',
                'mood_transitions': [],
                'sentiment_transitions': [],
                'sequence_analysis': []
            }
        
        # Analyze each text individually
        mood_results = self.mood_analyzer.batch_analyze(texts)
        sentiment_results = self.sentiment_analyzer.batch_analyze(texts)
        
        # Detect mood transitions
        mood_transitions = self._detect_mood_transitions(mood_results)
        
        # Detect sentiment transitions
        sentiment_transitions = self._detect_sentiment_transitions(sentiment_results)
        
        # Create sequence analysis
        sequence_analysis = self._create_sequence_analysis(mood_results, sentiment_results)
        
        # Overall transition summary
        transitions_detected = len(mood_transitions) > 0 or len(sentiment_transitions) > 0
        
        # Calculate transition patterns
        transition_patterns = self._analyze_transition_patterns(mood_transitions, sentiment_transitions)
        
        # Calculate emotional volatility
        emotional_volatility = self._calculate_emotional_volatility(mood_results, sentiment_results)
        
        return {
            'transitions_detected': transitions_detected,
            'mood_transitions': mood_transitions,
            'sentiment_transitions': sentiment_transitions,
            'sequence_analysis': sequence_analysis,
            'transition_patterns': transition_patterns,
            'emotional_volatility': emotional_volatility,
            'summary': self._create_summary(mood_transitions, sentiment_transitions, len(texts))
        }
    
    def _detect_mood_transitions(self, mood_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect significant mood transitions."""
        transitions = []
        
        for i in range(1, len(mood_results)):
            prev_mood = mood_results[i-1]
            curr_mood = mood_results[i]
            
            # Check if mood category changed
            mood_changed = prev_mood['primary_mood'] != curr_mood['primary_mood']
            
            # Calculate confidence change
            confidence_change = abs(curr_mood['confidence'] - prev_mood['confidence'])
            
            # Calculate distribution distance
            dist_change = self._calculate_distribution_distance(
                prev_mood['mood_distribution'],
                curr_mood['mood_distribution']
            )
            
            # Detect significant transition
            if mood_changed and (confidence_change > self.transition_threshold or 
                               dist_change > self.transition_threshold):
                
                transition = {
                    'position': i,
                    'from_mood': prev_mood['primary_mood'],
                    'to_mood': curr_mood['primary_mood'],
                    'from_confidence': prev_mood['confidence'],
                    'to_confidence': curr_mood['confidence'],
                    'confidence_change': confidence_change,
                    'distribution_change': dist_change,
                    'transition_type': self._classify_mood_transition(
                        prev_mood['primary_mood'], curr_mood['primary_mood']
                    ),
                    'intensity': self._calculate_transition_intensity(confidence_change, dist_change)
                }
                transitions.append(transition)
        
        return transitions
    
    def _detect_sentiment_transitions(self, sentiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect significant sentiment transitions."""
        transitions = []
        
        for i in range(1, len(sentiment_results)):
            prev_sent = sentiment_results[i-1]
            curr_sent = sentiment_results[i]
            
            # Check if sentiment polarity changed
            polarity_changed = prev_sent['polarity'] != curr_sent['polarity']
            
            # Calculate valence change
            valence_change = abs(curr_sent['valence'] - prev_sent['valence'])
            
            # Calculate continuous score change
            score_change = abs(curr_sent['continuous_score'] - prev_sent['continuous_score'])
            
            # Detect significant transition
            if polarity_changed or valence_change > self.transition_threshold or \
               score_change > self.transition_threshold:
                
                transition = {
                    'position': i,
                    'from_sentiment': prev_sent['primary_sentiment'],
                    'to_sentiment': curr_sent['primary_sentiment'],
                    'from_polarity': prev_sent['polarity'],
                    'to_polarity': curr_sent['polarity'],
                    'valence_change': valence_change,
                    'score_change': score_change,
                    'from_valence': prev_sent['valence'],
                    'to_valence': curr_sent['valence'],
                    'transition_type': self._classify_sentiment_transition(
                        prev_sent['polarity'], curr_sent['polarity']
                    ),
                    'intensity': self._calculate_transition_intensity(valence_change, score_change)
                }
                transitions.append(transition)
        
        return transitions
    
    def _create_sequence_analysis(self, mood_results: List[Dict[str, Any]], 
                                sentiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create detailed sequence analysis."""
        sequence = []
        
        for i, (mood, sentiment) in enumerate(zip(mood_results, sentiment_results)):
            analysis = {
                'position': i,
                'mood': {
                    'primary': mood['primary_mood'],
                    'confidence': mood['confidence'],
                    'intensity': mood['intensity'],
                    'polarity': mood['emotional_polarity']
                },
                'sentiment': {
                    'primary': sentiment['primary_sentiment'],
                    'polarity': sentiment['polarity'],
                    'confidence': sentiment['confidence'],
                    'valence': sentiment['valence'],
                    'intensity': sentiment['intensity']
                },
                'emotional_state': self._combine_mood_sentiment(mood, sentiment)
            }
            sequence.append(analysis)
        
        return sequence
    
    def _calculate_distribution_distance(self, dist1: Dict[str, float], 
                                       dist2: Dict[str, float]) -> float:
        """Calculate distance between two probability distributions."""
        # Ensure same keys and order
        common_keys = set(dist1.keys()) & set(dist2.keys())
        if not common_keys:
            return 1.0  # Maximum distance
        
        p1 = np.array([dist1.get(key, 0) for key in sorted(common_keys)])
        p2 = np.array([dist2.get(key, 0) for key in sorted(common_keys)])
        
        # Normalize
        p1 = p1 / np.sum(p1) if np.sum(p1) > 0 else p1
        p2 = p2 / np.sum(p2) if np.sum(p2) > 0 else p2
        
        # Use simple Euclidean distance normalized by sqrt(2)
        return float(np.linalg.norm(p1 - p2) / np.sqrt(2))
    
    def _classify_mood_transition(self, from_mood: str, to_mood: str) -> str:
        """Classify the type of mood transition."""
        positive_moods = {'happy', 'excited', 'calm', 'confident'}
        negative_moods = {'sad', 'angry', 'anxious', 'frustrated'}
        
        from_positive = from_mood in positive_moods
        to_positive = to_mood in positive_moods
        from_negative = from_mood in negative_moods
        to_negative = to_mood in negative_moods
        
        if from_positive and to_negative:
            return "positive_to_negative"
        elif from_negative and to_positive:
            return "negative_to_positive"
        elif from_positive and to_positive:
            return "positive_to_positive"
        elif from_negative and to_negative:
            return "negative_to_negative"
        else:
            return "neutral_transition"
    
    def _classify_sentiment_transition(self, from_polarity: str, to_polarity: str) -> str:
        """Classify the type of sentiment transition."""
        if from_polarity == to_polarity:
            return f"stable_{from_polarity}"
        elif from_polarity == "positive" and to_polarity == "negative":
            return "positive_to_negative"
        elif from_polarity == "negative" and to_polarity == "positive":
            return "negative_to_positive"
        else:
            return f"{from_polarity}_to_{to_polarity}"
    
    def _calculate_transition_intensity(self, change1: float, change2: float) -> str:
        """Calculate transition intensity based on changes."""
        avg_change = (change1 + change2) / 2
        
        if avg_change < 0.3:
            return "low"
        elif avg_change < 0.6:
            return "moderate"
        elif avg_change < 0.8:
            return "high"
        else:
            return "very_high"
    
    def _combine_mood_sentiment(self, mood: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
        """Combine mood and sentiment into overall emotional state."""
        mood_polarity = mood['emotional_polarity']
        sentiment_polarity = sentiment['polarity']
        mood_name = mood['primary_mood']
        
        if mood_polarity == sentiment_polarity:
            if mood_polarity == "positive":
                return f"positive_{mood_name}"
            elif mood_polarity == "negative":
                return f"negative_{mood_name}"
            else:
                return "neutral"
        else:
            return f"mixed_{mood_name}"
    
    def _analyze_transition_patterns(self, mood_transitions: List[Dict], 
                                   sentiment_transitions: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in transitions."""
        total_transitions = len(mood_transitions) + len(sentiment_transitions)
        
        if total_transitions == 0:
            return {
                'pattern_type': 'stable',
                'volatility': 'low',
                'dominant_direction': 'none'
            }
        
        # Analyze mood transition patterns
        mood_pattern = self._get_transition_pattern(mood_transitions, 'transition_type')
        sentiment_pattern = self._get_transition_pattern(sentiment_transitions, 'transition_type')
        
        # Overall volatility
        volatility = "high" if total_transitions > 2 else "moderate" if total_transitions > 1 else "low"
        
        return {
            'mood_pattern': mood_pattern,
            'sentiment_pattern': sentiment_pattern,
            'total_transitions': total_transitions,
            'volatility': volatility,
            'transition_frequency': total_transitions
        }
    
    def _get_transition_pattern(self, transitions: List[Dict], key: str) -> Dict[str, Any]:
        """Get pattern analysis for transitions."""
        if not transitions:
            return {'dominant': 'none', 'distribution': {}}
        
        # Count transition types
        type_counts = {}
        for trans in transitions:
            trans_type = trans.get(key, 'unknown')
            type_counts[trans_type] = type_counts.get(trans_type, 0) + 1
        
        # Find dominant pattern
        dominant = max(type_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'dominant': dominant,
            'distribution': type_counts,
            'count': len(transitions)
        }
    
    def _calculate_emotional_volatility(self, mood_results: List[Dict], 
                                      sentiment_results: List[Dict]) -> Dict[str, float]:
        """Calculate overall emotional volatility metrics."""
        if len(mood_results) < 2:
            return {'mood_volatility': 0.0, 'sentiment_volatility': 0.0, 'overall_volatility': 0.0}
        
        # Mood volatility (variance in confidence scores)
        mood_confidences = [result['confidence'] for result in mood_results]
        mood_volatility = float(np.var(mood_confidences))
        
        # Sentiment volatility (variance in valence scores)
        sentiment_valences = [result['valence'] for result in sentiment_results]
        sentiment_volatility = float(np.var(sentiment_valences))
        
        # Overall volatility
        overall_volatility = (mood_volatility + sentiment_volatility) / 2
        
        return {
            'mood_volatility': mood_volatility,
            'sentiment_volatility': sentiment_volatility,
            'overall_volatility': overall_volatility
        }
    
    def _create_summary(self, mood_transitions: List[Dict], sentiment_transitions: List[Dict], 
                       total_texts: int) -> Dict[str, Any]:
        """Create a summary of transition analysis."""
        total_transitions = len(mood_transitions) + len(sentiment_transitions)
        
        return {
            'total_texts_analyzed': total_texts,
            'mood_transitions_detected': len(mood_transitions),
            'sentiment_transitions_detected': len(sentiment_transitions),
            'total_transitions': total_transitions,
            'transition_rate': total_transitions / max(total_texts - 1, 1),
            'most_volatile_aspect': 'mood' if len(mood_transitions) > len(sentiment_transitions) else 'sentiment',
            'stability': 'stable' if total_transitions == 0 else 'moderate' if total_transitions < 2 else 'volatile'
        }