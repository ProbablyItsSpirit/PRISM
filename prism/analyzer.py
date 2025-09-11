"""
PRISM Analyzer

Main analyzer class that integrates all components for comprehensive
analysis of code-mixed text including hate speech detection, mood analysis,
sentiment analysis, and transition detection.
"""

from typing import Dict, List, Any, Optional, Union
import torch

from .models import MuRILModel, HateSpeechDetector, MoodAnalyzer, SentimentAnalyzer
from .preprocessing import TextCleaner, CodeMixedTokenizer
from .detection import TransitionDetector
from .utils import PRISMLogger, validate_text_input, format_analysis_report


class PRISMAnalyzer:
    """
    Main PRISM analyzer for comprehensive code-mixed text analysis.
    
    Integrates MuRIL transformer with hate speech detection, mood analysis,
    sentiment analysis, and transition detection capabilities.
    """
    
    def __init__(self, 
                 model_name: str = "google/muril-base-cased",
                 device: Optional[str] = None,
                 logger_level: str = "INFO"):
        """
        Initialize the PRISM analyzer.
        
        Args:
            model_name (str): HuggingFace model name for MuRIL
            device (Optional[str]): Device to use ('cuda', 'cpu', or None for auto)
            logger_level (str): Logging level
        """
        self.logger = PRISMLogger("PRISM", logger_level)
        self.logger.info("Initializing PRISM analyzer...")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        try:
            # Core model
            self.muril_model = MuRILModel(model_name)
            self.logger.info(f"Loaded MuRIL model: {model_name}")
            
            # Analysis models
            self.hate_speech_detector = HateSpeechDetector(self.muril_model)
            self.mood_analyzer = MoodAnalyzer(self.muril_model)
            self.sentiment_analyzer = SentimentAnalyzer(self.muril_model)
            
            # Transition detector
            self.transition_detector = TransitionDetector(
                self.mood_analyzer, 
                self.sentiment_analyzer
            )
            
            # Preprocessing
            self.text_cleaner = TextCleaner()
            self.tokenizer = CodeMixedTokenizer(self.text_cleaner)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.warning("Some components may not work properly")
    
    def analyze(self, text: str, include_transitions: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a single text.
        
        Args:
            text (str): Input text to analyze
            include_transitions (bool): Whether to include transition analysis
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        if not text or not isinstance(text, str):
            return {'error': 'Invalid text input'}
        
        self.logger.info(f"Analyzing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        results = {
            'original_text': text,
            'analysis_timestamp': None,
            'preprocessing': None,
            'hate_speech': None,
            'mood': None,
            'sentiment': None,
            'transitions': None
        }
        
        try:
            # Preprocessing
            preprocessing_info = self.tokenizer.preprocess_for_model(text)
            results['preprocessing'] = preprocessing_info
            
            processed_text = preprocessing_info['processed_text']
            
            # Hate speech detection
            hate_speech_results = self.hate_speech_detector.detect(processed_text)
            results['hate_speech'] = hate_speech_results
            
            # Mood analysis
            mood_results = self.mood_analyzer.analyze(processed_text)
            results['mood'] = mood_results
            
            # Sentiment analysis
            sentiment_results = self.sentiment_analyzer.analyze(processed_text)
            results['sentiment'] = sentiment_results
            
            # Transition analysis (if requested and applicable)
            if include_transitions:
                # For single text, we'll analyze sentence-level transitions
                sentences = self._split_into_sentences(processed_text)
                if len(sentences) > 1:
                    transition_results = self.transition_detector.detect_transitions(sentences)
                    results['transitions'] = transition_results
                else:
                    results['transitions'] = {
                        'transitions_detected': False,
                        'reason': 'Single sentence - no transitions to detect'
                    }
            
            # Add analysis metadata
            import datetime
            results['analysis_timestamp'] = datetime.datetime.now().isoformat()
            results['model_info'] = {
                'muril_available': self.muril_model.is_available(),
                'device': str(self.device)
            }
            
            self.logger.info("Analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_batch(self, texts: List[str], include_transitions: bool = True) -> List[Dict[str, Any]]:
        """
        Perform analysis on multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            include_transitions (bool): Whether to include transition analysis
            
        Returns:
            List[Dict[str, Any]]: List of analysis results
        """
        valid_texts = validate_text_input(texts)
        
        if not valid_texts:
            return [{'error': 'No valid texts provided'}]
        
        self.logger.info(f"Analyzing batch of {len(valid_texts)} texts")
        
        results = []
        
        try:
            # Individual analysis for each text
            for i, text in enumerate(valid_texts):
                self.logger.debug(f"Processing text {i+1}/{len(valid_texts)}")
                result = self.analyze(text, include_transitions=False)
                results.append(result)
            
            # Transition analysis across the sequence
            if include_transitions and len(valid_texts) > 1:
                self.logger.info("Performing cross-sequence transition analysis")
                transition_results = self.transition_detector.detect_transitions(valid_texts)
                
                # Add transition results to the batch metadata
                batch_metadata = {
                    'batch_transition_analysis': transition_results,
                    'batch_size': len(valid_texts),
                    'analysis_timestamp': results[0].get('analysis_timestamp') if results else None
                }
                
                # Add batch metadata to first result
                if results:
                    results[0]['batch_metadata'] = batch_metadata
            
        except Exception as e:
            self.logger.error(f"Error during batch analysis: {e}")
            results.append({'error': str(e)})
        
        return results
    
    def analyze_conversation(self, messages: List[str], speaker_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a conversation with transition detection.
        
        Args:
            messages (List[str]): List of conversation messages
            speaker_ids (Optional[List[str]]): Optional speaker identifiers
            
        Returns:
            Dict[str, Any]: Conversation analysis results
        """
        valid_messages = validate_text_input(messages)
        
        if not valid_messages:
            return {'error': 'No valid messages provided'}
        
        self.logger.info(f"Analyzing conversation with {len(valid_messages)} messages")
        
        # Analyze each message
        message_analyses = []
        for i, message in enumerate(valid_messages):
            analysis = self.analyze(message, include_transitions=False)
            analysis['message_index'] = i
            if speaker_ids and i < len(speaker_ids):
                analysis['speaker_id'] = speaker_ids[i]
            message_analyses.append(analysis)
        
        # Conversation-level transition analysis
        transition_results = self.transition_detector.detect_transitions(
            valid_messages, text_type="conversation"
        )
        
        # Conversation summary
        conversation_summary = self._create_conversation_summary(message_analyses, transition_results)
        
        return {
            'conversation_type': 'multi_message',
            'message_count': len(valid_messages),
            'message_analyses': message_analyses,
            'transition_analysis': transition_results,
            'conversation_summary': conversation_summary,
            'analysis_timestamp': message_analyses[0].get('analysis_timestamp') if message_analyses else None
        }
    
    def get_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a formatted analysis report.
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results
            
        Returns:
            str: Formatted report
        """
        original_text = analysis_results.get('original_text', 'N/A')
        return format_analysis_report(analysis_results, original_text)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for transition analysis."""
        import re
        
        # Simple sentence splitting (can be improved)
        sentences = re.split(r'[.!?।]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_conversation_summary(self, message_analyses: List[Dict[str, Any]], 
                                   transition_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of conversation analysis."""
        if not message_analyses:
            return {}
        
        # Aggregate statistics
        hate_speech_count = sum(1 for msg in message_analyses 
                               if msg.get('hate_speech', {}).get('is_hate_speech', False))
        
        mood_distribution = {}
        sentiment_distribution = {}
        
        for msg in message_analyses:
            # Mood distribution
            mood = msg.get('mood', {}).get('primary_mood')
            if mood:
                mood_distribution[mood] = mood_distribution.get(mood, 0) + 1
            
            # Sentiment distribution
            sentiment = msg.get('sentiment', {}).get('polarity')
            if sentiment:
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
        
        return {
            'total_messages': len(message_analyses),
            'hate_speech_detected': hate_speech_count,
            'hate_speech_rate': hate_speech_count / len(message_analyses),
            'mood_distribution': mood_distribution,
            'sentiment_distribution': sentiment_distribution,
            'transitions_detected': transition_results.get('transitions_detected', False),
            'emotional_volatility': transition_results.get('emotional_volatility', {}),
            'conversation_stability': transition_results.get('summary', {}).get('stability', 'unknown')
        }
    
    def is_ready(self) -> bool:
        """Check if all components are ready for analysis."""
        return (self.muril_model is not None and 
                self.hate_speech_detector is not None and
                self.mood_analyzer is not None and
                self.sentiment_analyzer is not None)
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get information about available capabilities."""
        return {
            'muril_model_available': self.muril_model.is_available() if self.muril_model else False,
            'hate_speech_detection': self.hate_speech_detector is not None,
            'mood_analysis': self.mood_analyzer is not None,
            'sentiment_analysis': self.sentiment_analyzer is not None,
            'transition_detection': self.transition_detector is not None,
            'code_mixed_preprocessing': self.tokenizer is not None,
            'cuda_available': torch.cuda.is_available(),
            'device': str(self.device)
        }