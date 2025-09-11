"""
Text Cleaning Module

This module provides text preprocessing and cleaning utilities for code-mixed text.
It handles various languages and mixed scripts commonly found in Indian languages.
"""

import re
import string
from typing import List, Dict, Any, Optional
import unicodedata


class TextCleaner:
    """
    Text cleaning and preprocessing for code-mixed text.
    
    Handles multiple scripts (Latin, Devanagari, etc.) and normalizes
    text for better analysis while preserving linguistic characteristics.
    """
    
    def __init__(self, preserve_emojis: bool = True, preserve_mentions: bool = False):
        """
        Initialize the text cleaner.
        
        Args:
            preserve_emojis (bool): Whether to preserve emoji characters
            preserve_mentions (bool): Whether to preserve @mentions and #hashtags
        """
        self.preserve_emojis = preserve_emojis
        self.preserve_mentions = preserve_mentions
        
        # Common patterns for cleaning
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,15}')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
        # Emoji pattern (basic range)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        
        # Devanagari and other script patterns
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        
        # Common transliteration mappings
        self.transliteration_map = {
            'aur': 'और', 'hai': 'है', 'me': 'में', 'ka': 'का', 'ki': 'की',
            'ko': 'को', 'se': 'से', 'pe': 'पे', 'bhi': 'भी', 'nahi': 'नहीं'
        }
    
    def clean(self, text: str, level: str = "moderate") -> str:
        """
        Clean and preprocess text.
        
        Args:
            text (str): Input text to clean
            level (str): Cleaning level - "light", "moderate", "aggressive"
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Start with the original text
        cleaned = text
        
        if level in ["moderate", "aggressive"]:
            # Remove URLs
            cleaned = self.url_pattern.sub(' ', cleaned)
            
            # Remove email addresses
            cleaned = self.email_pattern.sub(' ', cleaned)
            
            # Remove phone numbers
            cleaned = self.phone_pattern.sub(' ', cleaned)
            
            # Handle mentions and hashtags
            if not self.preserve_mentions:
                cleaned = self.mention_pattern.sub(' ', cleaned)
                cleaned = self.hashtag_pattern.sub(' ', cleaned)
            
            # Handle emojis
            if not self.preserve_emojis:
                cleaned = self.emoji_pattern.sub(' ', cleaned)
        
        if level == "aggressive":
            # Remove extra punctuation (keep basic ones)
            cleaned = re.sub(r'[^\w\s\u0900-\u097F।.!?,-]', ' ', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Unicode normalization
        cleaned = unicodedata.normalize('NFKC', cleaned)
        
        return cleaned
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better processing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase (preserve Devanagari)
        normalized = text.lower()
        
        # Normalize repeated characters (e.g., "sooooo" -> "so")
        normalized = re.sub(r'(.)\1{2,}', r'\1\1', normalized)
        
        # Fix common spacing issues
        normalized = re.sub(r'\s+([.!?])', r'\1', normalized)
        normalized = re.sub(r'([.!?])\s*([.!?])+', r'\1', normalized)
        
        # Handle mixed script spacing
        normalized = re.sub(r'(\w)([।])', r'\1 \2', normalized)
        
        return normalized.strip()
    
    def detect_languages(self, text: str) -> Dict[str, float]:
        """
        Detect languages present in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Language ratios
        """
        if not text:
            return {}
        
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return {}
        
        # Count characters by script
        latin_chars = len(re.findall(r'[A-Za-z]', text))
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        numbers = len(re.findall(r'\d', text))
        other_chars = total_chars - latin_chars - devanagari_chars - numbers
        
        # Calculate ratios
        ratios = {
            'english': latin_chars / total_chars,
            'hindi': devanagari_chars / total_chars,
            'numeric': numbers / total_chars,
            'other': other_chars / total_chars
        }
        
        return ratios
    
    def is_code_mixed(self, text: str, threshold: float = 0.1) -> bool:
        """
        Check if text is code-mixed (contains multiple languages).
        
        Args:
            text (str): Input text
            threshold (float): Minimum ratio for considering a language present
            
        Returns:
            bool: True if text is code-mixed
        """
        ratios = self.detect_languages(text)
        
        # Count languages above threshold
        significant_languages = sum(1 for ratio in ratios.values() if ratio >= threshold)
        
        return significant_languages >= 2
    
    def split_by_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into segments by dominant language.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: List of text segments with language info
        """
        if not text:
            return []
        
        words = text.split()
        segments = []
        current_segment = []
        current_lang = None
        
        for word in words:
            # Detect dominant script in word
            word_ratios = self.detect_languages(word)
            if word_ratios:
                dominant_lang = max(word_ratios.items(), key=lambda x: x[1])[0]
                
                if current_lang is None:
                    current_lang = dominant_lang
                    current_segment.append(word)
                elif current_lang == dominant_lang:
                    current_segment.append(word)
                else:
                    # Language change detected
                    if current_segment:
                        segments.append({
                            'text': ' '.join(current_segment),
                            'language': current_lang,
                            'start_pos': len(' '.join(segments[:-1] if segments else [])),
                            'length': len(' '.join(current_segment))
                        })
                    
                    current_segment = [word]
                    current_lang = dominant_lang
        
        # Add final segment
        if current_segment:
            segments.append({
                'text': ' '.join(current_segment),
                'language': current_lang,
                'start_pos': len(' '.join([s['text'] for s in segments])),
                'length': len(' '.join(current_segment))
            })
        
        return segments
    
    def remove_noise(self, text: str) -> str:
        """
        Remove common noise patterns from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Denoised text
        """
        if not text:
            return ""
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove standalone numbers (unless they seem meaningful)
        text = re.sub(r'\b\d{1,2}\b(?!\d)', '', text)
        
        # Remove very short words (likely noise) but preserve important ones
        important_short = {'i', 'a', 'is', 'it', 'me', 'my', 'we', 'he', 'go', 'no', 'up', 'on', 'in', 'to'}
        words = text.split()
        filtered_words = []
        
        for word in words:
            if len(word) >= 3 or word.lower() in important_short or self.devanagari_pattern.search(word):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def standardize_text(self, text: str) -> str:
        """
        Standardize text format for consistent processing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Standardized text
        """
        if not text:
            return ""
        
        # Apply all cleaning steps in sequence
        text = self.clean(text, level="moderate")
        text = self.normalize_text(text)
        text = self.remove_noise(text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text