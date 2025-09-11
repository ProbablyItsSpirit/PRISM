"""
Tokenization Utilities

This module provides tokenization utilities for code-mixed text,
working with the MuRIL tokenizer and providing fallback options.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from .text_cleaner import TextCleaner


class CodeMixedTokenizer:
    """
    Tokenizer for code-mixed text with multiple language support.
    
    Provides utilities for tokenizing code-mixed text while preserving
    language boundaries and handling mixed scripts effectively.
    """
    
    def __init__(self, text_cleaner: Optional[TextCleaner] = None):
        """
        Initialize the tokenizer.
        
        Args:
            text_cleaner (Optional[TextCleaner]): Text cleaner instance
        """
        self.text_cleaner = text_cleaner or TextCleaner()
        
        # Define word boundaries for different scripts
        self.latin_word_pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
        self.devanagari_word_pattern = re.compile(r"[\u0900-\u097F]+")
        self.number_pattern = re.compile(r"\d+")
        self.punctuation_pattern = re.compile(r"[.,!?;:।\-\"'()]")
        
        # Common code-mixed patterns
        self.common_patterns = {
            'contractions': re.compile(r"\b\w+'\w+\b"),  # don't, won't, etc.
            'hashtags': re.compile(r"#\w+"),
            'mentions': re.compile(r"@\w+"),
            'urls': re.compile(r"http[s]?://[^\s]+"),
            'emails': re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        }
    
    def tokenize(self, text: str, preserve_special: bool = True) -> List[str]:
        """
        Tokenize code-mixed text into words and tokens.
        
        Args:
            text (str): Input text to tokenize
            preserve_special (bool): Whether to preserve special tokens
            
        Returns:
            List[str]: List of tokens
        """
        if not text:
            return []
        
        # Clean text if needed
        cleaned_text = self.text_cleaner.standardize_text(text)
        
        tokens = []
        remaining_text = cleaned_text
        
        if preserve_special:
            # Extract special patterns first
            special_tokens = self._extract_special_tokens(remaining_text)
            tokens.extend(special_tokens['tokens'])
            remaining_text = special_tokens['remaining_text']
        
        # Tokenize the remaining text
        word_tokens = self._tokenize_words(remaining_text)
        tokens.extend(word_tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def _extract_special_tokens(self, text: str) -> Dict[str, Any]:
        """Extract special tokens like URLs, mentions, etc."""
        special_tokens = []
        remaining = text
        
        for pattern_name, pattern in self.common_patterns.items():
            matches = pattern.findall(remaining)
            special_tokens.extend(matches)
            remaining = pattern.sub(' [SPECIAL_TOKEN] ', remaining)
        
        return {
            'tokens': special_tokens,
            'remaining_text': remaining
        }
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize regular words from text."""
        tokens = []
        
        # Split by whitespace first
        chunks = text.split()
        
        for chunk in chunks:
            if chunk == '[SPECIAL_TOKEN]':
                continue
                
            # Tokenize each chunk
            chunk_tokens = self._tokenize_chunk(chunk)
            tokens.extend(chunk_tokens)
        
        return tokens
    
    def _tokenize_chunk(self, chunk: str) -> List[str]:
        """Tokenize a single chunk of text."""
        tokens = []
        remaining = chunk
        
        while remaining:
            # Try to match different patterns
            latin_match = self.latin_word_pattern.match(remaining)
            devanagari_match = self.devanagari_word_pattern.match(remaining)
            number_match = self.number_pattern.match(remaining)
            punct_match = self.punctuation_pattern.match(remaining)
            
            if latin_match:
                tokens.append(latin_match.group())
                remaining = remaining[latin_match.end():]
            elif devanagari_match:
                tokens.append(devanagari_match.group())
                remaining = remaining[devanagari_match.end():]
            elif number_match:
                tokens.append(number_match.group())
                remaining = remaining[number_match.end():]
            elif punct_match:
                tokens.append(punct_match.group())
                remaining = remaining[punct_match.end():]
            else:
                # Skip unknown characters
                remaining = remaining[1:]
        
        return tokens
    
    def tokenize_with_language_tags(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize text and tag each token with its language.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, str]]: List of tokens with language tags
        """
        tokens = self.tokenize(text)
        tagged_tokens = []
        
        for token in tokens:
            language = self._detect_token_language(token)
            tagged_tokens.append({
                'token': token,
                'language': language,
                'type': self._get_token_type(token)
            })
        
        return tagged_tokens
    
    def _detect_token_language(self, token: str) -> str:
        """Detect the language of a single token."""
        if not token:
            return 'unknown'
        
        # Check script patterns
        if self.latin_word_pattern.fullmatch(token):
            return 'english'
        elif self.devanagari_word_pattern.fullmatch(token):
            return 'hindi'
        elif self.number_pattern.fullmatch(token):
            return 'numeric'
        elif self.punctuation_pattern.fullmatch(token):
            return 'punctuation'
        else:
            return 'mixed'
    
    def _get_token_type(self, token: str) -> str:
        """Get the type of token (word, punctuation, number, etc.)."""
        if self.punctuation_pattern.fullmatch(token):
            return 'punctuation'
        elif self.number_pattern.fullmatch(token):
            return 'number'
        elif any(pattern.fullmatch(token) for pattern in self.common_patterns.values()):
            return 'special'
        else:
            return 'word'
    
    def get_language_segments(self, text: str) -> List[Dict[str, Any]]:
        """
        Get text segments by language with position information.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: Language segments with positions
        """
        tokens_with_tags = self.tokenize_with_language_tags(text)
        segments = []
        
        if not tokens_with_tags:
            return segments
        
        current_lang = tokens_with_tags[0]['language']
        current_segment = [tokens_with_tags[0]['token']]
        start_pos = 0
        
        for i, token_info in enumerate(tokens_with_tags[1:], 1):
            token = token_info['token']
            lang = token_info['language']
            
            if lang == current_lang or lang in ['punctuation', 'numeric']:
                current_segment.append(token)
            else:
                # Language change detected
                segment_text = ' '.join(current_segment)
                segments.append({
                    'text': segment_text,
                    'language': current_lang,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(segment_text),
                    'token_count': len(current_segment)
                })
                
                start_pos += len(segment_text) + 1  # +1 for space
                current_segment = [token]
                current_lang = lang
        
        # Add final segment
        if current_segment:
            segment_text = ' '.join(current_segment)
            segments.append({
                'text': segment_text,
                'language': current_lang,
                'start_pos': start_pos,
                'end_pos': start_pos + len(segment_text),
                'token_count': len(current_segment)
            })
        
        return segments
    
    def count_tokens_by_language(self, text: str) -> Dict[str, int]:
        """
        Count tokens by language.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, int]: Token counts by language
        """
        tokens_with_tags = self.tokenize_with_language_tags(text)
        counts = {}
        
        for token_info in tokens_with_tags:
            lang = token_info['language']
            counts[lang] = counts.get(lang, 0) + 1
        
        return counts
    
    def get_code_mixing_index(self, text: str) -> float:
        """
        Calculate code-mixing index for the text.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Code-mixing index (0 = monolingual, 1 = highly mixed)
        """
        counts = self.count_tokens_by_language(text)
        
        # Remove punctuation and numeric from calculation
        word_counts = {lang: count for lang, count in counts.items() 
                      if lang not in ['punctuation', 'numeric', 'unknown']}
        
        if not word_counts:
            return 0.0
        
        total_words = sum(word_counts.values())
        if total_words <= 1:
            return 0.0
        
        # Calculate entropy-based mixing index
        entropy = 0.0
        for count in word_counts.values():
            if count > 0:
                p = count / total_words
                entropy -= p * (p.bit_length() - 1)  # Simple entropy approximation
        
        # Normalize by maximum possible entropy
        max_entropy = (len(word_counts).bit_length() - 1) if len(word_counts) > 1 else 1
        
        return min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
    
    def preprocess_for_model(self, text: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Preprocess text for model input.
        
        Args:
            text (str): Input text
            max_length (Optional[int]): Maximum sequence length
            
        Returns:
            Dict[str, Any]: Preprocessed text with metadata
        """
        # Clean and tokenize
        cleaned_text = self.text_cleaner.standardize_text(text)
        tokens = self.tokenize(cleaned_text)
        
        # Truncate if needed
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            truncated = True
        else:
            truncated = False
        
        # Rejoin tokens
        processed_text = ' '.join(tokens)
        
        # Get metadata
        language_counts = self.count_tokens_by_language(cleaned_text)
        code_mixing_index = self.get_code_mixing_index(cleaned_text)
        
        return {
            'processed_text': processed_text,
            'original_text': text,
            'tokens': tokens,
            'token_count': len(tokens),
            'truncated': truncated,
            'language_distribution': language_counts,
            'code_mixing_index': code_mixing_index,
            'is_code_mixed': code_mixing_index > 0.1
        }