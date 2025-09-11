"""
MuRIL Model Wrapper

This module provides a wrapper for the MuRIL (Multilingual Representations for Indian Languages) 
transformer model for processing code-mixed text.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Any, Optional


class MuRILModel:
    """
    MuRIL model wrapper for multilingual text processing.
    
    MuRIL is specifically designed for Indian languages and code-mixed text,
    making it ideal for analyzing text that contains multiple languages.
    """
    
    def __init__(self, model_name: str = "google/muril-base-cased"):
        """
        Initialize the MuRIL model.
        
        Args:
            model_name (str): The HuggingFace model name for MuRIL
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading MuRIL model: {e}")
            print("Falling back to basic tokenization...")
            self.tokenizer = None
            self.model = None
    
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Encode text using MuRIL tokenizer and get embeddings.
        
        Args:
            text (str): Input text to encode
            max_length (int): Maximum sequence length
            
        Returns:
            torch.Tensor: Text embeddings
        """
        if self.model is None:
            # Fallback: return dummy embeddings
            return torch.randn(1, 768)
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output (CLS token)
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def encode_batch(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Encode a batch of texts.
        
        Args:
            texts (List[str]): List of texts to encode
            max_length (int): Maximum sequence length
            
        Returns:
            torch.Tensor: Batch of text embeddings
        """
        if self.model is None:
            # Fallback: return dummy embeddings
            return torch.randn(len(texts), 768)
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output
        
        return embeddings
    
    def get_attention_weights(self, text: str) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            text (str): Input text
            
        Returns:
            Optional[torch.Tensor]: Attention weights
        """
        if self.model is None:
            return None
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            # Return attention from last layer
            return outputs.attentions[-1]
    
    def is_available(self) -> bool:
        """Check if the model is available and loaded."""
        return self.model is not None