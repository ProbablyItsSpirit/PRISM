# PRISM (Profanity Recognition & Intelligent Sentiment Monitoring)

PRISM is an advanced NLP system for analyzing code-mixed text using the MuRIL (Multilingual Representations for Indian Languages) transformer model. It provides comprehensive analysis including:

- **Hate Speech Detection**: Identifies offensive and harmful content in multilingual text
- **Mood Analysis**: Detects emotional states and moods
- **Sentiment Analysis**: Analyzes positive, negative, and neutral sentiments
- **Transition Detection**: Tracks changes in mood and sentiment over time/context

## Features

- Support for code-mixed text (multiple languages in single text)
- Pre-trained MuRIL transformer integration
- Real-time analysis capabilities
- Transition detection for temporal sentiment analysis
- Comprehensive preprocessing pipeline
- Easy-to-use API

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from prism import PRISMAnalyzer

# Initialize the analyzer
analyzer = PRISMAnalyzer()

# Analyze code-mixed text
text = "यह बहुत अच्छा है but I hate this part"
results = analyzer.analyze(text)

print(f"Hate Speech: {results['hate_speech']}")
print(f"Mood: {results['mood']}")
print(f"Sentiment: {results['sentiment']}")
```

## Project Structure

```
prism/
├── __init__.py
├── analyzer.py          # Main PRISM analyzer
├── models/
│   ├── __init__.py
│   ├── muril_model.py   # MuRIL model wrapper
│   ├── hate_speech.py   # Hate speech detection
│   ├── mood_analysis.py # Mood analysis
│   └── sentiment.py     # Sentiment analysis
├── preprocessing/
│   ├── __init__.py
│   ├── text_cleaner.py  # Text preprocessing
│   └── tokenizer.py     # Tokenization utilities
├── detection/
│   ├── __init__.py
│   └── transition.py    # Transition detection
└── utils/
    ├── __init__.py
    └── helpers.py       # Utility functions
```

## License

MIT License - see LICENSE file for details.
