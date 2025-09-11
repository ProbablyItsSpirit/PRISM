# Contributing to PRISM

Thank you for your interest in contributing to PRISM! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ProbablyItsSpirit/PRISM.git
cd PRISM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the demo to verify setup:
```bash
python demo.py
```

## Project Structure

```
prism/
├── __init__.py              # Main package init
├── analyzer.py              # Main PRISM analyzer
├── models/                  # Core analysis models
│   ├── muril_model.py       # MuRIL transformer wrapper
│   ├── hate_speech.py       # Hate speech detection
│   ├── mood_analysis.py     # Mood analysis
│   └── sentiment.py         # Sentiment analysis
├── preprocessing/           # Text preprocessing
│   ├── text_cleaner.py      # Text cleaning utilities
│   └── tokenizer.py         # Code-mixed tokenization
├── detection/              # Transition detection
│   └── transition.py       # Mood/sentiment transitions
└── utils/                  # Utility functions
    └── helpers.py          # Helper functions
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings for all public functions and classes
- Keep functions focused and modular

## Testing

Run the basic tests:
```bash
python tests/test_basic.py
```

Run the demo:
```bash
python demo.py
```

## Adding New Features

1. Create a new branch for your feature
2. Implement your changes with appropriate tests
3. Update documentation as needed
4. Submit a pull request

## Code-Mixed Text Guidelines

When working with code-mixed text:
- Support multiple scripts (Latin, Devanagari, etc.)
- Handle transliteration variations
- Preserve linguistic characteristics
- Test with various language combinations

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Include examples for new features
- Update type hints

## Issues and Bug Reports

Please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.