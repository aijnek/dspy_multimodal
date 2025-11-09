# dspy_multimodal

Multimodal image processing experiments using DSPy framework with local Ollama models.

## Features

- **Image Captioning**: Generate text descriptions from images
- **People Counting**: Detect and count people in images using Chain-of-Thought reasoning
- **Dataset Creation**: Automated creation of DSPy Example datasets from image directories
- **Evaluation**: Custom metrics for evaluating model performance
- **Optimization**: GEPA (Generative Prompt Optimization with Adaptive feedback) for improving model accuracy

## Requirements

- Python 3.13+
- Ollama (local LLM server)
- Gemma3 27b model
- Anthropic API key (for optimization with Claude)

## Setup

```bash
# Install dependencies
uv sync

# Install Ollama and pull models
ollama pull gemma3:27b

# Set up environment variables
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Project Structure

```
dspy_multimodal/
├── simple_captioning.py      # Image captioning example
├── count.py                   # People counting with evaluation
├── create_dataset.py          # Dataset creation utilities
├── optimize.py                # GEPA optimization for improving accuracy
├── optimized.json             # Saved optimized program
└── images/
    └── count/
        ├── 0/                 # Images with 0 people
        ├── 1/                 # Images with 1 person
        ├── 2/                 # Images with 2 people
        ...
        └── 10/                # Images with 10 people
```

## Usage

### Image Captioning

```bash
uv run simple_captioning.py
```

### People Counting with Evaluation

```bash
uv run count.py
```

This will:
1. Load the dataset from `images/count/`
2. Split into train/dev/test sets (60%/20%/20%)
3. Evaluate on the test set using the `count_exact_match` metric
4. Display baseline accuracy
5. Load the optimized model and display improved accuracy

### Optimizing the Model

```bash
uv run optimize.py
```

This will:
1. Load the dataset and split it into train/val/test sets
2. Configure DSPy with Gemma3 27b (base model) and Claude Haiku (reflection model)
3. Run GEPA optimization with adaptive feedback
4. Save the optimized program to `optimized.json`

The optimization uses Chain-of-Thought reasoning and provides detailed feedback to improve accuracy.

### Creating Custom Datasets

```python
from create_dataset import create_count_dataset, split_dataset

# Create dataset from images/count/
dataset = create_count_dataset()

# Split into train/dev/test (60%/20%/20%)
trainset, devset, testset = split_dataset(dataset)

print(f"Training: {len(trainset)}")
print(f"Dev: {len(devset)}")
print(f"Test: {len(testset)}")
```

Note: Images are automatically resized to 512x512 thumbnails for efficient processing.

## Evaluation Metrics

### Custom Metric: `count_exact_match`

A simple exact match metric for people counting:

```python
def count_exact_match(example, pred, trace=None):
    """
    人物カウントの完全一致メトリクス

    Args:
        example: 正解ラベルを含むExampleオブジェクト
        pred: プログラムの予測結果
        trace: 中間ステップ（オプショナル）

    Returns:
        bool: 一致した場合True、不一致の場合False
    """
    return example.number_of_people == pred.number_of_people
```

This metric follows DSPy's evaluation patterns:
- Accepts `example`, `pred`, and optional `trace` parameters
- Returns boolean for exact match evaluation
- Can be used with DSPy's `Evaluate` utility for parallel evaluation

### Optimization Metric: `count_exact_match_with_feedback`

An enhanced metric that provides feedback for GEPA optimization:

```python
def count_exact_match_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Returns:
        dspy.Prediction with:
        - score: correctness (True/False)
        - feedback: detailed feedback text for model improvement
    """
    correctness = (example.number_of_people == pred.number_of_people)
    feedback_text = f"""Correct answer is {example.number_of_people}.
    Your answer is {pred.number_of_people}. Your reasoning is: {pred.reasoning}.
    """
    # Provides adaptive feedback based on correctness and edge cases
    return dspy.Prediction(score=correctness, feedback=feedback_text)
```

This metric:
- Provides detailed feedback including the correct answer and reasoning
- Handles edge cases (e.g., images with 10+ people)
- Returns a `dspy.Prediction` object with score and feedback for GEPA optimization

## Tech Stack

- [DSPy](https://github.com/stanfordnlp/dspy) - LLM programming framework with optimization capabilities
- [Ollama](https://ollama.ai/) - Local LLM server (Gemma3 27b)
- [Anthropic Claude](https://www.anthropic.com/) - Reflection model for GEPA optimization
- Pillow - Image processing library
- uv - Python package manager
- python-dotenv - Environment variable management

## Key Implementation Details

- **Model**: Uses `dspy.ChainOfThought` instead of `dspy.Predict` for better reasoning
- **Image Processing**: Images are resized to 512x512 for efficient processing
- **Optimization**: GEPA (Generative Prompt Optimization with Adaptive feedback) improves accuracy through iterative refinement
- **Dataset Split**: 60% training, 20% validation, 20% test

## References

- [DSPy Evaluation Data](https://dspy.ai/learn/evaluation/data/)
- [DSPy Evaluation Metrics](https://dspy.ai/learn/evaluation/metrics/)
- [DSPy Optimizers](https://dspy.ai/api/category/optimizers)
