# dspy_multimodal

Multimodal image processing experiments using DSPy framework with local Ollama models.

## Features

- **Image Captioning**: Generate text descriptions from images
- **People Counting**: Detect and count people in images
- **Dataset Creation**: Automated creation of DSPy Example datasets from image directories
- **Evaluation**: Custom metrics for evaluating model performance

## Requirements

- Python 3.13+
- Ollama (local LLM server)
- Gemma3 models (4b or 27b)

## Setup

```bash
# Install dependencies
uv sync

# Install Ollama and pull models
ollama pull gemma3:4b
ollama pull gemma3:27b
```

## Project Structure

```
dspy_multimodal/
├── simple_captioning.py      # Image captioning example
├── count.py                   # People counting with evaluation
├── create_dataset.py          # Dataset creation utilities
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
1. Test a single image
2. Load the dataset from `images/count/`
3. Split into train/dev/test sets (70%/15%/15%)
4. Evaluate on the development set using the `count_exact_match` metric
5. Display accuracy and detailed results

### Creating Custom Datasets

```python
from create_dataset import create_count_dataset, split_dataset

# Create dataset from images/count/
dataset = create_count_dataset()

# Split into train/dev/test
trainset, devset, testset = split_dataset(dataset)

print(f"Training: {len(trainset)}")
print(f"Dev: {len(devset)}")
print(f"Test: {len(testset)}")
```

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

## Tech Stack

- [DSPy](https://github.com/stanfordnlp/dspy) - LLM programming framework
- [Ollama](https://ollama.ai/) - Local LLM server
- Pillow - Image processing library
- uv - Python package manager

## References

- [DSPy Evaluation Data](https://dspy.ai/learn/evaluation/data/)
- [DSPy Evaluation Metrics](https://dspy.ai/learn/evaluation/metrics/)
