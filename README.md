# MWK-FLAIR

**FLAIR with HarMA Adapter Integration** - A parameter-efficient fine-tuning framework for vision-language models.

## Overview

MWK-FLAIR combines the power of [FLAIR](https://github.com/ExplainableML/flair) (Fine-grained Language-Image Representations) with [HarMA](https://github.com/) (Hierarchical Multimodal Adapter) for efficient vision-language learning.

### Key Features

- **Parameter-Efficient Fine-Tuning**: Only train adapter layers while freezing the pretrained backbone
- **HarMA Integration**: Multimodal Gated Adapters for cross-modal feature enhancement
- **FLAIR Architecture**: Fine-grained attention pooling for image-text retrieval
- **Flexible Training**: Support for classification, retrieval, and contrastive learning tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MWK-FLAIR Model                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │  Vision Encoder │          │  Text Encoder   │          │
│  │   (ViT-B/16)    │          │  (Transformer)  │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│  ┌────────▼────────┐          ┌────────▼────────┐          │
│  │  HarMA Adapter  │          │  HarMA Adapter  │          │
│  │  (Trainable)    │          │  (Trainable)    │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│  ┌────────▼────────┐          ┌────────▼────────┐          │
│  │ Visual PostProc │          │  Text PostProc  │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│           └──────────┬─────────────────┘                    │
│                      │                                      │
│           ┌──────────▼──────────┐                          │
│           │  Attention Pooling  │                          │
│           │  (Fine-grained)     │                          │
│           └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mwkflair.git
cd mwkflair

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from mwkflair import create_model_and_transforms, get_tokenizer

# Create model and transforms
model, preprocess_train, preprocess_val = create_model_and_transforms(
    'flair-ViT-B-16',
    pretrained='path/to/checkpoint.pt',
    use_harma=True,
    harma_reduction_dim=128
)

# Get tokenizer
tokenizer = get_tokenizer('flair-ViT-B-16')

# Prepare inputs
image = preprocess_val(your_image).unsqueeze(0)
text = tokenizer(["a photo of a cat", "a photo of a dog"])

# Forward pass
with torch.no_grad():
    outputs = model(image, text)
    image_features = outputs['image_features']
    text_features = outputs['text_features']
    
    # HarMA-enhanced features (if use_harma=True)
    harma_image_features = outputs['harma_image_features']
    harma_text_features = outputs['harma_text_features']
```

### Training with HarMA Adapter

```python
from mwkflair import FLAIR, freeze_model_except_adapters

# Create model with HarMA
model = FLAIR(
    embed_dim=512,
    vision_cfg=vision_config,
    text_cfg=text_config,
    use_harma=True,
    harma_reduction_dim=128,
    harma_num_heads=8
)

# Freeze backbone, only train adapters
freeze_model_except_adapters(model)

# Check trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

## HarMA Adapter Components

### BiShareAdapter
Bidirectional shared adapter with attention mechanism.

```python
from mwkflair import BiShareAdapter

adapter = BiShareAdapter(
    hidden_dim=512,
    num_heads=8,
    reduction_factor=2,
    dropout=0.1
)
```

### MultimodalGatedAdapter
Multimodal gated adapter with cross-modal attention.

```python
from mwkflair import MultimodalGatedAdapter

adapter = MultimodalGatedAdapter(
    hidden_dim=512,
    reduction_dim=128,
    num_heads=8,
    dropout=0.1
)

# Self-attention mode
output = adapter(features, context=None)

# Cross-attention mode
output = adapter(image_features, context=text_features)
```

## Project Structure

```
mwkflair/
├── mwkflair/                # Core package
│   ├── __init__.py          # Package exports
│   ├── model.py             # FLAIR model with HarMA integration
│   ├── transformer.py       # Vision & Text transformers
│   ├── harma_modules.py     # HarMA adapter modules
│   ├── harma_adapter.py     # Additional adapter implementations
│   ├── factory.py           # Model creation utilities
│   ├── loss.py              # Loss functions
│   └── model_configs/       # Model configuration files
├── examples/                # Training examples
│   └── train_with_harma.py  # Example training script
├── configs/                 # Configuration files
│   └── train_config.py      # Training configuration
├── scripts/                 # Utility scripts
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── LICENSE                  # MIT License
└── README.md                # This file
```

## Loss Functions

MWK-FLAIR provides several loss functions:

- **FlairLoss**: Combined FLAIR loss with fine-grained matching
- **AdaptiveTripletLoss**: Adaptive triplet ranking loss with focal weighting
- **StandardTripletLoss**: Standard triplet ranking loss

```python
from mwkflair import AdaptiveTripletLoss

loss_fn = AdaptiveTripletLoss(
    margin=0.2,
    gamma=2.0,
    max_violation=True
)

loss = loss_fn(image_features, text_features)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mwkflair2024,
  title={MWK-FLAIR: Parameter-Efficient Vision-Language Learning with HarMA Adapters},
  author={MWK},
  year={2024}
}
```

## Acknowledgments

- [FLAIR](https://github.com/ExplainableML/flair) - Fine-grained Language-Image Representations
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open source CLIP implementation
- [HarMA](https://github.com/) - Hierarchical Multimodal Adapter

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
