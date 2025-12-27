# SentimentScope: Sentiment Analysis using Transformers

A transformer-based deep learning model built from scratch for binary sentiment classification on the IMDB movie reviews dataset.

## Project Overview

This project implements a complete transformer architecture to classify movie reviews as positive or negative. Built as part of the AWS AI & ML Scholarship program, it demonstrates deep understanding of transformer mechanics, from attention mechanisms to classification heads.

**Final Achievement: 81.00% test accuracy** ✨

## Key Features

- **Custom Transformer Architecture**: Built entirely from scratch including:
  - Multi-head self-attention mechanisms
  - Positional embeddings
  - Feed-forward networks with GELU activation
  - Layer normalization and residual connections
  
- **Optimized for Classification**: 
  - Mean pooling for sequence aggregation
  - Binary classification head
  - Cross-entropy loss optimization

- **Production-Ready Pipeline**:
  - Custom PyTorch Dataset implementation
  - Efficient DataLoader with batching
  - BERT tokenizer integration
  - GPU-accelerated training

## Dataset

**IMDB Movie Reviews Dataset**
- 50,000 reviews total (25,000 train / 25,000 test)
- Perfectly balanced (50% positive / 50% negative)
- Average review length: ~234 words


## Model Architecture

### Configuration
```python
config = {
    "vocabulary_size": 30522,  # BERT tokenizer vocab
    "num_classes": 2,          # Binary classification
    "d_embed": 256,            # Embedding dimension
    "context_size": 256,       # Max sequence length
    "layers_num": 6,           # Transformer blocks
    "heads_num": 4,            # Attention heads
    "head_size": 64,           # Per-head dimension
    "dropout_rate": 0.1,       # Regularization
    "use_bias": True
}
```

### Components

1. **Token & Positional Embeddings**
   - Converts token IDs to dense vectors
   - Adds positional information to maintain sequence order

2. **Multi-Head Self-Attention**
   - 4 parallel attention heads
   - Causal masking for autoregressive behavior
   - Scaled dot-product attention

3. **Feed-Forward Networks**
   - Two-layer MLP with GELU activation
   - 4x expansion factor
   - Dropout for regularization

4. **Classification Head**
   - Mean pooling across sequence dimension
   - Linear layer mapping to 2 classes
   - Outputs raw logits for cross-entropy loss

## Training Process

- **Optimizer**: AdamW (lr=3e-4)
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 4
- **Batch Size**: 32
- **Hardware**: NVIDIA GPU (CUDA-enabled)

### Training Results

| Epoch | Train Loss | Val Accuracy |
|-------|-----------|--------------|
| 1     | 0.4773    | 77.96%       |
| 2     | 0.3902    | 80.92%       |
| 3     | 0.3061    | 83.04%       |
| 4     | 0.2825    | 82.12%       |

**Final Test Accuracy: 81.00%**

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/sentimentscope.git
cd sentimentscope

# Install dependencies
pip install torch torchvision pandas matplotlib transformers

# Download IMDB dataset
# Dataset is automatically loaded from aclImdb_v1.tar.gz
```

## Usage

### Training

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create datasets
train_dataset = IMDBDataset(train_data, tokenizer, MAX_LENGTH=256)
val_dataset = IMDBDataset(val_data, tokenizer, MAX_LENGTH=256)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize and train model
model = DemoGPT(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# Training loop (see notebook for full implementation)
```

### Inference

```python
# Load trained model
model = DemoGPT(config)
model.load_state_dict(torch.load('sentiment_model_checkpoint.pth'))
model.eval()

# Predict sentiment
review = "This movie was absolutely fantastic!"
tokens = tokenizer(review, return_tensors="pt", max_length=256, 
                   truncation=True, padding="max_length")
logits = model(tokens['input_ids'])
prediction = torch.argmax(logits, dim=1)
print("Positive" if prediction == 1 else "Negative")
```

## Project Structure

```
sentimentscope/
│
├── SentimentScope_starter.ipynb  # Main notebook with complete implementation
├── sentiment_model_checkpoint.pth # Trained model weights
├── aclImdb_v1.tar.gz             # IMDB dataset
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Performance Optimization

Several architectural improvements were made to achieve 81% accuracy:

1. **Increased Embedding Dimension**: 128 → 256
2. **Added Transformer Layers**: 4 → 6 blocks
3. **Extended Context Window**: 128 → 256 tokens
4. **Attention Head Optimization**: Balanced 4 heads with 64-dim each

## Lessons Learned

1. **Transformer Depth Matters**: Increasing from 4 to 6 layers provided measurable accuracy gains
2. **Context Size Impact**: Longer sequences (256 vs 128 tokens) captured more review context
3. **Hyperparameter Sensitivity**: Small changes in architecture had significant effects
4. **Data Quality**: Balanced dataset simplified training but required careful validation split
5. **GPU Efficiency**: Batch processing on GPU reduced training time dramatically

## Future Improvements

- [ ] Implement learning rate scheduling
- [ ] Add gradient clipping for stability
- [ ] Experiment with different pooling strategies (CLS token, max pooling)
- [ ] Try data augmentation techniques
- [ ] Implement attention visualization
- [ ] Add model interpretability features
- [ ] Test on other sentiment datasets
