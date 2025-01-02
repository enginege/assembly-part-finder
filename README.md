# Assembly Retrieval System

This project implements a multi-modal retrieval system for mechanical assemblies that combines image-based, part-based, and graph-based features to enable efficient and accurate similarity search across CAD assemblies.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Technical Details](#technical-details)
- [Memory Management](#memory-management)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)


## Features

- **Multi-modal Retrieval:** Combines image, part, and graph features for comprehensive similarity search
- **Memory-Efficient Caching:** Implements adaptive caching with memory monitoring
- **Mixed Precision Training:** Uses PyTorch AMP for efficient training
- **Flexible Query Types:** Supports both full assembly and individual part queries
- **Interactive Visualization:** Generates visual results for easy interpretation
- **Early Stopping:** Prevents overfitting during training
- **Gradient Accumulation:** Enables training with larger effective batch sizes

## Project Structure
```plaintext
retrieval_system/
├── dataset.py # Dataset loading and caching implementation
├── feature_extractors.py # Image and graph feature extraction models
├── losses.py # Implementation of triplet and contrastive losses
├── main.py # Entry point and CLI interface
├── retrieval.py # Core retrieval system implementation
├── retrieval_model.py # Multi-modal neural network model
├── trainer.py # Model training and validation logic
└── visualization.py # Results visualization utilities
```


## Installation

1. Create a new conda environment:

```bash
conda create -n retrieval python=3.10
conda activate retrieval
```


2. Install PyTorch and CUDA dependencies:

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

3. Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Training Mode

```bash
python run.py --mode train --data_dir /path/to/dataset --batch_size 4 --epochs 10
```

### Query Mode

```bash
python run.py --mode query --query_image /path/to/image.png --query_type [assembly|part] --data_dir path/to/dataset
```

## System Architecture

### 1. Feature Extraction

The system uses three parallel feature extractors:

- **Image Feature Extractor:**
  - ResNet18 backbone
  - 512-dimensional output
  - ImageNet pretrained weights
  - Adaptive pooling for variable sizes

- **Graph Feature Extractor:**
  - GCN-based encoder
  - 512-dimensional node embeddings
  - Edge feature integration
  - Global pooling layer

- **Part Feature Extractor:**
  - Specialized ResNet18
  - Part-specific augmentations
  - Consistent feature space

### 2. Multi-modal Fusion

Features are combined using:
- Weighted feature concatenation
- Cross-modal attention mechanisms
- Normalized feature spaces

### 3. Memory-Aware Caching

Implements an adaptive caching system that:
- Monitors system memory usage
- Automatically adjusts cache size
- Implements LRU eviction policy
- Provides separate train/validation caches

### 4. Training Pipeline

The training process includes:
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Cache-aware batch processing

## Technical Details

### Feature Dimensions

- Image Features: 512-dimensional vectors
- Graph Features: 512-dimensional vectors
- Combined Embedding: 512-dimensional unified space

### Loss Functions

1. **Triplet Loss:**
   - Margin: 1.0
   - Hard negative mining
   - Batch-wise mining strategy

2. **Contrastive Loss:**
   - Temperature: 0.07
   - InfoNCE formulation
   - Cross-modal contrastive learning

### Caching System

The `MemoryAwareCache` implements:
- Memory usage monitoring (30% threshold)
- Automatic cache clearing
- Statistics tracking
- Separate train/val caches

## Memory Management

The system implements several memory optimization strategies:

1. **Adaptive Batch Processing:**
   - Dynamic batch size adjustment
   - Gradient accumulation
   - Memory-aware cache eviction

2. **Mixed Precision Training:**
   - FP16 computation
   - FP32 parameter storage
   - Automatic loss scaling

3. **Resource Monitoring:**
   - System memory tracking
   - Process memory tracking
   - GPU memory management

## Visualization

The system provides two types of visualizations:

1. **Assembly Query Results:**
   - Query assembly
   - Top-k similar assemblies
   - Similarity scores
   - Visual grid layout

2. **Part Query Results:**
   - Query part
   - Matching parts
   - Parent assemblies
   - Similarity metrics

Results are saved as PNG files in the `retrieval_results` directory.

## Troubleshooting

### Common Issues

1. **Memory Issues:**
   - Reduce batch size
   - Enable cache clearing: `cache_manager.clear_unused_cache()`
   - Monitor memory usage with `print(torch.cuda.memory_summary())`

2. **Training Issues:**
   - Check learning rate scheduling
   - Monitor loss convergence
   - Verify data loading pipeline

3. **Retrieval Issues:**
   - Verify index building
   - Check similarity thresholds
   - Validate embedding dimensions

---

This retrieval system complements the STEP file processing pipeline by enabling efficient search and retrieval across processed assemblies. Together, they form a complete pipeline for CAD assembly processing, analysis, and retrieval.