# EfficientNet for Image Classification on PACS Dataset

## Project Overview
This project implements a domain generalization approach for image classification using EfficientNet on the PACS dataset. The model is trained on multiple source domains ("art_painting", "cartoon", "photo") and evaluated on a target domain ("sketch").

## Dataset
- **PACS dataset**: Contains images across 4 domains (art_painting, cartoon, photo, sketch)
- **Classes**: 7 categories (dog, elephant, giraffe, guitar, horse, house, person)
- **Organization**: Hierarchical folder structure with domain and class subfolders

## Model Architecture

### EfficientNet
- **Base Model**: EfficientNet-B0
- **Key Components**:
  - **Mobile Inverted Bottleneck Convolution (MBConv)** blocks as the primary building blocks
  - **Squeeze-and-Excitation (SE)** modules for channel-wise attention
  - **Compound Scaling** using the formula:
    - Depth: (d) = alpha^phi
    - Width: (w) = beta^phi
    - Resolution: (r) = gamma^phi
  - **B0 Parameters**: The project uses EfficientNet-B0 with phi=0, resolution=224x224, and dropout rate=0.2 (no additional scaling applied to the base architecture)

### Architectural Details
- **Entry Flow**: Standard convolution with batch normalization and SiLU activation
- **Middle Flow**: Series of MBConv blocks with varying expansion ratios, channels, and kernel sizes
- **Exit Flow**: Final convolution, global average pooling, and classification layer
- **Regularization**: Stochastic depth during training (survival probability 0.8)
- **Activation**: SiLU (Swish) activation functions throughout the network

## Training Process
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32 with gradient accumulation over 4 steps (effective batch size 128)
- **Training Strategy**: 100 epochs with evaluation after each epoch
- **Data Augmentation**: Basic resizing and normalization (ImageNet means and standard deviations)

## Evaluation Metrics
- **Accuracy**: Percentage of correctly classified images in the target domain

## Usage
```python
# Train and evaluate model
python main.py