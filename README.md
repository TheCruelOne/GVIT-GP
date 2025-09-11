GVIT-GP: Injecting the Genomic Relationship Matrix as an Inductive Bias into a Vision Transformer via Cross-Attention for Genomic Prediction

1. Abstract

Applying Transformer architectures to genomic prediction (GP) is hindered by two fundamental challenges: the trade-off between computational cost and the integrity of genomic information, and the discrepancy between model complexity and the limited sample sizes typical of biological datasets. To address these challenges, we propose GViT-GP, a Vision Transformer (ViT) architecture specifically tailored for genotyping data. GViT-GP addresses the first challenge by employing a Selective Patch Embedding (SPE) strategy, which prioritizes and selects informative loci before tokenization, thereby preserving critical genetic information within computational constraints. Crucially, to resolve the conflict between model complexity and data scarcity, GViT-GP incorporates population structure, encapsulated in the Genomic Relationship Matrix (GRM), as a dynamic inductive bias. This is implemented through a dual-pathway, cross-attention architecture, wherein the SNP sequence serves as the query to actively integrate structural information from the GRM. This mechanism effectively mitigates the difficulty of training ViTs on small-sample datasets. Evaluated on benchmark datasets spanning 4 species and 20 traits, GViT-GP significantly outperforms four established methods, including GBLUP and LightGBM, across a majority of the predictive tasks. Our results demonstrate the effectiveness of GViT-GP and establish its potential as a promising next-generation framework for genomic prediction.

2. Overview

This project implements a novel deep learning architecture for genomic prediction. It leverages a Vision Transformer (ViT)-like model to capture complex patterns in Single Nucleotide Polymorphism (SNP) data and fuses this information with a Genomic Relationship Matrix (GRM) using a cross-attention mechanism to predict continuous phenotypes.

3.Features

Hybrid Model Architecture: Combines a ViT-style backbone for SNP sequences with an MLP for GRM vectors.

Cross-Attention Fusion: Utilizes an advanced Cross-Attention module to effectively fuse the two different data modalities.

Feature Selection: Integrates LightGBM to reduce the dimensionality and noise of SNP data, improving model efficiency and performance.

Rigorous Evaluation Framework: Employs a robust evaluation scheme with K-fold cross-validation (train mode) and a final hold-out set evaluation (test mode).

Multi-Trait Processing: Capable of processing multiple phenotypic traits individually and sequentially.

TensorBoard Integration: Logs training/validation losses and final performance metrics for visualization and tracking.

Attention Visualization: Generates attention maps for model interpretability, showing which genomic regions the model focuses on.

4. File Structure

```
GViT-GP/
├── datasets/                 # Location for dataset files
│   └── dataset.npz
├── logs/                     # Stores training logs, model weights, and results
├── src/                      # All source code
│   ├── __init__.py
│   ├── config.py             # Configuration for model architecture, training hyperparameters, etc.
│   ├── dataloader.py         # Logic for data loading, preprocessing, and K-Fold generation
│   ├── evaluate.py           # Performance evaluation and attention visualization functions
│   ├── model.py              # Definition of the CrossAttentionFusionModel architecture
│   ├── train.py              # Main logic for training and testing execution
│   └── utils.py              # Utility functions like seeding, GRM calculation, feature selection
└──  main.py                  # The main script (entry point) to run the project
```
5. Model Architecture

The CrossAttentionFusionModel consists of two main branches:

SNP-ViT Branch: Takes the SNP sequence as input, embeds it, and passes it through Transformer encoder blocks similar to a Vision Transformer (ViT) to learn the complex relationships between SNPs.

GRM-MLP Branch: Takes the GRM vector, which represents the genetic relationship between samples, and processes it through a Multi-Layer Perceptron (MLP) to extract high-level features.

The features extracted from these two branches are then fused in a Cross-Attention module. The SNP features act as the "Query," while the GRM features act as the "Key" and "Value." This process effectively injects genetic relationship information into the SNP feature vectors for a more informed final prediction.