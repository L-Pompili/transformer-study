# Transformer Architecture Study

A "from-scratch" implementation of a GPT-style transformer, focusing on modern architectural components like **RoPE** (Rotary Positional Embeddings), **SwiGLU** (Swish-Gated Linear Units), and **RMSNorm**.

## Project Structure
- `config.py`: Hyperparameters and environment variables.
- `model.py`: PyTorch implementation of the Transformer layers.
- `train.py`: Training loop.
- `data.py`: Data ingestion pipeline for the Yu-Gi-Oh card dataset.
- `predict.py`: Generating text from trained network
- `onnx_export.py`: Exporting the model using ONNX to experiment compatibility for browser inference.

## Mathematical Focus
This repository contains a low-level implementation of a transformer using modern components.
- **RoPE**: Implements rotation in the complex plane to encode relative position.
- **SwiGLU**: Implements gated linear units to improve gradient flow approximation. It is used here without biases, unlike what was used in the original paper.
- **RMSNorm**: A LayerNorm without biases.

Links to the relevant papers:
[RMSNorm](https://arxiv.org/abs/1910.07467)
[RoPE](https://arxiv.org/abs/2104.09864)
[Gated Linear Units](https://arxiv.org/abs/2002.05202)

## Usage
1. Download `all_cards.csv` from [Kaggle](https://www.kaggle.com/datasets/thiagoamancio/yu-gi-oh-tcg-card-dataset), and place it in the root.
2. Run `python train.py`.
3. Run `predict.py` for a quick example of text generation.

## Browser-supported Inference

Exporting the above model in onnx to use it in a web browser presents some challenges. One problem faced was that the use of complex numbers in the RoPE implementation was not supported, so I had to stick to a real number implementation. The onnx file still does not work to generate text on a browser due to similar compatibility issues (in progress).