import torch

class Config:
    # Data
    batch_size = 64
    block_size = 256  # max_len
    train_split = 0.9

    # Model
    d_model = 256
    n_head = 8
    n_layer = 6
    d_ff = 1024  # Usually 4 * d_model
    dropout = 0.1
    
    # Training
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    dataset_path = 'yu_gi_oh_cards.csv'  # Update with actual path if needed
    model_path = 'modern_tiny_transformer.pth'
    onnx_path = 'modern_tiny_transformer.onnx'