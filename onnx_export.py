import torch
import torch.onnx
from config import Config
from data import prepare_data
from model import ModernTinyTransformer

def export_onnx():
    config = Config()
    
    # Need tokenizer to determine vocab_size for model init
    _, _, tokenizer = prepare_data(config)
    
    model = ModernTinyTransformer(config, tokenizer.vocab_size)
    
    try:
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        print(f"Loaded model from {config.model_path}")
    except FileNotFoundError:
        print("Model file not found. Exporting untrained model.")

    model.to('cpu')
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, config.block_size))

    print(f"Exporting to {config.onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        config.onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("Export complete.")

if __name__ == "__main__":
    export_onnx()