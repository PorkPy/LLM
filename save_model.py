# scripts/save_model.py
import torch
import os
import json
import sys

# Add the LLM project to path if needed
sys.path.append('path/to/your/LLM/project')

# Import your model and tokenizer classes
from model.transformer import SimpleTransformer
from training.tokenizer import SimpleTokenizer

# Create output directory
os.makedirs('src/model', exist_ok=True)

# Path to your checkpoint
checkpoint_path = 'path/to/your/LLM/checkpoints/model_step_2200.pt'
tokenizer_path = 'path/to/your/LLM/checkpoints/tokenizer.json'

print(f"Loading checkpoint from {checkpoint_path}")

# Load your checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Create a new model instance with the same parameters
model = SimpleTransformer(
    vocab_size=7100,  # Update this to match your model
    d_model=256,
    n_layers=4,
    n_heads=8,
    d_ff=1024,
    max_seq_length=128
)

# Load state dict from checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Save the model in a format suitable for inference
output_path = 'src/model/transformer_model.pt'
torch.save(model, output_path)
print(f"Model saved to {output_path}")

# Copy tokenizer
if os.path.exists(tokenizer_path):
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    
    # Save tokenizer - adjust this based on your tokenizer implementation
    with open('src/model/tokenizer.json', 'w') as f:
        if hasattr(tokenizer, 'to_dict'):
            json.dump(tokenizer.to_dict(), f)
        else:
            # If no to_dict method, create a dictionary with the essential attributes
            tokenizer_dict = {
                'word_to_idx': tokenizer.word_to_idx,
                'idx_to_word': {int(k): v for k, v in tokenizer.idx_to_word.items()},
                'vocab_size': tokenizer.vocab_size,
                'special_tokens': {
                    'pad': tokenizer.pad_token_id,
                    'unk': tokenizer.unk_token_id,
                    'bos': tokenizer.bos_token_id,
                    'eos': tokenizer.eos_token_id
                }
            }
            json.dump(tokenizer_dict, f)
    
    print(f"Tokenizer saved to src/model/tokenizer.json")
else:
    print(f"Tokenizer not found at {tokenizer_path}")

print("Model preparation complete!")