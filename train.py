# train.py
import argparse
import torch
import torch.nn as nn
import os
import random
import numpy as np
from model.transformer import SimpleTransformer
from training.tokenizer import SimpleTokenizer
from training.dataset import TextDataset, get_dataloader, download_sample_text
from training.trainer import Trainer
from utils.metrics import calculate_perplexity, calculate_top_k_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple transformer model for language modeling')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training data file (text file)')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Size of vocabulary')
    parser.add_argument('--seq_length', type=int, default=128,
                        help='Sequence length for training')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of transformer layers')

    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum number of training steps')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Download sample text if no data path provided
    if args.data_path is None:
        args.data_path = download_sample_text()
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    
    # Create dataset
    full_dataset = TextDataset(
        text_path=args.data_path,
        tokenizer=tokenizer,
        seq_length=args.seq_length
    )
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Save tokenizer
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Create model
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_length=args.seq_length,
        dropout=args.dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        device=device
    )
    
    # Resume from checkpoint if provided
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the final model
    print("Evaluating final model...")
    perplexity = calculate_perplexity(model, val_dataloader, device)
    top_1_acc = calculate_top_k_accuracy(model, val_dataloader, device, k=1)
    top_5_acc = calculate_top_k_accuracy(model, val_dataloader, device, k=5)
    
    print(f"Final metrics:")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Top-1 Accuracy: {top_1_acc:.4f}")
    print(f"  Top-5 Accuracy: {top_5_acc:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()