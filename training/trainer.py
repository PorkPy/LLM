# training/trainer.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    """
    Trainer for transformer language model
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        lr=3e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=10000,
        checkpoint_dir='checkpoints',
        log_interval=100,
        device=None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler with linear warmup and cosine decay
        self.scheduler = self._get_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.current_step = 0
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _get_scheduler(self):
        """
        Create a learning rate scheduler with warmup and cosine decay
        """
        def lr_lambda(current_step):
            # Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """
        Train the model
        """
        self.model.train()
        
        # Training loop
        start_time = time.time()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(total=self.max_steps, desc="Training")
        pbar.update(self.current_step)
        
        while self.current_step < self.max_steps:
            epoch_loss = 0.0
            
            for batch in self.train_dataloader:
                # Skip if we've reached max_steps
                if self.current_step >= self.max_steps:
                    break
                    
                # Get batch
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits, _ = self.model(input_ids)
                
                # Calculate loss
                # Reshape for cross entropy: [batch_size * seq_length, vocab_size]
                logits = logits.view(-1, logits.size(-1))
                targets = target_ids.view(-1)
                loss = self.criterion(logits, targets)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                running_loss += loss.item()
                epoch_loss += loss.item()
                self.current_step += 1
                
                # Log progress
                if self.current_step % self.log_interval == 0:
                    avg_loss = running_loss / self.log_interval
                    elapsed = time.time() - start_time
                    
                    print(f"Step {self.current_step}/{self.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                          f"Time: {elapsed:.2f}s")
                    
                    self.train_losses.append((self.current_step, avg_loss))
                    running_loss = 0.0
                    start_time = time.time()
                    
                    # Save checkpoint
                    self._save_checkpoint()
                    
                    # Evaluate on validation set
                    if self.val_dataloader is not None:
                        val_loss = self.evaluate()
                        self.val_losses.append((self.current_step, val_loss))
                        self.model.train()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
                
        pbar.close()
        
        # Final evaluation
        if self.val_dataloader is not None:
            final_val_loss = self.evaluate()
            print(f"Final validation loss: {final_val_loss:.4f}")
            
        # Save final model
        self._save_checkpoint(is_final=True)
        
        # Plot training curve
        self._plot_training_curve()
        
    def evaluate(self):
        """
        Evaluate the model on the validation set
        
        Returns:
            val_loss: Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # Get batch
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids)
                
                # Calculate loss
                logits = logits.view(-1, logits.size(-1))
                targets = target_ids.view(-1)
                loss = self.criterion(logits, targets)
                
                val_loss += loss.item()
                
        # Calculate average loss
        val_loss /= len(self.val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        return val_loss
    
    def _save_checkpoint(self, is_final=False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.current_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_final:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_step_{self.current_step}.pt')
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['step']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from {checkpoint_path} at step {self.current_step}")
        
    def _plot_training_curve(self):
        """
        Plot training and validation loss curves
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            plt.plot(steps, losses, label='Train Loss')
            
        # Plot validation loss
        if self.val_losses:
            steps, losses = zip(*self.val_losses)
            plt.plot(steps, losses, label='Validation Loss')
            
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Curve')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.checkpoint_dir, 'training_curve.png')
        plt.savefig(plot_path)
        print(f"Training curve saved to {plot_path}")