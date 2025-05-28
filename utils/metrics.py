# utils/metrics.py
import torch
import numpy as np
from collections import Counter

def calculate_perplexity(model, dataloader, device):
    """
    Calculate perplexity on a dataset
    
    Arguments:
        model: Transformer model
        dataloader: DataLoader for evaluation
        device: Device to run the model on
    
    Returns:
        perplexity: Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Cross entropy loss
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Calculate loss per token
            # logits shape: [batch_size, seq_length, vocab_size]
            # targets shape: [batch_size, seq_length]
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)
            
            # Calculate loss only for non-padding tokens
            mask = targets != 0  # Assuming 0 is the padding token
            losses = criterion(logits, targets)
            losses = losses * mask.float()
            
            total_loss += losses.sum().item()
            total_tokens += mask.sum().item()
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def calculate_top_k_accuracy(model, dataloader, device, k=5):
    """
    Calculate top-k accuracy on a dataset
    
    Arguments:
        model: Transformer model
        dataloader: DataLoader for evaluation
        device: Device to run the model on
        k: Consider a prediction correct if the correct token is in the top-k predictions
    
    Returns:
        accuracy: Top-k accuracy score (higher is better)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get top-k predictions
            # logits shape: [batch_size, seq_length, vocab_size]
            # targets shape: [batch_size, seq_length]
            pred_topk = logits.topk(k, dim=-1).indices
            targets = target_ids.unsqueeze(-1)  # Add a dimension for comparison
            
            # Check if target is in top-k predictions
            correct_topk = (pred_topk == targets).any(dim=-1)
            
            # Count only non-padding tokens
            mask = targets.squeeze(-1) != 0  # Assuming 0 is the padding token
            correct += (correct_topk & mask).sum().item()
            total += mask.sum().item()
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return accuracy

def calculate_diversity(model, tokenizer, prompts, max_length=50, temperature=1.0, top_k=50, device='cpu'):
    """
    Calculate generation diversity metrics
    
    Arguments:
        model: Transformer model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings to start generation
        max_length: Maximum length of generated sequences
        temperature: Sampling temperature
        top_k: Sample from top k tokens
        device: Device to run the model on
    
    Returns:
        metrics: Dictionary of diversity metrics
    """
    model.eval()
    
    all_tokens = []
    unique_tokens = set()
    unique_bigrams = set()
    
    for prompt in prompts:
        # Generate text
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids]).to(device)
        
        # Generate sequence
        generated = model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )[0].cpu().numpy().tolist()
        
        # Skip prompt tokens
        generated = generated[len(input_ids):]
        
        # Update metrics
        all_tokens.extend(generated)
        unique_tokens.update(generated)
        
        # Count bigrams
        for i in range(len(generated) - 1):
            bigram = (generated[i], generated[i+1])
            unique_bigrams.add(bigram)
    
    # Calculate metrics
    token_count = len(all_tokens)
    unique_token_count = len(unique_tokens)
    unique_bigram_count = len(unique_bigrams)
    
    # Type-token ratio (higher means more diverse vocabulary)
    ttr = unique_token_count / token_count if token_count > 0 else 0
    
    # Bigram entropy (higher means more diverse transitions)
    bigram_count = token_count - len(prompts)
    bigram_diversity = unique_bigram_count / bigram_count if bigram_count > 0 else 0
    
    # Token frequency distribution
    token_counter = Counter(all_tokens)
    token_freqs = np.array(list(token_counter.values())) / token_count
    
    # Entropy (higher means more uniform distribution, less repetition)
    entropy = -np.sum(token_freqs * np.log2(token_freqs))
    
    return {
        'type_token_ratio': ttr,
        'bigram_diversity': bigram_diversity,
        'entropy': entropy,
        'unique_token_percentage': 100 * unique_token_count / token_count if token_count > 0 else 0
    }