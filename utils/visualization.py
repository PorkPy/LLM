# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_attention_heads(attention, tokens=None, layer_idx=0, head_idx=None, cmap='viridis'):
    """
    Plot attention patterns from transformer model
    
    Arguments:
        attention: List of torch.Tensor attention weights from model
        tokens: List of token strings to display on axes
        layer_idx: Index of the layer to visualize
        head_idx: Index of the head to visualize, if None visualize all heads
        cmap: Matplotlib colormap name
    """
    # Get attention weights for the specified layer
    attn = attention[layer_idx].detach().cpu().numpy()
    
    # Shape: [batch_size, n_heads, seq_length, seq_length]
    n_heads = attn.shape[1]
    
    # Default to first example in batch
    attn = attn[0]
    
    # Set up the figure
    if head_idx is not None:
        # Plot a single attention head
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_attention_map(attn[head_idx], ax, tokens, tokens, cmap=cmap)
        plt.title(f"Layer {layer_idx+1}, Head {head_idx+1}")
    else:
        # Plot all attention heads
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
        
        for i, ax in enumerate(axes.flat):
            if i < n_heads:
                plot_attention_map(attn[i], ax, tokens, tokens, cmap=cmap)
                ax.set_title(f"Head {i+1}")
            else:
                ax.axis('off')
                
        plt.suptitle(f"Layer {layer_idx+1} Attention Heads", fontsize=16)
        
    plt.tight_layout()
    return fig

def plot_attention_map(attention_weights, ax, row_labels=None, col_labels=None, cmap='viridis'):
    """
    Plot a single attention map
    
    Arguments:
        attention_weights: 2D numpy array of attention weights
        ax: Matplotlib axis to plot on
        row_labels: Labels for the rows (target tokens)
        col_labels: Labels for the columns (source tokens)
        cmap: Matplotlib colormap name
    """
    # Create a custom colormap with white as the lowest value
    if cmap == 'attention':
        colors = ['white', 'teal', 'darkblue']
        cmap = LinearSegmentedColormap.from_list('attention_cmap', colors, N=100)
        
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        annot=False,
        cmap=cmap,
        ax=ax,
        square=True,
        cbar=True,
        xticklabels=col_labels,
        yticklabels=row_labels
    )
    
    # Set labels
    ax.set_xlabel('Source tokens')
    ax.set_ylabel('Target tokens')
    
    # Rotate x-axis labels if provided
    if col_labels is not None:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
    return ax

def visualize_model_attention(model, text, tokenizer, device='cpu'):
    """
    Visualize attention patterns for a given text input
    
    Arguments:
        model: Transformer model
        text: Input text to visualize attention for
        tokenizer: Tokenizer to convert text to tokens
        device: Device to run the model on
    
    Returns:
        tokens: List of token strings
        attentions: List of attention weight tensors
    """
    model.eval()
    
    # Tokenize input text
    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Get model predictions and attention weights
    with torch.no_grad():
        _, attentions = model(input_tensor)
        
    # Convert token IDs to strings
    tokens = [tokenizer.idx_to_word.get(idx, '<UNK>') for idx in input_ids]
    
    return tokens, attentions