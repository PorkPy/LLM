# app.py
import streamlit as st
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Add this import
from model.transformer import SimpleTransformer
from training.tokenizer import SimpleTokenizer
from utils.visualization import plot_attention_heads, visualize_model_attention

# Set page config
st.set_page_config(
    page_title="Transformer Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(checkpoint_path, tokenizer_path, device):
    """
    Load model and tokenizer
    """
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    
    # Create model with the SAME parameters as during training
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        max_seq_length=128,  # Match the sequence length used in training
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def main():
    st.title("🤖 Transformer from Scratch")
    st.markdown("""
    This app demonstrates a transformer model built from scratch with PyTorch.
    You can generate text and visualize attention patterns to understand how the model works.
    """)
    
    # Sidebar
    st.sidebar.header("Model Settings")
    
    # Check for available checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        st.error("No checkpoints found. Please train the model first using train.py.")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        st.error("No checkpoint files found in the checkpoints directory.")
        return
    
    # Model selection
    checkpoint_file = st.sidebar.selectbox(
        "Select checkpoint",
        checkpoint_files,
        index=0
    )
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    # Tokenizer path
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        st.error("Tokenizer file not found. Please train the model first using train.py.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model(checkpoint_path, tokenizer_path, device)
        st.sidebar.success(f"Model loaded from {checkpoint_file}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Model stats
    st.sidebar.subheader("Model Stats")
    st.sidebar.text(f"Vocabulary size: {len(tokenizer.word_to_idx)}")
    st.sidebar.text(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generation settings
    st.sidebar.subheader("Generation Settings")
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                  help="Higher values produce more diverse text, lower values are more deterministic")
    top_k = st.sidebar.slider("Top-k", min_value=1, max_value=100, value=50, step=1,
                            help="Sample from top k most probable tokens")
    max_length = st.sidebar.slider("Max Length", min_value=10, max_value=100, value=50, step=10,
                                 help="Maximum number of tokens to generate")
    
    # Visualization settings
    st.sidebar.subheader("Visualization Settings")
    layer_idx = st.sidebar.slider("Layer", min_value=0, max_value=model.encoder.layers.__len__()-1, value=0,
                                help="Transformer layer to visualize")
    visualize_all_heads = st.sidebar.checkbox("Show all attention heads", value=True,
                                           help="Show all attention heads or just one")
    if not visualize_all_heads:
        head_idx = st.sidebar.slider("Head", min_value=0, max_value=model.encoder.layers[0].self_attention.n_heads-1, value=0,
                                   help="Attention head to visualize")
    else:
        head_idx = None
    
    # Main content
    tabs = st.tabs(["Text Generation", "Attention Visualization", "Model Architecture", "Diagnostics"])
    
    # Text Generation Tab
    with tabs[0]:
        st.header("Text Generation")
        
        prompt = st.text_area("Enter a prompt", "Once upon a time", height=100)
        
        if st.button("Generate"):
            with st.spinner("Generating text..."):
                # Tokenize prompt
                input_ids = tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids]).to(device)
                
                # Generate text with UNK blocking
                with torch.no_grad():
                    # Start with the input tensor
                    output_tensor = input_tensor.clone()
                    
                    # Generate tokens one by one
                    for _ in range(max_length):
                        # Get model predictions
                        logits, _ = model(output_tensor)
                        
                        # Get logits for the last token
                        next_token_logits = logits[0, -1, :].clone()
                        
                        # Block UNK token
                        next_token_logits[tokenizer.unk_token_id] = -float('inf')
                        
                        # Apply temperature
                        next_token_logits = next_token_logits / temperature
                        
                        # Apply top-k filtering
                        if top_k > 0:
                            # Zero out all values below the top-k
                            values, _ = torch.topk(next_token_logits, top_k)
                            min_value = values[-1]
                            next_token_logits[next_token_logits < min_value] = -float('inf')
                        
                        # Sample from the filtered distribution
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).unsqueeze(0)
                        
                        # Add the sampled token to the sequence
                        output_tensor = torch.cat((output_tensor, next_token), dim=1)
                    
                    # Decode generated text
                    generated_text = tokenizer.decode(output_tensor[0].tolist())
                
                # Display result
                st.subheader("Generated Text")
                st.write(generated_text)
                
                # Show token probabilities for the last position
                if st.checkbox("Show token probabilities"):
                    with torch.no_grad():
                        logits, _ = model(input_tensor)
                        probs = torch.softmax(logits[0, -1], dim=-1)
                        
                    # Get top-10 tokens and their probabilities
                    top_probs, top_indices = probs.topk(10)
                    
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Get token strings
                    top_tokens = [tokenizer.idx_to_word.get(idx.item(), '<UNK>') for idx in top_indices]
                    
                    # Plot bars
                    ax.barh(top_tokens[::-1], top_probs.cpu().numpy()[::-1])
                    ax.set_xlabel('Probability')
                    ax.set_title('Top-10 token probabilities for next prediction')
                    
                    st.pyplot(fig)
    
    # Attention Visualization Tab
    with tabs[1]:
        st.header("Attention Visualization")
        
        viz_text = st.text_area("Enter text to visualize attention", "The quick brown fox jumps over the lazy dog.", height=100,
                              key="viz_text")
        
        if st.button("Visualize Attention"):
            with st.spinner("Generating attention visualization..."):
                # Get tokens and attention weights
                tokens, attentions = visualize_model_attention(model, viz_text, tokenizer, device)
                
                # Display tokens
                st.subheader("Tokens")
                st.write(" ".join(tokens))
                
                # Plot attention patterns
                st.subheader("Attention Patterns")
                fig = plot_attention_heads(attentions, tokens, layer_idx, head_idx, cmap='viridis')
                st.pyplot(fig)
                
                # Explanation
                st.subheader("What am I looking at?")
                st.markdown("""
                These heatmaps show the attention patterns of the transformer model. Each cell represents how much 
                a token (row) attends to another token (column) when processing the input. Brighter colors indicate 
                stronger attention weights.
                
                - **Multiple heads**: Transformers use multiple attention heads in parallel, each potentially focusing on different aspects of the relationships between tokens.
                - **Layers**: The model has multiple layers, with each layer's attention building on the previous layer's representations.
                
                These visualizations help understand what patterns the model has learned, such as:
                - Attending to relevant context words
                - Capturing syntactic relationships
                - Learning positional dependencies
                """)
    
    # Model Architecture Tab
    with tabs[2]:
        st.header("Model Architecture")
        
        st.subheader("Transformer Architecture")
        st.markdown("""
        This demo implements a simplified transformer model as described in the paper "Attention is All You Need" by Vaswani et al.
        
        Key components:
        
        1. **Token Embeddings**: Convert tokens to vectors
        2. **Positional Encoding**: Add position information
        3. **Multi-Head Attention**: Process relationships between tokens
           - Multiple attention heads learn different relationship patterns
           - Each head uses scaled dot-product attention
        4. **Feed-Forward Networks**: Process token representations
        5. **Layer Normalization**: Stabilize learning
        6. **Residual Connections**: Help with gradient flow
        
        The model for this demo includes:
        - 4 transformer encoder layers
        - 8 attention heads per layer
        - 256-dimensional embeddings
        - 1024-dimensional feed-forward networks
        
        For text generation, the model uses:
        - Temperature sampling for controlling randomness
        - Top-k sampling to filter unlikely tokens
        """)
        
        st.subheader("Self-Attention Mechanism")
        st.markdown("""
        The core of the transformer is the self-attention mechanism:
        
        1. Each token's embedding is transformed into three vectors:
           - **Query (Q)**: What the token is looking for
           - **Key (K)**: What the token offers to others
           - **Value (V)**: The information the token contains
           
        2. Attention scores are calculated as: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V`
           - This captures how much each token should attend to every other token
           
        3. Multiple attention heads allow the model to focus on different aspects of relationships simultaneously
        
        The attention visualization tab shows these attention patterns in action!
        """)
    
    # Diagnostics Tab
    with tabs[3]:
        st.header("Model Diagnostics")
        
        st.subheader("Token Distribution Analysis")
        
        diag_prompt = st.text_area("Enter a prompt for diagnostics", "Once upon a time", height=100, key="diag_prompt")
        
        if st.button("Analyze Token Distribution"):
            with st.spinner("Analyzing token distribution..."):
                # Tokenize prompt
                input_ids = tokenizer.encode(diag_prompt)
                input_tensor = torch.tensor([input_ids]).to(device)
                
                # Get model predictions
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                    
                # Get logits for the last token
                next_token_logits = logits[0, -1, :]
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Get top 20 tokens
                top_k = 20
                top_probs, top_indices = torch.topk(probs, top_k)
                
                # Display results
                results = []
                for i in range(top_k):
                    token_id = top_indices[i].item()
                    token = tokenizer.idx_to_word.get(token_id, f"<ID:{token_id}>")
                    prob = top_probs[i].item()
                    results.append({"Token": token, "ID": token_id, "Probability": f"{prob:.6f}"})
                
                st.write(pd.DataFrame(results))
                
                # Check for UNK concentration
                unk_prob = probs[tokenizer.unk_token_id].item()
                st.write(f"UNK token probability: {unk_prob:.6f}")
                st.write(f"Sum of top {top_k} token probabilities: {sum(top_probs.tolist()):.6f}")
                st.write(f"Sum of all probabilities: {sum(probs.tolist()):.6f}")
                
                # Visualize token probability distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort probabilities
                sorted_probs, _ = torch.sort(probs, descending=True)
                sorted_probs = sorted_probs[:100].cpu().numpy()  # Show top 100 for better visualization
                
                # Plot distribution
                ax.plot(range(len(sorted_probs)), sorted_probs)
                ax.set_xlabel('Token Rank')
                ax.set_ylabel('Probability')
                ax.set_title('Token Probability Distribution (Top 100 Tokens)')
                ax.set_yscale('log')
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Test generation with forced non-UNK
                st.subheader("Test Generation (Forced non-UNK)")
                with torch.no_grad():
                    # Get next token logits
                    next_token_logits = logits[0, -1, :].clone()
                    
                    # Block UNK token
                    next_token_logits[tokenizer.unk_token_id] = -float('inf')
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Convert to probabilities
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    
                    # Sample tokens
                    num_samples = 10
                    sampled_tokens = []
                    
                    for _ in range(num_samples):
                        next_token = torch.multinomial(probs, 1).item()
                        token = tokenizer.idx_to_word.get(next_token, f"<ID:{next_token}>")
                        sampled_tokens.append(token)
                    
                    st.write(f"Next 10 sampled tokens with UNK blocked: {', '.join(sampled_tokens)}")

if __name__ == "__main__":
    main()