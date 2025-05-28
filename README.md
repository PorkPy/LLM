# LLM
Building a simple LLM by hand just for fun

transformer-from-scratch/\
├── model/\
│   ├── __init__.py\
│   ├── attention.py       # Attention mechanisms\
│   ├── embedding.py       # Token and positional embeddings\
│   ├── encoder.py         # Transformer encoder blocks\
│   ├── decoder.py         # Optional: for seq2seq tasks\
│   └── transformer.py     # Full transformer model\
├── training/\
│   ├── __init__.py\
│   ├── dataset.py         # Dataset preparation\
│   ├── tokenizer.py       # Simple tokenizer implementation\
│   └── trainer.py         # Training loop and logic\
├── utils/\
│   ├── __init__.py\
│   ├── visualization.py   # Attention visualization tools\
│   └── metrics.py         # Evaluation metrics\
├── app.py                 # Streamlit application\
├── train.py               # Training script\
└── requirements.txt\
