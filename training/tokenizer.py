# training/tokenizer.py
import re
import os
import json
import torch
from collections import Counter

class SimpleTokenizer:
    """
    A simple tokenizer that splits text into words/subwords
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            
        self.pad_token_id = self.special_tokens['<PAD>']
        self.unk_token_id = self.special_tokens['<UNK>']
        self.bos_token_id = self.special_tokens['<BOS>']
        self.eos_token_id = self.special_tokens['<EOS>']
        
    def build_vocab(self, texts):
        """
        Build vocabulary from a list of texts
        
        Arguments:
            texts: List of strings
        """
        # Tokenize texts
        all_words = []
        for text in texts:
            words = self._tokenize(text)
            all_words.extend(words)
            
        # Count word frequencies
        counter = Counter(all_words)
        
        # Sort by frequency and take most common
        words_and_frequencies = counter.most_common(self.vocab_size - len(self.special_tokens))
        
        # Add to vocabulary
        for word, _ in words_and_frequencies:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
        print(f"Vocabulary built with {len(self.word_to_idx)} tokens")