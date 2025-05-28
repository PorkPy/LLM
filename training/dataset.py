# training/dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os
from tqdm import tqdm

class TextDataset(Dataset):
    """
    Dataset for language modeling
    """
    def __init__(self, text_path, tokenizer, seq_length=128, use_cache=True):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.use_cache = use_cache
        
        # Load or download the data
        self.text = self._load_data(text_path)
        
        # Tokenize the text
        self.tokenized_text = self._tokenize_text()
        
        # Create examples
        self.examples = self._create_examples()
        
    def _load_data(self, text_path):
        """
        Load text data from file or URL
        """
        # Check if text_path is a URL
        if text_path.startswith('http'):
            # Create a filename from the URL
            filename = os.path.basename(text_path)
            cache_dir = '.cache'
            os.makedirs(cache_dir, exist_ok=True)
            local_path = os.path.join(cache_dir, filename)
            
            # Download if not already cached
            if not os.path.exists(local_path) or not self.use_cache:
                print(f"Downloading {text_path}...")
                response = requests.get(text_path, stream=True)
                response.raise_for_status()
                
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                
                with open(local_path, 'wb') as f:
                    for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                        f.write(data)
                        
                print(f"Downloaded to {local_path}")
            else:
                print(f"Using cached file {local_path}")
                
            text_path = local_path
                
        # Load text from file
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        return text
    
    def _tokenize_text(self):
        """
        Tokenize the entire text
        """
        # If the tokenizer vocabulary is not built yet, build it
        if len(self.tokenizer.word_to_idx) <= len(self.tokenizer.special_tokens):
            print("Building tokenizer vocabulary...")
            # Split text into smaller chunks for tokenizer training
            chunks = [self.text[i:i+10000] for i in range(0, len(self.text), 10000)]
            self.tokenizer.build_vocab(chunks)
            
        # Tokenize the entire text
        print("Tokenizing text...")
        tokenized = self.tokenizer.encode(self.text, add_special_tokens=False)
        return tokenized
    
    def _create_examples(self):
        """
        Create training examples by sliding a window over the tokenized text
        """
        examples = []
        for i in range(0, len(self.tokenized_text) - self.seq_length):
            input_ids = self.tokenized_text[i:i + self.seq_length]
            target_ids = self.tokenized_text[i + 1:i + self.seq_length + 1]
            examples.append((input_ids, target_ids))
            
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids, target_ids = self.examples[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids)
        }
    
def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def download_sample_text():
    """
    Download a sample text for demonstration
    """
    # URL to a small text dataset (Project Gutenberg book)
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice
    
    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, 'sample_text.txt')
    
    if not os.path.exists(local_path):
        print(f"Downloading sample text from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        print(f"Downloaded to {local_path}")
    else:
        print(f"Using cached file {local_path}")
        
    return local_path