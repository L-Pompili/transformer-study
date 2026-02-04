import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def fit(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class YuGiOhDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

def prepare_data(config):
    try:
        df = pd.read_csv(config.dataset_path)
        text = "\n".join(df['description'].dropna().astype(str).tolist())
    except FileNotFoundError:
        print("Data file not found. Using dummy text for demonstration.")
        text = "Dummy text data " * 1000

    tokenizer = Tokenizer()
    tokenizer.fit(text)

    dataset = YuGiOhDataset(text, tokenizer, config.block_size)
    
    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_data, val_data, tokenizer