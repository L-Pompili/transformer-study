import torch
from torch.utils.data import DataLoader
from config import Config
from data import prepare_data
from model import ModernTinyTransformer

def estimate_loss(model, dataloaders, ctx):
    out = {}
    model.eval()
    for split, loader in dataloaders.items():
        losses = torch.zeros(Config.eval_iters)
        for k, (X, Y) in enumerate(loader):
            if k >= Config.eval_iters: break
            X, Y = X.to(Config.device), Y.to(Config.device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    config = Config()
    train_data, val_data, tokenizer = prepare_data(config)
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}

    model = ModernTinyTransformer(config, tokenizer.vocab_size)
    model.to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if 'cuda' in config.device else torch.no_grad()

    print(f"Starting training on {config.device}...")
    iter_num = 0
    while iter_num < config.max_iters:
        for X, Y in train_loader:
            if iter_num >= config.max_iters: break
            
            X, Y = X.to(config.device), Y.to(config.device)
            
            if iter_num % config.eval_interval == 0:
                losses = estimate_loss(model, dataloaders, ctx)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            with ctx:
                logits, loss = model(X, Y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            iter_num += 1

    print("Training finished.")
    torch.save(model.state_dict(), config.model_path)
    print(f"Model saved to {config.model_path}")

if __name__ == "__main__":
    train()