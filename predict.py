import torch
from config import Config
from model import ModernTinyTransformer
from data import prepare_data

conf = Config()
_, _, tokenizer = prepare_data(conf) 

model = ModernTinyTransformer(conf, tokenizer.vocab_size)
model.load_state_dict(torch.load('modern_tiny_transformer.pth', map_location='cuda'))
model.to('cuda')
model.eval()

start_text = "This card"
T = 0.7
encoded = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to('cuda')

for _ in range(200):
    logits, _ = model(encoded)
    last_logits = logits[:, -1, :] 
    probs = torch.softmax(last_logits/T, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    encoded = torch.cat((encoded, next_token), dim=1)

print(tokenizer.decode(encoded[0].tolist()))