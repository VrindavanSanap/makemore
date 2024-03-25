import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import tqdm

with open("./mit.txt") as f:
    words = f.read().splitlines()
    print(len(words), max(len(w) for w in words), min(len(w) for w in words))

chars = sorted(list(set((''.join(words)))))  # get unique characters
stoi = {s: i+1 for i, s in enumerate(chars)}  # map char to int
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}         # map int to char
vocab_size = len(itos)
# create training set
xs, ys = [], []
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # print(ch1,ch2) be carful with prints and for loops long for loops are very hard to stop ,kernel become un alive
        xs.append(ix1)
        ys.append(ix2)

xs = F.one_hot(torch.tensor(xs)).float()
ys = torch.tensor(ys)


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, x):

        return self.W(x)


model = BigramModel()
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9)
for i in range(100):
    logits = model(xs)
    loss = F.cross_entropy(logits, ys)
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for i in range(20):

    ix = 0
    out = []

    while True:
        logits = model(F.one_hot(torch.tensor([ix]), num_classes=27).float())
        counts = logits.exp()
        p = counts/counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
