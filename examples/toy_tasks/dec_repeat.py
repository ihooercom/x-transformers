import tqdm
import torch
import random
import numpy as np
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

NUM_BATCHES = 1000
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
GENERATE_EVERY = 10
NUM_TOKENS = 16 + 2
SEQ_LEN = 8 + 2
START_TOKEN = 0
END_TOKEN = 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cycle():
    while True:
        seed_everything(1)
        prefix = torch.ones((BATCH_SIZE, 1)).long() * START_TOKEN
        suffix = torch.ones((BATCH_SIZE, 1)).long() * END_TOKEN
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, 1)).expand(BATCH_SIZE, SEQ_LEN - 2)
        src = torch.cat((prefix, src, suffix), 1)
        mask = torch.ones(BATCH_SIZE, src.shape[1]).bool()
        yield (src, mask)


decoder = TransformerWrapper(
    num_tokens=NUM_TOKENS, max_seq_len=SEQ_LEN, attn_layers=Decoder(dim=512, depth=6, heads=8, attn_one_kv_head=True)
)
model = AutoregressiveWrapper(decoder)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, mask = next(cycle())

    optim.zero_grad()
    loss = model(src, mask=mask)
    loss.backward()
    optim.step()

    print(f'{i}: {loss.item()}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, mask = next(cycle())
        src, mask = src[:1], mask[:1]
        start_tokens = src[:, :2]

        preds = model.generate(start_tokens, SEQ_LEN - 2)
        incorrects = (src[:, 2:] != preds).float().abs().sum()

        print("target:  ", src[:, 1:])
        print("predicted output:  ", preds)
        print(f"incorrects: {incorrects}")
