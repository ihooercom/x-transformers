import tqdm
import torch
from x_transformers import XTransformer
import random
import numpy as np

# constants

NUM_BATCHES = 1000
BATCH_SIZE = 10
LEARNING_RATE = 3e-4
GENERATE_EVERY = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 64 + 1

# helpers

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cycle():
    while True:
        seed_everything(1)
        prefix = torch.ones((BATCH_SIZE, 1)).long()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, src.shape[1]).bool()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool()
        yield (src, tgt, src_mask, tgt_mask)


# instantiate model

model = XTransformer(dim=512,
                     tie_token_embeds=True,
                     return_tgt_loss=True,
                     enc_num_tokens=NUM_TOKENS,
                     enc_depth=3,
                     enc_heads=8,
                     enc_max_seq_len=ENC_SEQ_LEN,
                     dec_num_tokens=NUM_TOKENS,
                     dec_depth=3,
                     dec_heads=8,
                     dec_max_seq_len=DEC_SEQ_LEN)

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    src, tgt, src_mask, tgt_mask = next(cycle())

    optim.zero_grad()
    loss = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    loss.backward()
    optim.step()

    print(f'{i}: {loss.item()}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        src, tgt, src_mask, _ = next(cycle())
        src, src_mask, tgt = src[:1], src_mask[:1], tgt[:1]
        start_tokens = (torch.ones((1, 1)) * 1).long()

        preds = model.generate(src, start_tokens, DEC_SEQ_LEN, src_mask=src_mask)
        incorrects = (tgt[:, 1:] != preds[:, :-1]).float().abs().sum()

        print("target:  ", tgt[:, 1:])
        print("predicted output:  ", preds[:, :-1])
        print(f"incorrects: {incorrects}")
