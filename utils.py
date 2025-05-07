# === utils.py ===

import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    # Prevent attending to future tokens
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

# Usage:
# tgt_mask = generate_square_subsequent_mask(seq_len)  # [seq_len, seq_len]