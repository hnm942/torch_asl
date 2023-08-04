import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, device, dropout_rate=0.1):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.self_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first= True
        )
        self.enc_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first= True
        )
        self.self_dropout = nn.Dropout(p=dropout_rate)
        self.enc_dropout = nn.Dropout(p=dropout_rate)
        self.ffn_dropout = nn.Dropout(p=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = torch.arange(n_dest, device = self.device)[:, None]
        j = torch.arange(n_src, device = self.device)
        mask = i >= j - n_src + n_dest
        mask = mask.to(dtype)
        # print(dtype)
        mask = mask.view(1, n_dest, n_src)
        mult = torch.tensor([batch_size, 1, 1], dtype=int, device = self.device)
        return mask.repeat(*mult)

    def forward(self, enc_out, target):
        input_shape = target.size()
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size * self.num_heads, seq_len, seq_len, target.dtype)
        # causal_mask = causal_mask.to(self.device)
        # print("[decode] causal_mask: ",causal_mask.shape)
        target_att, _ = self.self_att(target, target, target, attn_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out, _ = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(enc_out + self.enc_dropout(enc_out))
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


