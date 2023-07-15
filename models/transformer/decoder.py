import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, device):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        mask = torch.triu(torch.ones(n_dest, n_src, device=device), diagonal=1)
        return mask.unsqueeze(0).expand(batch_size, -1, -1).bool()

    def forward(self, enc_out, target):
        input_shape = target.size()
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        device = target.device
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, device)
        target_att, _ = self.self_att(target, target, target, attn_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out, _ = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm
