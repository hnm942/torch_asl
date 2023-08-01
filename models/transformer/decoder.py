import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.att1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.att2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, enc_out, target):
        attn1_output, _ = self.att1(target, target, target)
        attn1_output = self.dropout1(attn1_output)
        out1 = self.layernorm1(target + attn1_output)
        # print("[decoder output] out1: target {}. attn1: {}".format(target.shape, out1.shape))

        attn2_output, _ = self.att2(enc_out, out1, out1)
        attn2_output = self.dropout2(attn2_output)
        out2 = self.layernorm2(out1 + attn2_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output)



