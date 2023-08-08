import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab, maxlen, num_hid, device):
        self.device = device
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.emb(x)
        positions = torch.arange(seq_len, dtype =  int, device=self.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        return x + positions

class LandmarkEmbedding(nn.Module):
    def __init__(self, embed_dim, maxlen, device):
        super(LandmarkEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=11, padding=5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        return x