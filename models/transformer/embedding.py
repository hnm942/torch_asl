import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab, maxlen, num_hid):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.emb(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        return x + positions

class LandmarkEmbedding(nn.Module):
    def __init__(self, num_hid, maxlen):
        super(LandmarkEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(num_hid, num_hid, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(num_hid, num_hid, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(num_hid, num_hid, kernel_size=11, padding=5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        return x
