import torch
import torch.nn as nn

from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .embedding import LandmarkEmbedding, TokenEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        super(Transformer, self).__init__()

        self.encoder = LandmarkEmbedding(num_hid, source_maxlen)
        self.decoder = TokenEmbedding(num_classes, target_maxlen, num_hid)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_enc)
        ])
        self.transformer_decoders = nn.ModuleList([
            TransformerDecoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_dec)
        ])
        self.classifier = nn.Linear(num_hid, num_classes)


    def forward(self, source, target):
        enc_out = self.encoder(source)
        dec_out = self.decoder(target)
        enc_out = self.transformer(enc_out)
        dec_out = self.transformer(dec_out)
        return enc_out, dec_out

