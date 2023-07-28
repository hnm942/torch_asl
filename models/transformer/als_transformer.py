import torch
import torch.nn as nn

from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .embedding import LandmarkEmbedding, TokenEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        num_hid=980,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=59,
    ):
        super(Transformer, self).__init__()

        self.source_emb = LandmarkEmbedding(num_hid, source_maxlen)
        self.target_emb = TokenEmbedding(num_classes, target_maxlen, num_hid)
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
        enc_out = self.source_emb(source)
        dec_out = self.target_emb(target)
        for encoder in self.transformer_encoders:
            enc_out = encoder(enc_out)
        for decoder in self.transformer_decoders:
            print(enc_out.shape, dec_out.shape)
            dec_out = decoder(enc_out, dec_out)
        return self.classifier(dec_out)