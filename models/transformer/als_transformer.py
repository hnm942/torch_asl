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
        
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen),
            *[TransformerEncoder(embed_dim=num_hid, num_heads=num_head, feed_forward_dim=num_feed_forward) for _ in range(num_layers_enc)]
        )

        self.decoder = nn.Sequential(
            TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid),
            *[TransformerDecoder(embed_dim=num_hid, num_heads=num_head, feed_forward_dim=num_feed_forward) for _ in range(num_layers_dec)]
        )

        self.classifier = nn.Linear(num_hid, num_classes)

    def forward(self, source, target):
        enc_out = self.encoder(source)
        dec_out = self.decoder(target)
        return self.classifier(dec_out)
