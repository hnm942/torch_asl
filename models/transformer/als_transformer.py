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
        super().__init__()
        self.loss_metric = nn.MSELoss()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = nn.Sequential(
            self.enc_input,
            *[
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = nn.Linear(num_hid, num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def forward(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    def training_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch[0]
        target = batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = torch.nn.functional.one_hot(dec_target, num_classes=self.num_classes)
        mask = dec_target != 0
        loss = self.loss_metric(preds, one_hot.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_metric.update_state(loss.item())
        return {"loss": self.loss_metric.result()}

    def validation_step(self, batch):
        source = batch[0]
        target = batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = torch.nn.functional.one_hot(dec_target, num_classes=self.num_classes)
        mask = dec_target != 0
        loss = self.loss_metric(preds, one_hot.float())
        self.loss_metric.update_state(loss.item())
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = source.size(0)
        enc = self.encoder(source)
        dec_input = torch.full((bs, 1), target_start_token_idx, dtype=torch.long, device=source.device)
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = torch.argmax(logits, dim=-1, keepdim=True)
            last_logit = logits[:, -1]
            dec_logits.append(last_logit.unsqueeze(1))
            dec_input = torch.cat([dec_input, last_logit], dim=-1)
        return dec_input
