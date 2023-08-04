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
        device = None
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.target_maxlen = target_maxlen
        self.source_emb = LandmarkEmbedding(num_hid, source_maxlen, self.device)
        self.target_emb = TokenEmbedding(num_classes, target_maxlen, num_hid, self.device)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(num_hid, num_head, num_feed_forward, self.device)
            for _ in range(num_layers_enc)
        ])
        self.transformer_decoders = nn.ModuleList([
            TransformerDecoder(num_hid, num_head, num_feed_forward, self.device)
            for _ in range(num_layers_dec)
        ])
        self.classifier = nn.Linear(num_hid, num_classes)
        self.loss_metric = nn.CrossEntropyLoss()  

    def encoder(self, source):
        enc_out = self.source_emb(source)
        for encoder in self.transformer_encoders:
            enc_out = encoder(enc_out)
        return enc_out

    def decoder(self, enc_out, target):
        dec_out = self.target_emb(target)
        # print(dec_out.shape)
        for decoder in self.transformer_decoders:
            # print(enc_out.shape, dec_out.shape)
            dec_out = decoder(enc_out, dec_out)
        return dec_out

    def forward(self, source, target):
        enc_out = self.encoder(source)
        dec_out = self.decoder(enc_out, target)
        return self.classifier(dec_out)
    
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

    def inference(self, landmarks, start_token_idx = 2):
        enc_out = self.encoder(landmarks)
        # export outout
        bs = landmarks.shape[0] # batch size
        # decoder input
        dec_input = torch.ones((bs, 1), dtype=torch.int32) * start_token_idx
        dec_input = dec_input.to(self.device)
        dec_logits = []
        for i in range(self.target_maxlen):
            print(enc_out.shape)
            dec_out = self.decoder(enc_out, dec_input)
            logits = self.classifier(dec_out)
            logits = torch.argmax(logits, axis=-1, output_type=torch.int32)
            last_logit = logits[:, -1][:, None]
            dec_logits.append(last_logit)
            dec_input = torch.cat([dec_input, last_logit], dim=-1)
        return dec_input
