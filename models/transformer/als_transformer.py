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
        learning_rate = 0.0001,
        device = None
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def encoder(self, source, source_mask = None):
        enc_out = self.source_emb(source)
        for encoder in self.transformer_encoders:
            enc_out = encoder(enc_out, mask = source_mask)
        return enc_out

    def decoder(self, enc_out, target):
        dec_out = self.target_emb(target)
        # print(dec_out.shape)
        for decoder in self.transformer_decoders:
            # print(enc_out.shape, dec_out.shape)
            dec_out = decoder(enc_out, dec_out)
        return dec_out

    def forward(self, source, target, source_mask = None):
        enc_out = self.encoder(source, source_mask = source_mask)
        dec_out = self.decoder(enc_out, target)
        return self.classifier(dec_out)
    
    def training_step(self, batch):
        """Processes one batch inside model.fit()."""
        landmark_input, phrase = batch 
        landmark, landmark_mask = landmark_input['inputs_embeds'], landmark_input["attention_mask"]
        dec_input = phrase[:, :-1]
        dec_target = phrase[:, 1:]
        preds = self(landmark, dec_input, source_mask = landmark_mask)
        mask = dec_target != 0
        loss = self.loss_metric(preds[mask], dec_target[mask])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, batch):
        landmark_input, phrase = batch 
        landmark, landmark_mask = landmark_input['inputs_embeds'], landmark_input["attention_mask"]
        dec_input = phrase[:, :-1]
        dec_target = phrase[:, 1:]
        preds = self(landmark, dec_input, source_mask = landmark_mask)
        mask = dec_target != 0
        loss = self.loss_metric(preds[mask], dec_target[mask])
        return {"loss": loss.item()}


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
            logits = torch.argmax(logits, dim = 1)
            print(logits.shape)
            last_logit = logits[:, -1][:, None]
            print(last_logit.shape)
            dec_logits.append(last_logit)
            # print(dec_logits)
            dec_input = torch.cat([dec_input, last_logit], dim=-1)
        print(dec_input.shape)
        return dec_input
