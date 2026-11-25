import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel
from .layers import TransformerEncoder

r"""
GRU4Rec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

"""

class SASRec(AbstractBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.type = ["sequential"]
        self.IR_type = ["retrieval", "ranking"]

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["embedding_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = float(config["layer_norm_eps"])

        self.initializer_range = config["initializer_range"]

        self.history_length = config['history_length']
        self.history_padding_id = config.get('history_padding_id')
        if self.history_padding_id is not None:
            if self.history_padding_id != config['item_num']:
                raise ValueError('SASRec PAD must be outside the real item catalog')
            self.item_embedding = nn.Embedding(
                config['sequence_item_embedding_num'], self.hidden_size,
                padding_idx=self.history_padding_id)

        self.position_embedding = nn.Embedding(self.history_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.apply(self._init_weights)
        if self.history_padding_id is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.history_padding_id].zero_()


    # def forward(self, user_ids, item_ids):
    #     user_embeds = self.user_embedding(user_ids)
    #     item_embeds = self.item_embedding(item_ids)
    #     dot_product = (user_embeds * item_embeds).sum(1)
    #     return self.sigmoid(dot_product)


    def compute_loss(self, interaction):

        #user = interaction['user_ids']
        #history_ids = interaction['history_ids']
        pos_items = interaction['item_ids']
        user_emb = self.get_user_embedding(interaction)
        test_item_emb = self.item_embedding.weight[:self.config['item_num']]
        logits = torch.matmul(user_emb, test_item_emb.transpose(0, 1))
        loss = self.loss(logits, pos_items)
        return loss

    def get_user_embedding(self, user_dict):
        users = user_dict['history_ids']
        if self.history_padding_id is not None and 'history_mask' not in user_dict:
            raise ValueError('Padded SASRec histories require history_mask')
        valid = user_dict.get('history_mask', torch.ones_like(users)).bool()
        if valid.shape != users.shape:
            raise ValueError('history_mask and history_ids must have matching shapes')
        item_seq_emb = self.item_embedding(users)
        ids = torch.arange(
            self.history_length, dtype=torch.long, device=users.device
        )
        position_ids = ids.repeat(users.size(0), 1)
        position_embedding = self.position_embedding(position_ids)
        input_emb = item_seq_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Keys must be real history items at or before the query position.
        # A finite additive mask keeps left-padded query rows numerically stable;
        # these rows can never become keys for a real query.
        causal = torch.ones((users.size(1), users.size(1)),
                            device=users.device, dtype=torch.bool).tril()
        allowed = valid[:, None, None, :] & causal[None, None, :, :]
        attention_mask = torch.zeros_like(allowed, dtype=input_emb.dtype)
        attention_mask.masked_fill_(~allowed, -10000.0)
        input_emb = input_emb * valid.unsqueeze(-1)

        trm_output = self.trm_encoder(
            input_emb, attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = output[:,-1,:]
        #seq_output = self.gather_indexes(gru_output, self.config['history_length'] - 1)
        return seq_output



    def forward(self, user_dict, item_id):
        ###here user_id denotes the historical ids
        user_embeds = self.get_user_embedding(user_dict)
        item_embeds = self.item_embedding(item_id)
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)

