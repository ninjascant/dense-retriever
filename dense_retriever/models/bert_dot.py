import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertEmbedModel(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        mean_pool = torch.mean(last_hidden_state, 1)

        return torch.tensor(0), mean_pool


class BertDotBCEModel(nn.Module):
    def __init__(self, model_name, in_batch_neg=False):
        super(BertDotBCEModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        self.in_batch_neg = in_batch_neg

    def get_embed(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        mean_pool = torch.mean(last_hidden_state, 1)
        return mean_pool

    def forward(self, context_input_ids, query_input_ids, context_attention_mask, query_attention_mask, labels):
        query_embed = self.get_embed(query_input_ids, query_attention_mask)
        doc_embed = self.get_embed(context_input_ids, context_attention_mask)

        if self.in_batch_neg:
            logits = torch.mm(query_embed, doc_embed.T)
            logits = torch.flatten(logits)
            if labels is not None:
                new_labels = np.zeros(logits.size)
                print(logits.size())
                np.fill_diagonal(new_labels, labels)
                new_labels = torch.flatten(torch.tensor(new_labels)).float()

                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, new_labels)

                return SequenceClassifierOutput(logits=logits, loss=loss)
            else:
                return SequenceClassifierOutput(logits=logits)
        else:
            logits = torch.bmm(
                doc_embed.unsqueeze(1),
                query_embed.unsqueeze(2)).squeeze(-1).squeeze(-1)

            if labels is not None:
                loss_fn = nn.BCEWithLogitsLoss()
                labels = labels.float()
                loss = loss_fn(logits, labels)
                return SequenceClassifierOutput(logits=logits, loss=loss)
            else:
                return SequenceClassifierOutput(logits=logits)


class BertDotPairwiseRankingModel(nn.Module):
    def __init__(self, model_name, in_batch_neg=False):
        super(BertDotPairwiseRankingModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        if in_batch_neg:
            raise NotImplementedError('In-batch negative training not implemented yet for BertDotPairwiseRankingModel')

    def get_embed(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        mean_pool = torch.mean(last_hidden_state, 1)
        return mean_pool

    def forward(self, context_input_ids, query_input_ids, context_attention_mask, query_attention_mask, labels):
        query_embed = self.get_embed(query_input_ids, query_attention_mask)
        ctx_embed = self.get_embed(context_input_ids, context_attention_mask)

        distance = torch.bmm(
            ctx_embed.unsqueeze(1),
            query_embed.unsqueeze(2)).squeeze(-1).squeeze(-1)

        if labels is not None:
            labels = torch.where(labels != 0, labels, -1)
            loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
            loss = loss_fn(query_embed, ctx_embed, labels)
            return SequenceClassifierOutput(logits=distance, loss=loss)
        else:
            return SequenceClassifierOutput(logits=distance)


class BertClsModel(nn.Module):
    def __init__(self, model_name_or_path, **kwargs):
        super(BertClsModel, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name_or_path)

        self.linear = nn.Linear(312 * 2, 312)

    def get_embed(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        mean_pool = torch.mean(last_hidden_state, 1)
        return mean_pool

    def forward(self, context_input_ids, query_input_ids, context_attention_mask, query_attention_mask, labels):
        query_embed = self.get_embed(query_input_ids, query_attention_mask)
        ctx_embed = self.get_embed(context_input_ids, context_attention_mask)

        embed_cat = torch.cat((query_embed, ctx_embed), dim=1)

        logits = self.linear(embed_cat)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fn(logits, labels)
            return SequenceClassifierOutput(logits=logits, loss=loss)
        else:
            return SequenceClassifierOutput(logits=logits)
