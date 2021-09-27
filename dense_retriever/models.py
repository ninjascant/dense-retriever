import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, AutoModel, PreTrainedModel, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return loss.mean(),


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class BertDot(nn.Module):
    def __init__(self, model_name):
        super(BertDot, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.8)

    def get_embed(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        mean_pool = torch.mean(last_hidden_state, 1)
        return mean_pool

    def forward(self, doc_input_ids, query_input_ids, doc_attention_mask, query_attention_mask, labels):
        query_embed = self.get_embed(query_input_ids, query_attention_mask)
        doc_embed = self.get_embed(doc_input_ids, doc_attention_mask)

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


class IRBert(PreTrainedModel):
    def __init__(self, config):
        super(IRBert, self).__init__(config)
        self.transformer = AutoModel.from_pretrained(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        mean_pool = torch.mean(last_hidden_state, 1)
        return mean_pool


models = {
    'ance': RobertaDot_NLL_LN,
    'tinybert': IRBert
}


def load_model(model_name, model_path):
    if model_name == 'ance':
        model = RobertaDot_NLL_LN.from_pretrained(
            model_path
        )
    elif model_name == 'tinybert':
        model = BertDotEmbed(model_path)
    else:
        raise NotImplementedError
    return model
