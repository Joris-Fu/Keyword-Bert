import torch.nn as nn
import torch,copy
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertLayer,BertPooler

class CustomDense(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2,hidden_size)
        self.activation = nn.Tanh()
    def forward(self,input):
        out = self.dense(input)
        out = self.activation(out)
        return out

class BertWithKeywordForSequenceClassification(BertPreTrainedModel):
    def __init__(self,config=None,num_labels=None):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        self.bert_layer = BertLayer(config)
        self.pooler_dense = CustomDense(config.hidden_size)
        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                keyword_mask=None,
                sent_type_mask=None,
                real_mask_a=None,
                real_mask_b=None,
                position_ids=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=True,
                ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        second_last_hidden_state = outputs.hidden_states[-2]
        keyword_hidden_state = self.bert_layer(second_last_hidden_state,attention_mask=keyword_mask.unsqueeze(1).unsqueeze(1))[0]
        real_mask_a = torch.reshape(real_mask_a, [batch_size, seq_length, 1]).float()
        real_mask_b = torch.reshape(real_mask_b, [batch_size, seq_length, 1]).float()
        real_len_a = torch.sum(real_mask_a, dim=1).float()  # [B, 1]
        real_len_b = torch.sum(real_mask_b, dim=1).float()  # [B, 1]
        seq_output = keyword_hidden_state
        rep_a = torch.sum(seq_output * real_mask_a, dim=1)  # [B, H]
        rep_a = torch.divide(rep_a, real_len_a)
        rep_b = torch.sum(seq_output * real_mask_b, dim=1)  # [B, H]
        rep_b = torch.divide(rep_b, real_len_b)
        # fusion_rep = [rep_a, rep_b, rep_a-rep_b, rep_b-rep_a]
        fusion_rep = [rep_a, rep_b]
        final_rep = torch.cat(fusion_rep, -1)
        kw_pooled_output = self.pooler_dense(final_rep)
        kw_pooled_output = self.dropout(kw_pooled_output)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        pooled_output_together = torch.cat([pooled_output,kw_pooled_output],-1)
        logits = self.classifier(pooled_output_together)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

