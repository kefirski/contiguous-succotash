import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Perplexity(nn.Module):
    def __init__(self):
        super(Perplexity, self).__init__()

    def forward(self, logits, target):
        """
        :param logits: tensor with shape of [batch_size, seq_len, input_size]
        :param target: tensor with shape of [batch_size, seq_len] of Long type filled with indexes to gather from logits
        :return: tensor with shape of [batch_size] with perplexity evaluation
        """

        [batch_size, seq_len, input_size] = logits.size()

        logits = logits.view(-1, input_size)
        log_probs = F.log_softmax(logits)
        del logits

        log_probs = log_probs.view(batch_size, seq_len, input_size)
        target = target.unsqueeze(2)

        out = t.gather(log_probs, dim=2, index=target).squeeze(2).neg()

        ppl = out.mean(1).exp()

        return ppl