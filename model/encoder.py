import torch as t
import torch.nn as nn
import torch.nn.functional as F

from selfModules.highway import Highway
from utils.functional import parameters_allocation_check


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.hw1 = Highway(self.params.sum_depth + self.params.word_embed_size, 2, F.relu)
        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           dropout=0.5,
                           batch_first=True,
                           bidirectional=True)

        self.hw2 = Highway(self.params.encoder_rnn_size * 2, 3, F.relu)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentences with shape of [batch_size, latent_variable_size]
        """

        [batch_size, seq_len, embed_size] = input.size()

        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        '''
        _, (final_state, _) = self.rnn(input)

        final_state = final_state.view(self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)

        final_state = self.hw2(final_state)

        return final_state


class HREncoder(nn.Module):
    """
    Encoder with holistic regularisation: regularisation (Gaussian) is applied at every hidden layer of the
    LSTM encoder
    """
    def __init__(self, params):
        super(HREncoder, self).__init__()

        self.params = params
        self.layer_dim = self.params.encoder_num_layers * 2 * self.params.encoder_rnn_size

        self.rnn = nn.LSTM(input_size=self.params.word_embed_size + self.params.sum_depth,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           dropout=0.5,
                           batch_first=True,
                           bidirectional=True)

        self.linear_mu = nn.Linear(self.layer_dim * 2, self.layer_dim * 2)
        self.linear_var = nn.Linear(self.layer_dim * 2, self.layer_dim * 2)

    def forward(self, input):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        [batch_size, seq_len, embed_size] = input.size()
        #TEST: note sure this is right (was done for giving right dimensions to the lstm)
        input = input.view(seq_len, batch_size, embed_size)
        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        '''

        mux, logvarx = [], []
        hx = None
        for i in range(seq_len - 1):
            _, hx = self.rnn(input[i].unsqueeze(1), hx)
            h = self.ziphidden(*hx)
            mu = self.linear_mu(h)
            logvar = self.linear_var(h)
            h = self.reparameterize(mu, logvar)
            mux.append(mu)
            logvarx.append(logvar)

        return h

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = t.exp(0.5 * logvar)
        eps = t.randn_like(std)
        return mu + eps * std

    def ziphidden(self, hidden, cell):
        b_size = hidden.shape[1]
        h = t.cat([hidden, cell], dim=2)
        h = t.transpose(h, 0, 1).contiguous()
        h = h.view(b_size, -1)
        return h
