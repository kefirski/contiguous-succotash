import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from utils.functional import parameters_allocation_check


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params

        self.kernels = [Parameter(t.Tensor(out_chan, in_chan, width).normal_(0, 0.05))
                        for out_chan, in_chan, width in params.decoder_kernels]
        self._add_to_parameters(self.kernels, 'decoder_kernel')

        self.biases = [Parameter(t.Tensor(out_chan).normal_(0, 0.05))
                       for out_chan, in_chan, width in params.decoder_kernels]
        self._add_to_parameters(self.biases, 'decoder_bias')

        self.out_size = self.params.decoder_kernels[-1][0]

        self.fc = nn.Linear(self.out_size, self.params.word_vocab_size)

    def forward(self, decoder_input, z, drop_prob):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence latent variable with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :return: unnormalized logits of sentense words distribution probabilities
                 with shape of [batch_size, seq_len, word_vocab_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        # x is tensor with shape [batch_size, input_size=in_channels, seq_len=input_width]
        x = decoder_input.transpose(1, 2).contiguous()

        for layer, kernel in enumerate(self.kernels):

            # apply conv layer with non-linearity and drop last elements of sequence to perfrom input shifting
            x = F.conv1d(x, kernel,
                         bias=self.biases[layer],
                         dilation=self.params.decoder_dilations[layer],
                         padding=self.params.decoder_paddings[layer])

            x = F.relu(x)

            x_width = x.size()[2]
            x = x[:, :, :(x_width - self.params.decoder_paddings[layer])].contiguous()

        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, self.out_size)
        x = self.fc(x)
        result = x.view(-1, seq_len, self.params.word_vocab_size)

        return result

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)
