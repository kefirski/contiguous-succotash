import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder, HREncoder

from selfModules.embedding import Embedding
from selfModules.perplexity import Perplexity

from utils.functional import kld_coef, parameters_allocation_check, fold


class RVAE_dilated(nn.Module):
    def __init__(self, params, regularised):
        super(RVAE_dilated, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.regularised = regularised

        if self.regularised:
            print("Highly regularised Encoder")
            self.encoder = HREncoder(self.params)
            self.layer_dim = self.params.encoder_num_layers * 2 * self.params.encoder_rnn_size
            self.context_to_mu = nn.Linear(self.layer_dim * 2, self.params.latent_variable_size)
            self.context_to_logvar = nn.Linear(self.layer_dim * 2, self.params.latent_variable_size)
        elif not self.regularised:
            print('Classic encoder')
            self.encoder = Encoder(self.params)
            self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
            self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_word_input=None, encoder_character_input=None,
                decoder_word_input=None,
                z=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 kld loss estimation
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
                                  [encoder_word_input, encoder_character_input, decoder_word_input],
                                  True) \
               or (z is not None and decoder_word_input is not None), \
            "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        decoder_input = self.embedding.word_embed(decoder_word_input)
        out = self.decoder(decoder_input, z, drop_prob)

        return out, kld

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        perplexity = Perplexity()

        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, _, target] = input

            logits, kld = self(dropout,
                               encoder_word_input, encoder_character_input,
                               decoder_word_input,
                               z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target, ignore_index=0, reduction="sum")

            # since cross entropy is averaged over seq_len, it is necessary to approximate new kld
            # loss = 79 * cross_entropy + kld
            #loss = cross_entropy + kld

            logits = logits.view(batch_size, -1, self.params.word_vocab_size)
            target = target.view(batch_size, -1)
            ppl = perplexity(logits, target).mean()

            optimizer.zero_grad()
            cross_entropy.backward()
            optimizer.step()

            return ppl, kld, cross_entropy

        return train

    def validater(self, batch_loader):
        perplexity = Perplexity()

        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, _, target] = input

            logits, kld = self(0.,
                               encoder_word_input, encoder_character_input,
                               decoder_word_input,
                               z=None)
            ppl = perplexity(logits, target).mean()
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target, ignore_index=0, reduction="sum")
            #loss = cross_entropy + kld

            return ppl, kld, cross_entropy

        return validate

    def tester(self, batch_loader):
        perplexity = Perplexity()

        def test(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'test')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, _, target] = input

            logits, kld = self(0.,
                               encoder_word_input, encoder_character_input,
                               decoder_word_input,
                               z=None)
            ppl = perplexity(logits, target).mean()
            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)
            cross_entropy = F.cross_entropy(logits, target, ignore_index=0, reduction="sum")
            #loss = cross_entropy + kld

            return ppl, kld, cross_entropy

        return test

    def sample(self, batch_loader, seq_len, seed, use_cuda):
        seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, _ = batch_loader.go_input(1)
        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())

        if use_cuda:
            decoder_word_input = decoder_word_input.cuda()

        result = ''

        for i in range(seq_len):
            logits, _ = self(0., None, None,
                             decoder_word_input,
                             seed)

            [_, sl, _] = logits.size()

            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)
            prediction = prediction.view(1, sl, -1)

            # take the last word from prefiction and append it to result
            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[0, -1])

            if word == batch_loader.end_token:
                break

            result += ' ' + word

            word = np.array([[batch_loader.word_to_idx[word]]])

            decoder_word_input_np = np.append(decoder_word_input_np, word, 1)
            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())

            if use_cuda:
                decoder_word_input = decoder_word_input.cuda()

        return result
