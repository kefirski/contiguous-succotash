import argparse
import os

import numpy as np
import torch as t
from torch.optim import Adam

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae_dilated import RVAE_dilated

if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE_dilated')
    parser.add_argument('--num-iterations', type=int, default=1000, metavar='NI',
                        help='num iterations (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ppl-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')git branch --set-upstream-to origin/master master')
    parser.add_argument('--nll-result', default='', metavar='nll',
                        help='ce result path (default: '')')
    parser.add_argument('--regularised', type=bool, default=False, metavar='REG',
                        help='Use holistic regularisation in the encoder (default: False)')
    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)


    # defaults parameters are in the parameters.py module
    rvae = RVAE_dilated(parameters, args.regularised)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate, eps=1e-6, weight_decay=1e-5, betas=(0.5,0.9))

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ppl_result_train, kld_result_train, loss_result_train = [], [], []
    ppl_result_val, kld_result_val, loss_result_val = [], [], []

    for iteration in range(args.num_iterations):

        ppl, kld, cross_entropy = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)
        batch_size = args.batch_size
        ppl = ppl.item()
        kld = kld.item()
        cross_entropy = cross_entropy.item()
        if iteration % 10 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('---------PERPLEXITY-----------')
            print(ppl)
            print('-------------KLD--------------')
            print(kld / batch_size)
            print('-------------NLL--------------')
            print((cross_entropy + kld) / batch_size)
            print('------------------------------')
            ppl_result_train.append(ppl)
            kld_result_train.append(kld / batch_size)
            loss_result_train.append((cross_entropy + kld) / batch_size)

        if iteration % 10 == 0:
            ppl, kld, cross_entropy = validate(args.batch_size, args.use_cuda)

            ppl = ppl.item()
            kld = kld.item()
            cross_entropy = cross_entropy.item()

            print('\n')
            print('------------VALID-------------')
            print('---------PERPLEXITY-----------')
            print(ppl)
            print('-------------KLD--------------')
            print(kld / batch_size)
            print('-------------NLL--------------')
            print((cross_entropy + kld) / batch_size)
            print('------------------------------')

            ppl_result_val.append(ppl)
            kld_result_val.append(kld / batch_size)
            loss_result_val.append((cross_entropy + kld) / batch_size)

        if iteration % 20 == 0:
            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            sample = rvae.sample(batch_loader, 50, seed, args.use_cuda)

            print('\n')
            print('------------SAMPLE------------')
            print(sample)
            print('------------------------------')

    t.save(rvae.state_dict(), 'trained_RVAE')

    np.save('ppl_result_train.npy'.format(args.ppl_result), np.array(ppl_result_train))
    np.save('kld_result_train.npy'.format(args.kld_result), np.array(kld_result_train))
    np.save('nll_result_train.npy'.format(args.nll_result), np.array(loss_result_train))
    np.save('ppl_result_val.npy'.format(args.ppl_result), np.array(ppl_result_val))
    np.save('kld_result_val.npy'.format(args.kld_result), np.array(kld_result_val))
    np.save('nll_result_val.npy'.format(args.nll_result), np.array(loss_result_val))
