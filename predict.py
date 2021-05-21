import argparse
import os

import numpy as np
import torch as t

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae_dilated import RVAE_dilated

if __name__ == '__main__':

    assert os.path.exists('trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Prediction on the test set')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--num-iterations', type=int, default=1000, metavar='NI',
                        help='num iterations (default: 1000)')
    parser.add_argument('--regularised', type=bool, default=False, metavar='REG',
                        help='Use holistic regularisation in the encoder (default: False)')


    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE_dilated(parameters, args.regularised)
    batch_size = args.batch_size
    rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    test = rvae.tester(batch_loader)
    ppl_result_test, kld_result_test, loss_result_test = [], [], []

    for iteration in range(args.num_iterations):
        ppl, kld, cross_entropy = test(args.batch_size, args.use_cuda)
        ppl = ppl.item()
        kld = kld.item()
        cross_entropy = cross_entropy.item()

        print('\n')
        print('---------PERPLEXITY-----------')
        print(ppl)
        print('-------------KLD--------------')
        print(kld / batch_size)
        print('-------------NLL--------------')
        print((cross_entropy + kld) / batch_size)
        print('------------------------------')

        ppl_result_test.append(ppl)
        kld_result_test.append(kld / batch_size)
        loss_result_test.append((cross_entropy + kld) / batch_size)

    np.save('ppl_result_test.npy', np.array(ppl_result_test))
    np.save('kld_result_test.npy', np.array(kld_result_test))
    np.save('nll_result_test.npy', np.array(loss_result_test))