
import os

seed = 1001

import random
random.seed(seed)

import torch
from torchtext.vocab import load_word_vectors
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

from train_encoder_decoder import make_encdec_hook, make_criterion

from seqmod.modules.encoder_decoder import EncoderDecoder
from seqmod import utils as u

from seqmod.misc.dataset import PairedDataset, Dict
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import EncoderDecoderTrainer
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.preprocess import text_processor

'''
A script for generating questions from answers.
'''

def load_data(path, exts, text_processor=text_processor()):
    src_data, trg_data = [], []
    path = os.path.expanduser(path)
    with open(path + exts[0]) as src, open(path + exts[1]) as trg:
        for src_line, trg_line in zip(src, trg):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if text_processor is not None:
                src_line = text_processor(src_line)
                trg_line = text_processor(trg_line)
            if src_line and trg_line:
                src_data.append(src_line), trg_data.append(trg_line)
    return src_data, trg_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True,
                        help='Path to the a directory containing source and target text files')
    parser.add_argument('--target', default=None)
    parser.add_argument('--pretrained', type=str, default='empty')

    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--max_size', default=10000, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--bidi', action='store_false')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=264, type=int)
    parser.add_argument('--hid_dim', default=128, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--project_init', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--hooks_per_epoch', default=2, type=int)
    parser.add_argument('--optim', default='RMSprop', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    # Logging
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--visdom', action='store_true')
    args = parser.parse_args()

    src, trg = load_data(args.path, ('.answers', '.questions'))
    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
                    max_size=args.max_size, min_freq=args.min_freq)
    src_dict.fit(src, trg)
    train, valid = PairedDataset(
        src, trg, {'src': src_dict, 'trg': src_dict},
        batch_size=args.batch_size, gpu=args.gpu
    ).splits(dev=args.dev, test=None, sort_key=lambda pair: len(pair[0]))

    print(' * vocabulary size. %d' % len(src_dict))
    print(' * number of train batches. %d' % len(train))
    print(' * maximum batch size. %d' % args.batch_size)


    print('Building model...')

    model = EncoderDecoder(
        # removed (args.hid_dim, args.hid_dim) added args.hid_dim
        (args.layers, args.layers), args.emb_dim, args.hid_dim,
        args.att_dim, src_dict, att_type=args.att_type, dropout=args.dropout,
        bidi=args.bidi, cell=args.cell)

    # Load Glove Pretrained Embeddings

    if args.pretrained != 'empty':
        wv_dict, wv_arr, wv_size = load_word_vectors(args.pretrained, 'glove.6B', 50)
        print('Loaded', len(wv_arr), 'words from pretrained embeddings.')
        model.emb_dim = wv_size
        wv_list = list(wv_dict)
        model.load_embeddings(wv_arr, wv_list, verbose=False)

    # Optimisation
    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = make_criterion(len(src_dict), src_dict.get_pad())

    model.apply(u.make_initializer(
        rnn={'type': 'orthogonal', 'args': {'gain': 1.0}}))

    print('* number of parameters: %d' % model.n_params())


    print(model)

    if args.gpu:
        model.cuda(), criterion.cuda()

    trainer = EncoderDecoderTrainer(
        model, {'train': train, 'valid': valid}, criterion, optimizer)

    # Logging
    if args.logging:
        trainer.add_loggers(StdLogger())
    if args.visdom:
        trainer.add_loggers(StdLogger(), VisdomLogger(env='encdec'))

    target = args.target.split() if args.target else None
    hook = make_encdec_hook(args.target, args.gpu)
    num_checkpoints = len(train) // (args.checkpoint * args.hooks_per_epoch)
    trainer.add_hook(hook, num_checkpoints=num_checkpoints)

    trainer.train(args.epochs, args.checkpoint, shuffle=True, gpu=args.gpu)