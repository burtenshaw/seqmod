# from visdom import Visdom

import logging
import numpy as np

import os
from datetime import datetime
import csv

import sys
sys.path.append('/home/burtenshaw/code/IMDB/seqmod/seqmod/misc/')
import nlp
from translate import translate

class Logger(object):

    def log(self, event, payload, verbose=True):
        if verbose and hasattr(self, event):
            getattr(self, event)(payload)


def loss_str(loss, phase):
    return "; ".join([phase + " %s: %g" % (k, v) for (k, v) in loss.items()])


class StdLogger(Logger):
    """
    Standard python logger.

    Parameters:
    ===========
    - outputfile: str, file to print log to. If None, only a console
        logger will be used.
    - level: str, one of 'INFO', 'DEBUG', ... See logging.
    - msgfmt: str, message formattter
    - datefmt: str, date formatter
    """

    def __init__(self, outputfile=None,
                 level='INFO',
                 msgfmt="[%(asctime)s] %(message)s",
                 datefmt='%m-%d %H:%M:%S'):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.handlers = []
        self.logger.setLevel(getattr(logging, level))
        formatter = logging.Formatter(msgfmt, datefmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if outputfile is not None:
            fh = logging.FileHandler(outputfile)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def epoch_begin(self, payload):
        self.logger.info("Starting epoch [%d]" % payload['epoch'])

    def epoch_end(self, payload):
        speed = payload["examples"] / payload["duration"]
        loss = loss_str(payload['loss'], 'train')
        self.logger.info("Epoch [%d]; %s; speed: %d tokens/sec" %
                         (payload['epoch'], loss, speed))

    def validation_end(self, payload):
        loss = loss_str(payload['loss'], 'valid')
        self.logger.info("Epoch [%d]; %s" % (payload['epoch'], loss))

    def test_begin(self, payload):
        self.logger.info("Testing...")

    def test_end(self, payload):
        self.logger.info(loss_str(payload['loss'], 'Test'))

    def checkpoint(self, payload):
        e, b, bs = payload['epoch'], payload['batch'], payload['total_batches']
        speed = payload["examples"] / payload["duration"]
        loss = loss_str(payload['loss'], 'train')
        self.logger.info("Epoch[%d]; batch [%d/%d]; %s; speed %d tokens/sec" %
                         (e, b, bs, loss, speed))

    def info(self, payload):
        if isinstance(payload, dict):
            payload = payload['message']
        self.logger.info(payload)


class VisdomLogger(Logger):
    """
    Logger that uses visdom to create learning curves

    Parameters:
    ===========
    - env: str, name of the visdom environment
    - log_checkpoints: bool, whether to use checkpoints or epoch averages
        for training loss
    - legend: tuple, names of the different losses that will be plotted.
    """

    def __init__(self,
                 env=None,
                 log_checkpoints=True,
                 losses=('loss', ),
                 phases=('train', 'valid'),
                 server='http://localhost',
                 port=8097,
                 max_y=None,
                 **opts):

        self.viz = Visdom(server=server, port=port, env=env)
        self.legend = ['%s.%s' % (p, l) for p in phases for l in losses]
        opts.update({'legend': self.legend})
        self.opts = opts
        self.env = env
        self.max_y = max_y
        self.pane = self._init_pane()
        self.losses = set(losses)
        self.log_checkpoints = log_checkpoints
        self.last = {p: {l: None for l in losses} for p in phases}

    def _init_pane(self):
        nan = np.array([np.NAN, np.NAN])
        X = np.column_stack([nan] * len(self.legend))
        Y = np.column_stack([nan] * len(self.legend))
        return self.viz.line(
            X=X, Y=Y, env=self.env, opts=self.opts)

    def _update_last(self, epoch, loss, phase, loss_label):
        self.last[phase][loss_label] = {'X': epoch, 'Y': loss}

    def _line(self, X, Y, phase, loss_label):
        name = "%s.%s" % (phase, loss_label)
        X = np.array([self.last[phase][loss_label]['X'], X])
        Y = np.array([self.last[phase][loss_label]['Y'], Y])
        if self.max_y:
            Y = np.clip(Y, Y.min(), self.max_y)
        self.viz.updateTrace(
            X=X, Y=Y, name=name, append=True, win=self.pane, env=self.env)

    def _plot_payload(self, epoch, losses, phase):
        for label, loss in losses.items():
            if label not in self.losses:
                continue
            if self.last[phase][label] is not None:
                self._line(epoch, loss, phase=phase, loss_label=label)
            self._update_last(epoch, loss, phase, label)

    def epoch_end(self, payload):
        if self.log_checkpoints:
            # only use epoch end if checkpoint isn't being used
            return
        losses, epoch = payload['loss'], payload['epoch'] + 1
        self._plot_payload(epoch, losses, 'train')

    def validation_end(self, payload):
        losses, epoch = payload['loss'], payload['epoch'] + 1
        self._plot_payload(epoch, losses, 'valid')

    def checkpoint(self, payload):
        epoch = payload['epoch'] + payload["batch"] / payload["total_batches"]
        losses = payload['loss']
        self._plot_payload(epoch, losses, 'train')

    def attention(self, payload):
        title = "epoch {epoch}/ batch {batch_num}".format(**payload)
        if 'title' in self.opts:
            title = self.opts['title'] + ": " + title
        self.viz.heatmap(
            X=np.array(payload["att"]),
            env=self.env,
            opts={'rownames': payload["hyp"],
                  'columnnames': payload["target"],
                  'title': title})


class notesLogger(Logger):
    """
    Standard python logger.

    Parameters:
    ===========
    - outputfile: str, file to print log to. If None, only a console
        logger will be used.
    - level: str, one of 'INFO', 'DEBUG', ... See logging.
    - msgfmt: str, message formattter
    - datefmt: str, date formatter
    """

    def __init__(self, outputfile=None,
                 level='INFO',
                 msgfmt="[%(asctime)s] %(message)s",
                 datefmt='%m-%d %H:%M:%S',
                 args=None,
                 save_path="/home/burtenshaw/logging/testing_logger.csv",
                 model=None):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.handlers = []
        self.logger.setLevel(getattr(logging, level))
        self.args = args
        self.save_path = save_path
        self.target = args.target
        self.model = model
        formatter = logging.Formatter(msgfmt, datefmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if outputfile is not None:
            fh = logging.FileHandler(outputfile)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def modelLogger(self):
        '''
        Save a record of the pytorch model being used.
        from 'seqmod.modules.encoder_decoder.EncoderDecoder'
        '''
        # TODO add indexing.
        opts = {k: v for k, v in vars(self.args).items()}
        
        head, args = [], []
        
        for i,k in opts.items():
            head.append(i)
            args.append(k)

        return head, args

    def store(self, save_path, payload):
        '''
        '''
        epoch_info = self.payloadWrapper(payload)
        
        if payload['epoch'] == 1:
            # Get the model header and entries from arsparse
            head, args = self.modelLogger()

            # Epoch info
            #Save it as a line in the CSV
            epoch_header = ['Date','Epoch', "Loss", "Smith Waterman", "Levenstein", "Bleu", "Generated String"]

            print("Saving to: ", save_path)

            # Check if we're reusing a file.
            if os.path.isfile(save_path):
                m = 'a'
            else:
                m = 'w'

            # Write a new line of model info
            with open(save_path, m) as f:
                writer = csv.writer(f)
                writer.writerows([head, args])
                writer.writerows([epoch_header, epoch_info])
                f.close
        else:
            print("Saving to: ", save_path)
            with open(save_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(epoch_info)

    def payloadWrapper(self, payload):
        '''
        Wrap up the 'per-event' data
        '''
        # Performance Metrics
        date = str(datetime.now())
        e = payload['epoch']
        l = payload['loss']['loss']

        # Language Metrics
        target_string = self.target
        gen = translate(self.model, self.target, self.args.gpu)

        # Smith Waterman
        sw = nlp.smith_waterman(target_string, gen)
        # Levenstein
        lv = nlp.levenstein(target_string, gen)
        # Bleu
        bleu = nlp.bleu(target_string, gen)

        wrapped = [date, e, l, sw, lv, bleu, gen]
        return wrapped

    def epoch_begin(self, payload):
        self.logger.info("Starting epoch [%d]" % payload['epoch'])

    def epoch_end(self, payload):
        speed = payload["examples"] / payload["duration"]
        loss = loss_str(payload['loss'], 'train')
        self.logger.info("Epoch [%d]; %s; speed: %d tokens/sec" %
                         (payload['epoch'], loss, speed))

    def validation_end(self, payload):
        loss = loss_str(payload['loss'], 'valid')
        self.logger.info("Epoch [%d]; %s" % (payload['epoch'], loss))
        self.store(self.save_path, payload)

    def test_begin(self, payload):
        self.logger.info("Testing...")

    def test_end(self, payload):
        self.logger.info(loss_str(payload['loss'], 'Test'))

    def checkpoint(self, payload):
        e, b, bs = payload['epoch'], payload['batch'], payload['total_batches']
        speed = payload["examples"] / payload["duration"]
        loss = loss_str(payload['loss'], 'train')
        self.logger.info("Epoch[%d]; batch [%d/%d]; %s; speed %d tokens/sec" %
                         (e, b, bs, loss, speed))

    def info(self, payload):
        if isinstance(payload, dict):
            payload = payload['message']
        self.logger.info(payload)


# payload = { 'epoch': 2, 'batch':2, 'speed':3, 'loss':
#    {'':0.7}, 'examples':5, 'duration':6}

# notesLogger.store(directory, payload)
# # hypothesis = 'It is a cat at room'
# # reference = 'It is a cat inside the room'
# # print(notesLogger.languageMetrics(hypothesis, reference))

# # args_dict = {'path': 'data/faq_0/train', 'target': 'Compare the levenstein distance between the words in a sentence, and a dataset of keywords.', 'pretrained': 'empty', 'dev': 0.1, 'max_size': 40, 'min_freq': 5, 'bidi': True, 'layers': 1, 'cell': 'LSTM', 'emb_dim': 264, 'hid_dim': 128, 'att_dim': 64, 'att_type': 'Bahdanau', 'dropout': 0.2, 'project_init': False, 'batch_size': 64, 'epochs': 20, 'checkpoint': 50, 'hooks_per_epoch': 2, 'optim': 'RMSprop', 'plot': False, 'learning_rate': 0.01, 'learning_rate_decay': 0.5, 'start_decay_at': 8, 'max_grad_norm': 5.0, 'beam': False, 'gpu': True, 'logging': True, 'visdom': False}
# # {'path': 'data/faq_0/train', 'target': 'Compare the levenstein distance between the words in a sentence, and a dataset of keywords.', 'pretrained': 'empty', 'dev': 0.1, 'max_size': 40, 'min_freq': 5, 'bidi': True, 'layers': 1, 'cell': 'LSTM', 'emb_dim': 264, 'hid_dim': 128, 'att_dim': 64, 'att_type': 'Bahdanau', 'dropout': 0.2, 'project_init': False, 'batch_size': 64, 'epochs': 20, 'checkpoint': 50, 'hooks_per_epoch': 2, 'optim': 'RMSprop', 'plot': False, 'learning_rate': 0.01, 'learning_rate_decay': 0.5, 'start_decay_at': 8, 'max_grad_norm': 5.0, 'beam': False, 'gpu': True, 'logging': True, 'visdom': False}



# # head, args = notesLogger.modelLogger(args_dict)
# # print(len(head), len(args))
# # print(head, args)

