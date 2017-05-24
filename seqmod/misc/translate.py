import random                  # nopep8
import torch                    # nopep8
from torch.autograd import Variable
import numpy as np

seed = 1005
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# TODO Get unk and padding from argpars

def translate(model, target, gpu, beam=False):
    print("info", "Translating %s" % target)
    target = torch.LongTensor(
        list(model.src_dict.transform([target], bos=False)))
    batch = Variable(target.t(), volatile=True)
    batch = batch.cuda() if gpu else batch
    if beam:
        scores, hyps, att = model.translate_beam(
            batch, beam_width=5, max_decode_len=4)
    else:
        scores, hyps, att = model.translate(batch, max_decode_len=4)
    hyps = ' '.join([model.trg_dict.vocab[c] for c in hyps[0]])
#    hyps = ' '.join([model.trg_dict.vocab[c] for c in hyps[0] if c not in unk])
    return hyps

