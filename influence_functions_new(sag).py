from pathlib import Path
import torch
from tqdm import tqdm
from pif.hvp_grad import grad_z
from pif.utils import save_json
from torch.autograd import grad
from collections import OrderedDict
from torch.nn import functional as F, CrossEntropyLoss
import time

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.linalg import norm

def tf_similarity(s1, s2):

    def add_space(s):
        return ' '.join(list(s))

    s1, s2 = add_space(s1), add_space(s2)

    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()

    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

def calc_all_grad(config, model, train_loader, test_loader,
                  ntest_start, ntest_end, mode='TC'):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/

    '''
    depth, r = config['recursion_depth'], config['r_averaging']

    outdir = Path(config["outdir"])

    # breakpoint()
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    influence_results = {}

    ntrainiter = len(train_loader.dataset)
    ntest_end = len(test_loader.dataset)
    model.eval()
    grad_z_test = ()
    # for i in tqdm(range(ntest_start, ntest_end)):
    for i, batch in enumerate(tqdm(test_loader)):

        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
        else:
            device = torch.device("cpu")

        input_ids, token_type_ids, lm_labels = batch
        input_ids, token_type_ids, lm_labels = \
            input_ids.to(device), token_type_ids.to(device), lm_labels.to(device)
        di, dt, dl, idx, dd = test_loader.dataset[i]
        idx = int(idx)

        if outdir.joinpath(f'did-{idx}.{mode}.json').exists():
            continue

        if mode == 'TC':

            # grad_z_test = grad_z(input_ids, token_type_ids, lm_labels, model)
            # outputs = model(
            #     input_ids=input_ids,
            #     token_type_ids=token_type_ids,
            #     labels=lm_labels
            # )
            # loss, logits = outputs[0], outputs[1]
            # 
            # # device = torch.device(f"cuda:0")
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = lm_labels[..., 1:].contiguous().to(device)
            # pad_id = -100
            # loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
            # siz = shift_logits.size(-1)
            # log = shift_logits.view(-1, siz)
            # lab = shift_labels.view(-1)
            # loss = loss_fct(log, lab)
            # 
            # start = time.clock()
            # grad_z_test = grad(loss, model.parameters())  # , allow_unused=True)
            # end1 = time.clock()
            # print(end1-start)
            # grad_z_test = pick_gradient(grad_z_test, model)  # pick effective ones
            # end2 = time.clock()
            # print(end2-end1)

        if mode == 'IF':
            s_test = torch.load(config['stest_path'] + f"/did-{int(idx)}_recdep{depth}_r{r}.s_test")
            s_test = [s_t.cuda() for s_t in s_test]

        train_influences = {}

        for j, batch_t in enumerate(train_loader):  # in tqdm(range(ntrainiter)):

            input_ids_t, token_type_ids_t, lm_labels_t = batch_t
            input_ids_t, token_type_ids_t, lm_labels_t = \
                input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)
            ti, tt, tl, t_idx, td = train_loader.dataset[j]
            t_idx = int(t_idx)
            outputs = model(
                input_ids=input_ids_t,
                token_type_ids=token_type_ids_t,
                labels=lm_labels_t
            )
            loss_t, logits = outputs[0], outputs[1]
            # grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
            # grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
            # grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones

            score = 0
            if mode == 'IF':
                score = param_vec_dot_product(s_test, grad_z_train)
            elif mode == 'TC':
                # score = param_vec_dot_product(grad_z_test, grad_z_train)
                score = tf_similarity(di, ti)

            # breakpoint()

            if t_idx not in train_influences:
                train_influences[t_idx] = {'train_dat': (td),
                                          'if': float(score)}

        # train_influences1 = {}
        # train_influences1 = OrderedDict(sorted(train_influences, key=lambda x: x[1]['if'], reverse=True))#, reverse=True
        if idx not in influence_results:
            influence_results[idx] = {'test_dat': (dd),
                                      'ifs': train_influences}
        save_json(influence_results, outdir.joinpath(f'did-{idx}.{mode}.json'))


def param_vec_dot_product(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])
    # breakpoint()


def pick_gradient(grads, model):
    """
    pick the gradients by name.
    Specifically for BERTs, it extracts 10, 11 layer, pooler and classification layers params.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]
            # if 'layer.10.' in n or 'layer.11.' in n
            # or 'classifier.' in n or 'pooler.' in n



