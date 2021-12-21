import argparse
import os
import torch
import sys
sys.path.append("..")
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pif.influence_functions_new import calc_all_grad
from pif.utils import init_logging
from src.custom_data import CustomDataset, PadCollate
from torch.utils.data import DataLoader
import numpy as np
import random


def load_model(args, device):

    print("Loading the model...")
    model = GPT2LMHeadModel.from_pretrained(args.model_type).to(device)
    model.resize_token_embeddings(args.vocab_size)
    args.max_len = min(args.max_len, model.config.n_ctx)
    if args.ckpt_name is not None:
        if os.path.exists(f"{args.ckpt_dir}/{args.ckpt_name}.ckpt"):
            print("Loading the trained checkpoint...")
            ckpt = torch.load(f"{args.ckpt_dir}/{args.ckpt_name}.ckpt")
            model.load_state_dict(ckpt['model_state_dict'])
    return model
# python IF.py --data_dir data --model_type gpt2 --train_prefix train --valid_prefix valid --score_out_dir Score --log_file_name logfile --test_delta True --mode TC --ntest_start -1 --ntest_end -1 --ckpt_dir saved_models/gpt2 --ckpt_name best_ckpt_epoch=10_valid_loss=5.2944.ckpt
# python.py --score_out_dir Score --log_file_name logfile --test_delta True --mode TC --ntest_start -1 --ntest_end -1 --ckpt_name data/gpt2/best_ckpt_epoch=10_valid_loss=5.2944.ckpt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default="train", help="The running mode: train or inference?")
    parser.add_argument('--data_dir', type=str, default="data",
                        help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of the train data files' name.")
    parser.add_argument('--valid_prefix', type=str, default="valid",
                        help="The prefix of the validation data files' name.")
    parser.add_argument('--model_type', type=str, default="gpt2", help="The model type of GPT-2.")

    # parser.add_argument("--stest_path", default=None, type=str, required=False, help="The input testing data name")
    parser.add_argument("--score_out_dir", default=None, type=str, required=True,
                        help="specifies the name of model here")
    parser.add_argument("--log_file_name", default="logfile", type=str, required=True, help="The log file name")

    parser.add_argument("--test_delta", default="True", type=str, required=True,
                        help="multiple by delta test (True) or by test (False)")
    parser.add_argument('--mode', type=str, help='the mode of influence function: IF, IF+, TC, TC+')
    parser.add_argument("--ntest_start", default=-1, type=int, required=True, help="num of classes for the model")
    parser.add_argument("--ntest_end", default=-1, type=int, required=True, help="num of classes for the model")

    parser.add_argument('--pad_token', type=str, default="<pad>", help="The pad token.")
    parser.add_argument('--bos_token', type=str, default="<bos>", help="The BOS token.")
    parser.add_argument('--eos_token', type=str, default="<eos>", help="The EOS token.")
    parser.add_argument('--sp1_token', type=str, default="<sp1>", help="The speaker1 token.")
    parser.add_argument('--sp2_token', type=str, default="<sp2>", help="The speaker2 token.")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    # parser.add_argument('--lr', type=float, default=5e-4, help="The learning rate.")
    parser.add_argument('--batch_size', type=int, default=1, help="The batch size.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    # parser.add_argument('--num_epochs', type=int, default=10, help="The number of total epochs.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    # parser.add_argument('--top_p', type=float, default=0.9, help="The top-p value for nucleus sampling decoding.")
    parser.add_argument('--ckpt_dir', type=str, default="saved_models",
                         help="The directory name for saved checkpoints.")
    parser.add_argument('--ckpt_name', type=str, required=False, help="The name of the trained checkpoint. (without extension)")

    args = parser.parse_args()

    args.data_dir = f"{args.data_dir}/{args.model_type}"
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    special_tokens = {
        'bos_token': args.bos_token,
        'eos_token': args.eos_token,
        'pad_token': args.pad_token,
        'additional_special_tokens': [args.sp1_token, args.sp2_token]
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    args.pad_id = vocab[args.pad_token]
    args.bos_id = vocab[args.bos_token]
    args.eos_id = vocab[args.eos_token]
    args.sp1_id = vocab[args.sp1_token]
    args.sp2_id = vocab[args.sp2_token]
    args.utter_len = (args.max_len - args.max_turns - 2) // args.max_turns #

    train_set = CustomDataset(args.train_prefix, args)
    valid_set = CustomDataset(args.valid_prefix, args)
    ppd = PadCollate(pad_id=args.pad_id)

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    # fix_seed(args.seed)

    train_loader = DataLoader(train_set,
                              collate_fn=ppd.pad_collate,
                              # shuffle=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_set,
                              collate_fn=ppd.pad_collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    config = {
        "outdir": args.score_out_dir,
        # "stest_path": args.stest_path,
        "seed": 42,
        "gpu": 0,
        "recursion_depth": 1000,  # set recursion to use entire training data
        "r_averaging": 1,
        "scale": 1000,
        "damp": 0.01,
        "num_classes": 3,
        "log_filename": args.log_file_name
    }



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args, device)
    model.eval()

    init_logging(config["log_filename"])
    calc_all_grad(config, model, train_loader, valid_loader, args.ntest_start, args.ntest_end, 'TC')
