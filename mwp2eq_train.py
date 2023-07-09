
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import traceback
from pathlib import Path

import socket
from collections import OrderedDict
from typing import *

import numpy as np

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartModel,
    BartTokenizer,
    BartForConditionalGeneration,
    GPT2Model,
    GPT2Tokenizer,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoModelForSeq2SeqLM, # for indicBART
    AlbertTokenizer, #https://huggingface.co/ai4bharat/IndicBART
    AutoTokenizer
)

from torch.utils.data import DataLoader, Dataset


import util

# seed for random
seed = 42

lang_dict = {
    "en" : "English",
    "si" : "Sinhala",
    "ta" : "Tamil",
    "ur" : "Urdu",
    "or" : "Odia",
    "hi" : "Hindi",
    "as" : "Assamese",
    "al" : "Albanian",
    "zh" : "Chinese"
}

model_dict = {
    "t5-base" : "t5-base",
    "t5-small" : "t5-small",
    "t5-large" : "t5-large",
    "bart-base" : "facebook/bart-base",
    "bart-large" : "facebook/bart-large",
    "gpt2" : "gpt2",
    "gpt2-medium" : "gpt2-medium",
    "gpt2-large" : "gpt2-large",
    "mt5-base" : "google/mt5-base",
    "mt5-small" : "google/mt5-small",
    "mt5-large" : "google/mt5-large",
    "mt5-xl" : "google/mt5-xl",
    "mt5-xxl" : "google/mt5-xxl",
    "mbart-large-50" : "facebook/mbart-large-50",
    "mbart-large-50-one-to-many-mmt" : "facebook/mbart-large-50-one-to-many-mmt",
    "mbart-large-50-many-to-many-mmt" : "facebook/mbart-large-50-many-to-many-mmt",
    "mbart-large-50-many-to-one-mmt" : "facebook/mbart-large-50-many-to-one-mmt",
    "mbart-large-cc25" : "facebook/mbart-large-cc25",
    "m2m100_418M": "facebook/m2m100_418M",
    "m2m100_1.2B": "facebook/m2m100_1.2B",
    "indic-bart": "ai4bharat/IndicBART",
    "indic-bartSS": "ai4bharat/IndicBARTSS"
}

def get_model(model_name, weight_path = ""):


    if weight_path =="":
        model_path = model_dict[model_name]
    else:
        model_path = weight_path

    if(model_name =="t5-base" or model_name =="t5-small"  or model_name =="t5-large" ):
        model = T5ForConditionalGeneration.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)

    elif(model_name == "bart-base" or model_name == "bart-large"):

        # Model predictions are intended to be identical to the original implementation when forced_bos_token_id=0.
        # This only works, however, if the string you pass to fairseq.encode starts with a space.
        # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
        model = BartForConditionalGeneration.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)

    elif(model_name == "gpt2" or model_name == "gpt2-medium" or model_name == "gpt2-large"):
        model = GPT2Model.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)

    elif(model_name == "mt5-base" or model_name == "mt5-small" or model_name == "mt5-large" or model_name == "mt5-xl" or model_name == "mt5-xxl"):
        model = MT5ForConditionalGeneration.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)

    elif(model_name == "mbart-large-50" or model_name == "mbart-large-50-one-to-many-mmt" or model_name == "mbart-large-50-many-to-many-mmt" or model_name == "mbart-large-50-many-to-one-mmt" or model_name == "mbart-large-cc25"):
        model = MBartForConditionalGeneration.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)

    elif(model_name == "m2m100_418M" or model_name == "m2m100_1.2B"):
        model = M2M100ForConditionalGeneration.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)
    elif(model_name == "indic-bart" or model_name == "indic-bartSS"):
        model = MBartForConditionalGeneration.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)
        # Or use model = AutoModelForSeq2SeqLM.from_pretrained(model_path,  output_attentions=True, output_hidden_states=True, return_dict = True)

    return model

def get_tokenizer(model_name, weight_path = ""):


    if weight_path =="":
        model_path = model_dict[model_name]
    else:
        model_path = weight_path

    if(model_name =="t5-base" or model_name =="t5-small"  or model_name =="t5-large" ):
        tokenizer = T5Tokenizer.from_pretrained(model_path)

    elif(model_name == "bart-base" or model_name == "bart-large"):
        tokenizer = BartTokenizer.from_pretrained(model_path)

    elif(model_name == "indic-bart" or model_name == "indic-bartSS"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False, use_fast=False, keep_accents=True)
        # Or use tokenizer = AlbertTokenizer.from_pretrained(model_path, do_lower_case=False, use_fast=False, keep_accents=True)

    elif(model_name == "gpt2" or model_name == "gpt2-medium" or model_name == "gpt2-large"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    elif(model_name == "mt5-base" or model_name == "mt5-small" or model_name == "mt5-large" or model_name == "mt5-xl" or model_name == "mt5-xxl"):
        tokenizer = MT5Tokenizer.from_pretrained(model_path)

    elif(model_name == "mbart-large-50" or model_name == "mbart-large-50-one-to-many-mmt" or model_name == "mbart-large-50-many-to-many-mmt" or model_name == "mbart-large-50-many-to-one-mmt" or model_name == "mbart-large-cc25"):
        # MBart50Tokenizer or MBart50TokenizerFast
        tokenizer = MBart50Tokenizer.from_pretrained(model_path)

    elif(model_name == "mbart-large-cc25"):
        tokenizer = MBartTokenizer.from_pretrained(model_path)

    elif(model_name == "m2m100_418M" or model_name == "m2m100_1.2B"):
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)

    return tokenizer


# A dataset for our inputs.
# prepare dataloader
class MWPDataSet(Dataset):
    # TODO: can optionally include "token_type_id" to indicate sentence A or sentence B in the input
    def __init__(self, eqs, mwps, eqs_tok, mwps_tok, sep_tok, eos_tok, pad_tok):

        self.mwps = mwps
        self.eqs = eqs

        self.mwps_tok = mwps_tok
        self.eqs_tok = eqs_tok

        self.sep_tok = sep_tok
        self.eos_tok = eos_tok
        self.pad_tok = pad_tok

    def __getitem__(self, index):
        mwp = self.mwps[index]
        eq = self.eqs[index]
        mwp_tok = self.mwps_tok[index]
        eq_tok = self.eqs_tok[index]

        return {'eq':eq, 'mwp':mwp,
                'eq_tok':eq_tok, 'mwp_tok':mwp_tok,
                'sep_id':self.sep_tok, 'eos_id':self.eos_tok, 'pad_id':self.pad_tok,
                'decoder_input_ids':[self.sep_tok]+eq_tok[1:],
                'input_ids':mwp_tok, 'labels':eq_tok,
                }

    def __len__(self):
        assert(len(self.eqs) == len(self.mwps))
        return len(self.eqs)

def collate_fn(batch):
    """
    pad input_ids in a batch to max length
    compute labels, ignore eq and kw and only attend to mwp which is the prediction target
    compute attention mask, which ignores padding
    """
    pad_id = batch[0]['pad_id'] # currently this is the same as the seperation token
    btc_size = len(batch)
    lengths = [len(item['input_ids']) for item in batch]

    lengths_target = [len(item['decoder_input_ids']) for item in batch]

    x_batch = [x['input_ids'] + deepcopy([pad_id])*(max(lengths)-len(x['input_ids'])) for x in batch] # padding to the same length
    x_mask = [deepcopy([1])*l + deepcopy([0])*(max(lengths)-l) for l in lengths] # mask to ignore attention to pad tokens; required in gpt2
    target_batch = [x['decoder_input_ids'] + deepcopy([pad_id])*(max(lengths_target)-len(x['decoder_input_ids'])) for x in batch]
    y_batch = [x['labels'] + deepcopy([-100])*(max(lengths_target)-len(x['labels'])) for x in batch] # pad each label seq with -100 to ignore compute loss on pad tokens

    return {
            'eq': [item['eq'] for item in batch],
            'mwp': [item['mwp'] for item in batch],
            'eq_tok': [item['eq_tok'] for item in batch],
            'mwp_tok': [item['mwp_tok'] for item in batch],
            'input_ids': torch.tensor(x_batch, dtype=torch.long),
            'decoder_input_ids': torch.tensor(target_batch, dtype=torch.long),
            'labels': torch.tensor(y_batch, dtype=torch.long),
            'attention_mask': torch.tensor(x_mask, dtype=torch.float),
            }

def write_output_to_text(pred_list_all,save_path):
  with open(save_path, 'w') as f:
    for item in pred_list_all:
        f.write("{0}\n".format(item))

def main():
    parser = argparse.ArgumentParser(
        description="Running main script for Multilingual Math Word Problem generator ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model',
                        type=str,
                        help='NLP Model to use. Default: indic-bartSS',
                        default='indic-bartSS')
    parser.add_argument('--language',
                        help='Language of the model. Default: {}'.format('en'),
                        choices=['en', 'si', 'ta', 'as', 'ur','hi', 'or', 'al', 'zh'],
                        default='en')
    parser.add_argument('--mwp_type',
                        help='Language of the model.',
                        choices=['Simple', 'Algebraic', 'Combine'],
                        default='Simple')
    parser.add_argument('--experiment',
                        type=str,
                        help='One word name to identify the experement',
                        default='')
    parser.add_argument('--seed_len',
                        type=float,
                        help='Fraction of the input sentence to use as model input. Default: {}'.format(0.5),
                        default=0.5)
    parser.add_argument('--work_dir',
                        type=str,
                        help="Working directory. All the files, including the repo should be inside of this directory",
                        required=True)

    parser.add_argument('--data_dir',
                        type=str,
                        help="Directory of the dataset"
                        "data/",
                        default="data/")
    parser.add_argument('--weight_dir',
                        type=str,
                        help="Directory of the dataset"
                        "data/",
                        default="data/")

    parser.add_argument('--epochs',
                        type=int,
                        help="Number of epochs  Default: {}".format(20),
                        default=20)
########################## optim / sched ##########################
    parser.add_argument('--lr',
                        type=float,
                        help="Initial learning rate. Default: {}".format(1e-4),
                        default=1e-4)
    parser.add_argument('--adam_eps',
                        type=float,
                        help="Adam optimizer epsilon. Default: {}".format(1e-8),
                        default=1e-8)
    parser.add_argument('--weight_decay',
                        type=float,
                        help="Adam weight_decay. Default: {}".format(0.001),
                        default=0.001)
    parser.add_argument('--scheduler',
                        help='Name of the scheduler. Cosine or linear',
                        choices=['linear', 'cosine'],
                        default='Simple')
    parser.add_argument('--warmup_steps',
                        type=int,
                        help="Initial learning rate. Default: {}".format(0),
                        default=0)
    parser.add_argument('--max_grad_norm',
                        type=float,
                        help="Initial learning rate. Default: {}".format(1.0),
                        default=1.0)
    parser.add_argument('--train_split',
                        type=float,
                        help="Training split. Default: {}".format(0.6),
                        default=0.6)
    parser.add_argument('--test_split',
                        type=float,
                        help="Test split. Default: {}".format(0.3),
                        default=0.3)
########################## config info ##########################

    parser.add_argument('--num_train',
                        type=int,
                        help="num_train. Default: {}, will use all".format(-1),
                        default=-1)
    parser.add_argument('--num_val',
                        type=int,
                        help="num_val. Default: {}, will use all".format(-1),
                        default=-1)
    parser.add_argument('--batch_size',
                        type=int,
                        help="Batch size. Default: {}".format(4),
                        default=4)
    parser.add_argument('--num_workers',
                        type=int,
                        help="num_workers. Default: {}".format(4),
                        default=4)

    parser.add_argument('--use_wandb',
                        action='store_true',
                        help="Whether to log to wandb"
                             "(you'll need to set up wandb env info)",
                        default=False)

    parser.add_argument('--use_trained',
                        action='store_true',
                        help="use previously trained, saved model",
                        default=False)

# source and target lengths for dataloader. If you know your lengths you can change these, or
# add a collate function to handle different sizes. Depending on your inputs you should change these.


    parser.add_argument('--max_src_len',
                        type=int,
                        help="max_src_len. Default: {}".format(200),
                        default=200)

    parser.add_argument('--max_tgt_len',
                        type=int,
                        help="max_tgt_len. Default: {}".format(400),
                        default=400)
    parser.add_argument('--max_length',
                        type=int,
                        help="Value for model.config.max_length. Default: {}".format(50),
                        default=50)
    parser.add_argument('--alpha',
                        type=float,
                        help="Alpha wegting factor for loss calculation. Default: {}".format(0.5),
                        default=0.5)

    parser.add_argument('--print_every',
                        type=int,
                        help="print_every. Default: {}".format(50),
                        default=50)

# parametrs from wang model
    parser.add_argument('--op_weight',
                        type=int,
                        help="op_weight. Default: {}".format(1),
                        default=1)

    parser.add_argument('--use_op_weight',
                        help='use operator weight or not, default true',
                        choices=['true', 'false'],
                        default='true')

    parser.add_argument('--eq_coef',
                        type=int,
                        help="eq_coef. Default: {}".format(5),
                        default=5)

    parser.add_argument('--alternate_train',
                        action='store_true',
                        help="default to false: train by (1 epoch no eq loss, 1 epoch with eq loss), and repeat",
                        default=False)

    args = parser.parse_args()

    language = lang_dict[args.language]

    # log warnings with default optionals that will be used in the script run
    for arg in vars(args):
        def_val = parser.get_default(arg)
        if getattr(args, arg) == def_val:
            if isinstance(def_val, list):
                def_val = ' '.join(def_val)
            if arg == 'timeout':
                def_val = args.timeout = str(args.timeout * args.iterations)
            print("WARN: %s is not set, using default value ('%s')" % (arg, def_val))

    util.set_seed(seed)

    print("INFO: Train-Split: {}, Test-Split: {}".format(args.train_split, args.test_split))

    comment=\
        """
        Started finetuning {} for MWP Generation...
        Using mulipleSentenceDataset.
        """.format(args.model)

    experiment_name = "{}|{}|{}_{}|{} seed|{}-experiment".format(
        args.model, args.mwp_type, int(args.train_split*10), int(args.test_split*10),args.seed_len,args.experiment)

    save_dir = os.path.join(args.work_dir, "save", experiment_name)
    save_weights_dir = os.path.join(args.work_dir, "save_weights", experiment_name)
    log_dir = os.path.join(args.work_dir, "logs", experiment_name)

    print("=============================================================")
    print("DEBUG: experiment_name = {}".format(experiment_name))
    print("DEBUG: save_dir = {}".format(save_dir))
    print("DEBUG: save_weights_dir = {}".format(save_weights_dir))
    print("DEBUG: log_dir = {}".format(log_dir))

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(save_weights_dir + '/mwpeq/').mkdir(parents=True, exist_ok=True)
    Path(save_weights_dir + "/mwpgen/").mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # Save the predicted text file paths
    save_predicted_eval_output_text_dir = os.path.join(log_dir, 'eval.txt')
    save_predicted_test_output_text_dir =  os.path.join(log_dir, 'test.txt')

    if args.use_wandb:
        wandb.init()
        record_dir = wandb.run.dir
        wandb.tensorboard.patch(save=True, tensorboardX=True)
    else:
        record_dir = util.get_save_dir(save_dir, experiment_name)

    global log
    log = util.get_logger(record_dir, "root", "debug")
    tbx = SummaryWriter(record_dir, flush_secs=5)
    log.info(experiment_name)
    log.info(comment)


    all_config = {
        "save_dir": save_dir,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "model": args.model,
        "lr": args.lr,
        "adam_eps": args.adam_eps,
        "warmup": args.warmup_steps,
        "workers": args.num_workers,
        "max grad": args.max_grad_norm,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "batch_size": args.batch_size,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len
    }

    #Initialize model and tokenizer
    device, gpu_ids = util.get_available_devices()


    mwpgen_tokenizer = get_tokenizer(args.model)
    mwpeq_model = get_model(args.model)
    mwpeq_model.config.max_length = args.max_length
    mwpeq_model.to(device)



    if args.use_op_weight == 'true':
        all_operators = ['+', '-', '*', '/']
        all_operators_tok = [mwpgen_tokenizer(x)['input_ids'] for x in all_operators]
        all_operators_tok = [x[1] for x in all_operators_tok]
        cls_weights = torch.tensor([1]*len(mwpgen_tokenizer), dtype=torch.float)
        for idx in all_operators_tok:
            cls_weights[idx] = args.op_weight
        mwp2eq_loss_fct = nn.CrossEntropyLoss(weight=cls_weights)
        mwp2eq_loss_fct.cuda()



    eos_id = mwpgen_tokenizer.eos_token_id
    sep_id = mwpgen_tokenizer.sep_token_id
    bos_id = mwpgen_tokenizer.bos_token_id
    pad_id = mwpgen_tokenizer.pad_token_id

    # print("----------------------------")
    # print(model.config)
    # print("----------------------------")

    data_dir = os.path.join(args.data_dir, args.experiment, str(args.seed_len))


    train_eqs = open(os.path.join(data_dir, "train.target2")).read().splitlines()
    train_eqs = [eq.strip() for eq in train_eqs]
    test_eqs = open(os.path.join(data_dir, "test.target2")).read().splitlines()
    test_eqs = [eq.strip() for eq in test_eqs]
    val_eqs = open(os.path.join(data_dir, "val.target2")).read().splitlines()
    val_eqs = [eq.strip() for eq in val_eqs]

    train_mwps = open(os.path.join(data_dir, "train.target1")).read().splitlines()
    train_mwps = [mwp.strip() for mwp in train_mwps]
    test_mwps = open(os.path.join(data_dir, "test.target1")).read().splitlines()
    test_mwps = [mwp.strip() for mwp in test_mwps]
    val_mwps = open(os.path.join(data_dir, "val.target1")).read().splitlines()
    val_mwps = [mwp.strip() for mwp in val_mwps]


    train_mwps_tok = [mwpgen_tokenizer(mwp)['input_ids'] for mwp in train_mwps]
    train_eqs_tok = [mwpgen_tokenizer(eq)['input_ids'] for eq in train_eqs]

    test_mwps_tok = [mwpgen_tokenizer(mwp)['input_ids'] for mwp in test_mwps]
    test_eqs_tok = [mwpgen_tokenizer(eq)['input_ids'] for eq in test_eqs]

    val_mwps_tok = [mwpgen_tokenizer(mwp)['input_ids'] for mwp in val_mwps]
    val_eqs_tok = [mwpgen_tokenizer(eq)['input_ids'] for eq in val_eqs]


    train_dataset = MWPDataSet(train_eqs, train_mwps, train_eqs_tok, train_mwps_tok, sep_id, eos_id, pad_id)
    val_dataset = MWPDataSet(val_eqs, val_mwps, val_eqs_tok, val_mwps_tok, sep_id, eos_id, pad_id)
    test_dataset = MWPDataSet(test_eqs, test_mwps, test_eqs_tok, test_mwps_tok, sep_id, eos_id, pad_id)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn, drop_last=False)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn, drop_last=False)
    ####################################################################

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_val = len(dev_loader.dataset)
    num_test = len(test_loader.dataset)

    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs

    # model.to(device)

    optimizer = optim.AdamW(mwpeq_model.parameters() , lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay)

    if(args.scheduler == "linear"):
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=total_steps)



    log.info(f'device: {device}\n'
             f'gpu_ids: {gpu_ids}\n'
             f'total_steps: {total_steps}\n'
             f'total_train (num_t * epoch): {total_train}\n'
             f'machine: {socket.gethostname()}\n')

    config_str = "\n"
    for k, v in all_config.items():
        config_str += f'{k}: {v}\n'
    config_str += f'record_dir: {record_dir}\n'
    log.info(config_str)

    epoch = 0       # number of times we have passed through entire set of training examples
    step = 0        # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)


    print('start training ... ')

    best_test_loss = 9999
    best_epoch = -100
    ep_loss = []

    while epoch < args.epochs:
        epoch += 1


        train_loss = 0
        train_nll_loss = 0
        train_eq_loss = 0
        test_loss = 0
        test_nll_loss = 0
        test_eq_loss = 0

        mwpeq_model.train()
        log.info(f'Training at epoch {epoch}... ')
        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                batch_size = len(batch["mwp"])




                input_ids, labels, attention_mask, decoder_input_ids = batch['input_ids'], batch['labels'], batch['attention_mask'], batch['decoder_input_ids']
                outputs = mwpeq_model.forward(input_ids=input_ids.cuda(),
                                        decoder_input_ids=decoder_input_ids.cuda(),
                                        labels=labels.cuda(),
                                        attention_mask=attention_mask.cuda()) # pass through model


                # compute loss
                if args.use_op_weight == 'true':
                    logits = outputs[1]
                    # Shift so that tokens < n predict n
                    shift_out = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().cuda()
                    # Flatten the tokens
                    loss = mwp2eq_loss_fct(shift_out.view(-1, shift_out.size(-1)), shift_labels.view(-1))
                else:
                    loss = outputs[0]
                # set_trace()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                mwpeq_model.zero_grad()
                train_loss += loss.item()

                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=train_loss)

                tbx.add_scalar('train/loss_item', loss.item(), step)

                if (batch_num + 1) % args.print_every == 0:
                    print('\t\t iteration {}, training loss = {:.4f}'.format(batch_num+1, train_loss / (batch_num+1)))

            train_loss /= (batch_num+1)
            tbx.add_scalar('train/loss', train_loss, step)

            mwpeq_model.eval()
            with torch.no_grad(), tqdm(total=num_val) as progress_bar:
                for batch_num, batch in enumerate(dev_loader):
                    input_ids, labels, attention_mask, decoder_input_ids = batch['input_ids'], batch['labels'], batch['attention_mask'], batch['decoder_input_ids']
                    outputs = mwpeq_model.forward(input_ids=input_ids.cuda(),
                                            decoder_input_ids=decoder_input_ids.cuda(),
                                            labels=labels.cuda(),
                                            attention_mask=attention_mask.cuda()) # pass through model
                    # compute loss
                    if args.use_op_weight == 'true':
                        logits = outputs[1]
                        # Shift so that tokens < n predict n
                        shift_out = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous().cuda()
                        # Flatten the tokens
                        loss = mwp2eq_loss_fct(shift_out.view(-1, shift_out.size(-1)), shift_labels.view(-1))
                    else:
                        loss = outputs[0]

                    test_loss += loss.item()
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch, loss=test_loss)
                    tbx.add_scalar('test/loss_item', loss.item(), step)

                test_loss /= (batch_num+1)
                tbx.add_scalar('test/loss', test_loss, step)
                print('epoch {}, train loss={:.4f}, test loss={:.4f}'.format(epoch, train_loss, test_loss))

                # save model
                if test_loss <= best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch
                    print('\t best model at epoch {}'.format(epoch))
                if args.use_op_weight == 'true':
                    torch.save(mwpeq_model.state_dict(), save_weights_dir + '/mwpeq/mwpeq_{}_weight_op.pytorch'.format(args.model))
                else:
                    torch.save(mwpeq_model.state_dict(), save_weights_dir + '/mwpeq/mwpeq_{}.pytorch'.format(args.model))

        print('best epoch at {}; best loss = {:.4f}'.format(best_epoch, best_test_loss))
        print('model={}, lr={}, use_scheduler={}, bs={}, epochs={}, use loss weight={}, operator weight={}'.format(
            args.model, args.lr, args.scheduler, args.batch_size, args.epochs, args.use_op_weight, args.op_weight))


    tbx.close()
    print('start evaluation')
    avg_mwp2eq_precision = 0
    avg_mwp2eq_acc = 0

    # load the trained model
    if args.use_op_weight == 'true':
        mwpeq_model.load_state_dict(torch.load(save_weights_dir + '/mwpeq/mwpeq_{}_weight_op.pytorch'.format(args.model)))
    else:
        mwpeq_model.load_state_dict(torch.load(save_weights_dir + '/mwpeq/mwpeq_{}.pytorch'.format(args.model)))

    mwpeq_model.eval()

    pred_list_all=[]
    for test_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            mwp, eq, input_ids, attention_mask, eq_tok, mwp_tok = batch['mwp'], batch['eq'], batch['input_ids'], batch['attention_mask'], batch['eq_tok'], batch['mwp_tok']

            mwpeq_input = input_ids.cuda()
            generated_ids_eq = mwpeq_model.generate(mwpeq_input, attention_mask= attention_mask.cuda(),num_beams=25, max_length=50, early_stopping=True)
            reconstruction_eq = generated_ids_eq[:,1:].tolist()

            for bidx in range(len(batch['mwp'])):
                precision = len(set(reconstruction_eq[bidx]).intersection(set(eq_tok[bidx][1:-1]))) / (len(reconstruction_eq[bidx])-1) # remove last sep token

                avg_mwp2eq_precision += precision
                avg_mwp2eq_acc += reconstruction_eq[bidx][:-1]==eq_tok[bidx][1:-1]

            outputs_decoded_eq = mwpgen_tokenizer.batch_decode(generated_ids_eq[:,1:], skip_special_tokens=True)

            print("------------------------------------------------------------------------------------------------------------------------")
            print("Input mwp: ",mwp)
            print("Expected eq: ",eq)
            print("Predicted eq: ",outputs_decoded_eq)

            preds = list(zip(mwp,eq, outputs_decoded_eq))
            pred_list_all.extend(preds)

    avg_mwp2eq_precision /= len(test_loader)
    avg_mwp2eq_acc /= len(test_loader)


            # print(mwp2eq_acc)


    # save predictions for qualititative analysis
    util.save_preds(pred_list_all, record_dir, file_name="preds_all_test.csv")

    print('avg_mwp2eq_precision: {:.4f} \t avg_mwp2eq_acc: {:.4f}'.format(avg_mwp2eq_precision, avg_mwp2eq_acc))


# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())