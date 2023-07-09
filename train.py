
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
    def __init__(self, eqs, mwps, kws, eqs_tok, mwps_tok, kws_tok, sep_tok, eos_tok, pad_tok):

        self.mwps = mwps
        self.kws = kws
        self.eqs = eqs

        self.mwps_tok = mwps_tok
        self.eqs_tok = eqs_tok
        self.kws_tok = kws_tok

        self.sep_tok = sep_tok
        self.eos_tok = eos_tok
        self.pad_tok = pad_tok

    def __getitem__(self, index):
        mwp = self.mwps[index]
        eq = self.eqs[index]
        kw = self.kws[index]
        mwp_tok = self.mwps_tok[index]
        eq_tok = self.eqs_tok[index]
        kw_tok = self.kws_tok[index]
        
  
        
        return {'eq':eq, 'mwp':mwp, 'kw':kw,
                'kw_tok':kw_tok, 'eq_tok':eq_tok, 'mwp_tok':mwp_tok,

                'sep_id':self.sep_tok, 'eos_id':self.eos_tok, 'pad_id':self.pad_tok,

                'input_ids': kw_tok + eq_tok,
                'decoder_input_ids': mwp_tok,
                'labels'   : mwp_tok,
                'input_ids_mwp2eq': mwp_tok,
                'labels_mwp2eq': eq_tok,
                'decoder_input_ids_mwp2eq':[self.sep_tok]+eq_tok[1:],
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
    lengths_label = [len(item['labels']) for item in batch]
    lengths_target = [len(item['decoder_input_ids']) for item in batch]

    x_batch = [x['input_ids'] + deepcopy([pad_id])*(max(lengths)-len(x['input_ids'])) for x in batch] # padding to the same length
    x_mask = [deepcopy([1])*l + deepcopy([0])*(max(lengths)-l) for l in lengths] # mask to ignore attention to pad tokens; required in gpt
    target_batch = [x['decoder_input_ids'] + deepcopy([pad_id])*(max(lengths_target)-len(x['decoder_input_ids'])) for x in batch]
    y_batch = [x['labels'] + deepcopy([-100])*(max(lengths_label)-len(x['labels'])) for x in batch]

    # assemble the eq_tok into a matrix
    mwp2eq_lengths = [len(item['input_ids_mwp2eq']) for item in batch]
    mwp2eq_label_lengths = [len(item['labels_mwp2eq']) for item in batch]
    mwp2eq_lengths_target = [len(item['decoder_input_ids_mwp2eq']) for item in batch]

    mwp2eq_x_batch = [item['input_ids_mwp2eq'] + deepcopy([pad_id])*(max(mwp2eq_lengths)-len(item['input_ids_mwp2eq'])) for item in batch]
    mwp2eq_y_batch = [item['labels_mwp2eq'] + deepcopy([-100])*(max(mwp2eq_label_lengths)-len(item['labels_mwp2eq'])) for item in batch]
    mwp2eq_mask = [deepcopy([1])*l + deepcopy([0])*(max(mwp2eq_lengths)-l) for l in mwp2eq_lengths]

    mwp2eq_target_batch = [x['decoder_input_ids_mwp2eq'] + deepcopy([pad_id])*(max(mwp2eq_lengths_target)-len(x['decoder_input_ids_mwp2eq'])) for x in batch]

    return {
            'eq': [item['eq'] for item in batch],
            'mwp': [item['mwp'] for item in batch],
            'kw': [item['kw'] for item in batch],
            'kw_tok': [item['kw_tok'] for item in batch],
            'eq_tok': [item['eq_tok'] for item in batch],
            'mwp_tok': [item['mwp_tok'] for item in batch],

            'input_ids': torch.tensor(x_batch, dtype=torch.long),
            'attention_mask': torch.tensor(x_mask, dtype=torch.float),
            'decoder_input_ids': torch.tensor(target_batch, dtype=torch.long),
            'labels': torch.tensor(y_batch, dtype=torch.long),

            'input_ids_mwp2eq': torch.tensor(mwp2eq_x_batch, dtype=torch.long),
            'attention_mask_mwp2eq': torch.tensor(mwp2eq_mask, dtype=torch.float),
            'decoder_input_ids_mwp2eq': torch.tensor(mwp2eq_target_batch, dtype=torch.long),
            'labels_mwp2eq': torch.tensor(mwp2eq_y_batch, dtype=torch.long),
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
    parser.add_argument('--eq_model',
                        type=str,
                        help='NLP Model to use for mwp2eq model. Default: t5-base',
                        default='t5-base')
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
    parser.add_argument('--use_op_weight',
                        help='use operator weight or not, default true',
                        choices=['true', 'false'],
                        default='true')

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
                        help="Adam weight_decay. Default: {}".format(0.0),
                        default=0.0)
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

    parser.add_argument('--get_testing_results',
                        action='store_true',
                        help="get_testing_results: TODO: Write a proper description",
                        default=True)

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
    parser.add_argument('--min_length',
                        type=int,
                        help="The minimum length of the sequence to be generated. Default: None",
                        default=None)
    parser.add_argument('--alpha',
                        type=float,
                        help="Alpha wegting factor for loss calculation. Default: {}".format(0.5),
                        default=0.5)

    parser.add_argument('--print_every',
                        type=int,
                        help="print_every. Default: {}".format(50),
                        default=50)
                        
# parametrs from wang model
    parser.add_argument('--tau',
                        type=int,
                        help="tau value. Default: {}".format(1),
                        default=1)
    parser.add_argument('--epochs_tau',
                        type=int,
                        help="for tau decay in eq model. Default: {}".format(100),
                        default=100)

    parser.add_argument('--mwp2eq_start_epoch',
                        type=int,
                        help="mwp2eq_start_epoch. Default: {}".format(2),
                        default=2)

    parser.add_argument('--tau_exp_decay',
                        action='store_true',
                        help="",
                        default=False)

    parser.add_argument('--fix_mwp2eq_model',
                        action='store_true',
                        help="",
                        default=False)

    parser.add_argument('--eq_coef',
                        type=int,
                        help="eq_coef. Default: {}".format(5),
                        default=5)

    parser.add_argument('--op_weight',
                        type=int,
                        help="op_weight. Default: {}".format(1),
                        default=1)

    parser.add_argument('--do_sample',
                        action='store_true',
                        help="default to false: Whether or not to use sampling ; use greedy decoding otherwise",
                        default=False)
    parser.add_argument('--top_k',
                        type=int,
                        help="The number of highest probability vocabulary tokens to keep for top-k-filtering. Default: {}".format(50),
                        default=50)

    parser.add_argument('--num_beams',
                        type=int,
                        help="Number of beams for beam search. 1 means no beam search Default: {}".format(1),
                        default=1)

    parser.add_argument('--length_penalty',
                        type=float,
                        help="Exponential penalty to the length(beamscore). Default: {}".format(0.0),
                        default=0.0)

    parser.add_argument('--temperature',
                        type=float,
                        help="The value used to module the next token probabilities. Default: {}".format(1.0),
                        default=1.0)
    parser.add_argument('--epsilon',
                        type=float,
                        help="Epsilon value for gumbel. Default: {}".format(1e-10),
                        default=1e-10)
    parser.add_argument('--hard',
                        action='store_true',
                        help="",
                        default=False)
    parser.add_argument('--mwp_no_repeat_ngram_size',
                        type=int,
                        help="mwp_no_repeat_ngram_size. Default: {}".format(0),
                        default=0)
    parser.add_argument('--eq_no_repeat_ngram_size',
                        type=int,
                        help="eq_no_repeat_ngram_size. Default: {}".format(0),
                        default=0)

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

    experiment_name = "{}|{}|{}|{}_{}|{} seed|{}-experiment".format(
        args.model,language,args.mwp_type,
        int(args.train_split*10), int(args.test_split*10),args.seed_len,args.experiment)

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

    mwpgen_model = get_model(args.model)
    mwpgen_model.config.max_length = args.max_length
    mwpgen_model.to(device)

    mwp2eq_model = get_model(args.eq_model)
    mwp2eq_model.config.max_length = args.max_length

    if args.use_op_weight == 'true':
        mwp2eq_model.load_state_dict(torch.load(save_weights_dir + '/mwpeq/mwpeq_{}_weight_op.pytorch'.format(args.eq_model)))
    else:
        mwp2eq_model.load_state_dict(torch.load(save_weights_dir + '/mwpeq/mwpeq_{}.pytorch'.format(args.eq_model)))

    mwp2eq_model.to(device)


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

    data_dir = os.path.join(args.data_dir,  language, args.experiment, str(args.seed_len))

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

    train_kws = open(os.path.join(data_dir, "train.source1")).read().splitlines()
    train_kws = [kw.strip() for kw in train_kws]
    test_kws = open(os.path.join(data_dir, "test.source1")).read().splitlines()
    test_kws = [kw.strip() for kw in test_kws]
    val_kws = open(os.path.join(data_dir, "val.source1")).read().splitlines()
    val_kws = [kw.strip() for kw in val_kws]


    train_mwps_tok = [mwpgen_tokenizer(mwp)['input_ids'] for mwp in train_mwps]
    train_eqs_tok = [mwpgen_tokenizer(eq)['input_ids'] for eq in train_eqs]
    train_kws_tok = [mwpgen_tokenizer(kw)['input_ids'] for kw in train_kws]

    test_mwps_tok = [mwpgen_tokenizer(mwp)['input_ids'] for mwp in test_mwps]
    test_eqs_tok = [mwpgen_tokenizer(eq)['input_ids'] for eq in test_eqs]
    test_kws_tok = [mwpgen_tokenizer(kw)['input_ids'] for kw in test_kws]

    val_mwps_tok = [mwpgen_tokenizer(mwp)['input_ids'] for mwp in val_mwps]
    val_eqs_tok = [mwpgen_tokenizer(eq)['input_ids'] for eq in val_eqs]
    val_kws_tok = [mwpgen_tokenizer(kw)['input_ids'] for kw in val_kws]


    train_dataset = MWPDataSet(train_eqs, train_mwps, train_kws, train_eqs_tok, train_mwps_tok, train_kws_tok, sep_id, eos_id, pad_id)
    val_dataset = MWPDataSet(val_eqs, val_mwps, val_kws, val_eqs_tok, val_mwps_tok, val_kws_tok, sep_id, eos_id, pad_id)
    test_dataset = MWPDataSet(test_eqs, test_mwps, test_kws, test_eqs_tok, test_mwps_tok, test_kws_tok, sep_id, eos_id, pad_id)

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

    optimizer = AdamW( list(mwpgen_model.parameters()) + list(mwp2eq_model.parameters()) , lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay)

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


    best_test_loss = 9999
    best_epoch = -100

    mwpgen_model.train()
    mwp2eq_model.eval()

    while epoch < args.epochs:
        epoch += 1
        train_loss = 0
        train_nll_loss = 0
        train_eq_loss = 0
        test_loss = 0
        test_nll_loss = 0
        test_eq_loss = 0
        
        log.info(f'Training at epoch {epoch}... ')
        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                batch_size = len(batch["mwp"])

                mwps, input_ids, labels, attention_mask, decoder_input_ids = batch['mwp'], batch['input_ids'].cuda(), batch['labels'].cuda(), batch['attention_mask'].cuda(), batch['decoder_input_ids'].cuda()
                input_ids_mwp2eq, labels_mwp2eq, attention_mask_mwp2eq, decoder_input_ids_mwp2eq = batch['input_ids_mwp2eq'].cuda(), batch['labels_mwp2eq'].cuda(),\
                batch['attention_mask_mwp2eq'].cuda(),batch['decoder_input_ids_mwp2eq'].cuda()

                outputs = mwpgen_model.forward(input_ids = input_ids,
                                                labels = labels, attention_mask = attention_mask, output_hidden_states=True)

                if epoch >= args.mwp2eq_start_epoch:
                    nll_loss = outputs[0] # compute loss
                else:
                    loss = outputs[0]
                    train_nll_loss += outputs[0].item()


                if epoch >= args.mwp2eq_start_epoch:
                    if not args.fix_mwp2eq_model:
                        mwp2eq_model.train()
                    
                    logits = outputs['logits']

                    mwp_tok, eq_tok = batch['mwp_tok'], batch['eq_tok']
                            
                    mwp2eq_input_embeds = torch.zeros(logits.shape[0], logits.shape[1], outputs['decoder_hidden_states'][-1].shape[2]).cuda()
                    
                    for bidx in range(input_ids.shape[0]):

                        # get the mwp logits 
                        mwp_logits = logits[bidx]
                        # compute gumbel
                        if args.tau_exp_decay:
                            mwp_gumbel = torch.nn.functional.gumbel_softmax(mwp_logits, tau=args.tau*np.exp(-(epoch-args.mwp2eq_start_epoch)/args.epochs_tau), hard=args.hard, eps=args.epsilon, dim=-1)
                        else:
                            mwp_gumbel = torch.nn.functional.gumbel_softmax(mwp_logits, tau=args.tau, hard=args.hard, eps=args.epsilon, dim=-1)

                        # compute the mwp embeddings
                        mwp_embeds = torch.matmul(mwp_gumbel, mwpgen_model.model.shared.weight)   # torch.Size([31, 1024]) = torch.Size([31, 64015]) * torch.Size([64015, 1024]
                        # compute the eq embedding
                        eq_embeds = mwpgen_model.model.shared(torch.tensor(eq_tok[bidx], dtype=torch.long).cuda())
                        # assemble the mwp2eq input
                        mwp2eq_input_embeds[bidx, :mwp_embeds.shape[0]] = mwp_embeds

                        # sanity check 
                        # assert(input_ids[bidx].tolist() == -----------------------[bidx].tolist())
                        # assert(mwp_embeds.shape[0]+eq_embeds.shape[0] == len(mwp_tok[bidx]) + 1 + len(eq_tok[bidx]) + 1)
                        assert(input_ids_mwp2eq[bidx, :len(mwp_tok[bidx])].tolist() == mwp_tok[bidx])
                        # assert(input_ids_mwp2eq[bidx, :mwp_embeds.shape[0]+eq_embeds.shape[0]].tolist() == mwp_tok[bidx] + [sep_id] + eq_tok[bidx] + [sep_id])
                        assert(labels_mwp2eq[bidx,:eq_embeds.shape[0]].tolist() == eq_tok[bidx])
                    
                    # set_trace()
                    # compute the cycle consistency loss

                        
                    outputs_mwp2eq = mwp2eq_model.forward(inputs_embeds=mwp2eq_input_embeds,
                                                        decoder_input_ids=decoder_input_ids_mwp2eq.cuda(),
                                                        labels=labels_mwp2eq.cuda(),
                                                        attention_mask=attention_mask_mwp2eq.cuda()
                                                        )
                    if args.use_op_weight == 'true':
                        logits_mwp2eq = outputs_mwp2eq[1]
                        shift_out = logits_mwp2eq[..., :-1, :].contiguous()
                        shift_labels = labels_mwp2eq[..., 1:].contiguous().cuda()
                        eq_loss = mwp2eq_loss_fct(shift_out.view(-1, shift_out.size(-1)), shift_labels.view(-1))
                    else:
                        eq_loss = outputs_mwp2eq[0]
                    loss = nll_loss + args.eq_coef * eq_loss
                    train_nll_loss += nll_loss.item()
                    train_eq_loss += args.eq_coef * eq_loss.item()

                    tbx.add_scalar('train/train_nll_loss_item', nll_loss.item(), step)
                    tbx.add_scalar('train/train_eq_loss_item', eq_loss.item(), step)
                    tbx.add_scalar('train/train_loss_item', nll_loss.item() + args.eq_coef * eq_loss.item(), step)
                # gradient step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                mwpgen_model.zero_grad()
                mwp2eq_model.zero_grad()

                scheduler.step()

                # record stat
                train_loss += loss.item()

                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=train_loss)
                step += batch_size

                tbx.add_scalar('train/loss', loss.item(), step)
                tbx.add_scalar('train/train_loss', train_loss / (batch_num+1), step)
                tbx.add_scalar('train/train_nll_loss', train_nll_loss/ (batch_num+1), step)
                tbx.add_scalar('train/train_eq_loss', train_eq_loss/ (batch_num+1), step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)


                if (batch_num + 1) % args.print_every == 0:
                    print('\t\t iteration {}, training, nll loss = {:.4f}, eq loss = {:.4f}, total = {:.4f}'.format(
                        batch_num+1, train_nll_loss / (batch_num+1), train_eq_loss / (batch_num+1), train_loss / (batch_num+1)))



        ###############
          # Evaluate
        ###############
        mwpgen_model.eval()
        mwp2eq_model.eval()

        # for batch_num, batch in enumerate(dev_loader):
        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(dev_loader):
                input_ids, labels, attention_mask = batch['input_ids'], batch['labels'], batch['attention_mask']
                batch_size = len(batch["mwp"])

                input_ids, labels, attention_mask,decoder_input_ids_mwp2eq = batch['input_ids'], batch['labels'], batch['attention_mask'], batch['decoder_input_ids_mwp2eq']
                outputs = mwpgen_model.forward(input_ids=input_ids.cuda(), labels=labels.cuda(), attention_mask=attention_mask.cuda(), output_hidden_states=True) # pass through model  

                if epoch >= args.mwp2eq_start_epoch:
                    nll_loss = outputs[0] # compute loss
                else:
                    test_loss += outputs[0].item()
                    test_nll_loss += outputs[0].item()

                # cycle consistency using gumbel softmax trick
                if epoch>= args.mwp2eq_start_epoch:
                    logits = outputs['logits']
                    mwp_tok, eq_tok, input_ids_mwp2eq, labels_mwp2eq, attention_mask_mwp2eq = batch['mwp_tok'], batch['eq_tok'], \
                        batch['input_ids_mwp2eq'], batch['labels_mwp2eq'], batch['attention_mask_mwp2eq']
                    mwp2eq_input_embeds = torch.zeros(logits.shape[0], logits.shape[1], outputs['decoder_hidden_states'][-1].shape[2]).cuda()
                    
                    for bidx in range(input_ids.shape[0]):

                        # get the mwp logits 
                        mwp_logits = logits[bidx]
                        # compute gumbel
                        if args.tau_exp_decay:
                            mwp_gumbel = torch.nn.functional.gumbel_softmax(mwp_logits, tau=args.tau*np.exp(-(epoch-args.mwp2eq_start_epoch)/args.epochs_tau), hard=args.hard, eps=args.epsilon, dim=-1)
                        else:
                            mwp_gumbel = torch.nn.functional.gumbel_softmax(mwp_logits, tau=args.tau, hard=args.hard, eps=args.epsilon, dim=-1)

                        # compute the mwp embeddings
                        mwp_embeds = torch.matmul(mwp_gumbel, mwpgen_model.model.shared.weight)
                        # compute the eq embedding
                        eq_embeds = mwpgen_model.model.shared(torch.tensor(eq_tok[bidx] + [sep_id], dtype=torch.long).cuda())
                        # assemble the mwp2eq input
                        mwp2eq_input_embeds[bidx, :mwp_embeds.shape[0]] = mwp_embeds
                        assert(input_ids_mwp2eq[bidx, :len(mwp_tok[bidx])].tolist() == mwp_tok[bidx])
                    # compute the cycle consistency loss
                    
                    outputs_mwp2eq = mwp2eq_model.forward(inputs_embeds=mwp2eq_input_embeds,
                                                        decoder_input_ids=decoder_input_ids_mwp2eq.cuda(),
                                                        labels=labels_mwp2eq.cuda(),
                                                        attention_mask=attention_mask_mwp2eq.cuda()
                                                        ) 
                    if args.use_op_weight == 'true':
                        logits_mwp2eq = outputs_mwp2eq[1]
                        shift_out = logits_mwp2eq[..., :-1, :].contiguous()
                        shift_labels = labels_mwp2eq[..., 1:].contiguous().cuda()
                        eq_loss = mwp2eq_loss_fct(shift_out.view(-1, shift_out.size(-1)), shift_labels.view(-1))
                    else:
                        eq_loss = outputs_mwp2eq[0]
                    
                    # record state
                    test_nll_loss += nll_loss.item()
                    test_eq_loss += eq_loss.item()
                    test_loss += nll_loss.item() + args.eq_coef * eq_loss.item()

                    tbx.add_scalar('test/test_nll_loss_item', nll_loss.item(), step)
                    tbx.add_scalar('test/test_eq_loss_item', eq_loss.item(), step)
                    tbx.add_scalar('test/test_loss_item', nll_loss.item() + args.eq_coef * eq_loss.item(), step)
        
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=test_loss)

                tbx.add_scalar('test/test_loss', test_loss / (batch_num+1), step)
                tbx.add_scalar('test/test_nll_loss', test_nll_loss/ (batch_num+1), step)
                tbx.add_scalar('test/test_eq_loss', test_eq_loss/ (batch_num+1), step)


        if (epoch == args.epochs):
            gen_mwp_example = mwpgen_tokenizer.decode(torch.argmax(outputs[1][-1], dim=1))
            print("---------------------------------------------------------")
            print('true mwp: ' + batch['mwp'][-1])
            print('pred mwp: ' + gen_mwp_example)
            gen_eq_example = mwpgen_tokenizer.decode(torch.argmax(outputs_mwp2eq[1][-1][mwp_embeds.shape[0]-1:mwp_embeds.shape[0]+eq_embeds.shape[0]-1], dim=1))
            print('true eq: ' + batch['eq'][-1])
            print('pred eq: ' + gen_eq_example)

        test_loss /= (batch_num+1)
        test_nll_loss /= (batch_num+1)
        test_eq_loss /= (batch_num+1)

        # log
        print('epoch {}, train loss={:.4f} (nll={:.4f}, eq={:.4f}), test loss={:.4f} (nll={:.4f}, eq={:.4f})\n'.format(
                                epoch, train_loss, train_nll_loss, train_eq_loss, test_loss, test_nll_loss, test_eq_loss))

        # save model, ignore the first time step:
        if test_nll_loss <= best_test_loss:

            if epoch >= args.mwp2eq_start_epoch:
                best_test_loss = test_nll_loss
            best_epoch = epoch
            print('\t best model at epoch {}'.format(epoch))

        if args.use_op_weight == 'true':
            torch.save(mwp2eq_model.state_dict(), save_weights_dir + '/mwpeq/mwpeq_{}_weight_op.pytorch'.format(args.eq_model))
        else:
            torch.save(mwp2eq_model.state_dict(), save_weights_dir + '/mwpeq/mwpeq_{}.pytorch'.format(args.eq_model))

        if args.fix_mwp2eq_model:
            torch.save(mwpgen_model.state_dict(), save_weights_dir + '/mwpgen/mwpgen_{}_fix_mwp2eq.pytorch'.format(args.model))
        elif args.use_op_weight == 'true':
            torch.save(mwpgen_model.state_dict(), save_weights_dir + '/mwpgen/mwpgen_{}_with_op_weight.pytorch'.format(args.model))
        elif args.tau_exp_decay:
            torch.save(mwpgen_model.state_dict(), save_weights_dir + '/mwpgen/mwpgen_{}_varying_tau_exp_decay.pytorch'.format(args.model))
        elif args.tau != 1:
            torch.save(mwpgen_model.state_dict(), save_weights_dir + '/mwpgen/mwpgen_{}_varying_tau_{}.pytorch'.format(args.tau, args.model))
        else:
            torch.save(mwpgen_model.state_dict(), save_weights_dir + '/mwpgen/mwpgen_{}.pytorch'.format(args.model))

        print()
        print('best epoch at {}; best loss = {:.4f}'.format(best_epoch, best_test_loss))

    tbx.close()
    
    # load the mwp2eq model (NOT the jointly trained one; reload the original pre-fine-tuned one)

    if args.use_op_weight == 'true':
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpeq/mwpeq_{}_weight_op.pytorch'.format(args.eq_model)))
    else:
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpeq/mwpeq_{}.pytorch'.format(args.eq_model)))
    mwp2eq_model.cuda()
    mwp2eq_model.eval()

    if args.fix_mwp2eq_model:
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpgen/mwpgen_{}_fix_mwp2eq.pytorch'.format(args.model)))
    elif args.use_op_weight == 'true':
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpgen/mwpgen_{}_with_op_weight.pytorch'.format(args.model)))
    elif args.tau_exp_decay:
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpgen/mwpgen_{}_varying_tau_exp_decay.pytorch'.format(args.model)))
    elif args.tau != 1:
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpgen/mwpgen_{}_varying_tau_{}.pytorch'.format(args.model, args.tau)))
    else:
        mwpgen_model.load_state_dict(torch.load(save_weights_dir + '/mwpgen/mwpgen_{}.pytorch'.format(args.model)))


    pred_list_all=[]
    mwp2eq_acc = []
    for test_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            kw, mwp, eq, input_ids, attention_mask, attention_mask_mwp2eq, eq_tok = batch['kw'], batch['mwp'], batch['eq'], batch['input_ids'], batch['attention_mask'], \
                                                                            batch['attention_mask_mwp2eq'],batch['eq_tok']

            mwpgen_input = input_ids.cuda()
            generated_ids_mwp = mwpgen_model.generate(mwpgen_input,
                                                    max_length=args.max_length,
                                                    do_sample=args.do_sample,
                                                    num_beams=args.num_beams,
                                                    top_k=args.top_k,
                                                    temperature=args.temperature,
                                                    no_repeat_ngram_size=args.mwp_no_repeat_ngram_size,
                                                    pad_token_id=pad_id)

            # mwp2eq acc
            mwpeq_input = generated_ids_mwp.cuda()
            generated_ids_eq = mwp2eq_model.generate(mwpeq_input,
                                                    max_length=50,
                                                    do_sample=args.do_sample,
                                                    num_beams=args.num_beams,
                                                    top_k=args.top_k,
                                                    temperature=args.temperature,
                                                    no_repeat_ngram_size=args.mwp_no_repeat_ngram_size,
                                                    pad_token_id=pad_id)

            for bidx in range(len(batch['mwp'])):
                mwp2eq_acc.append(generated_ids_eq[bidx].tolist()==eq_tok[bidx])

            outputs_decoded_mwp = mwpgen_tokenizer.batch_decode(generated_ids_mwp, skip_special_tokens=True)
            outputs_decoded_eq = mwpgen_tokenizer.batch_decode(generated_ids_eq, skip_special_tokens=True)
            for bidx in range(len(batch['mwp'])):
                print("------------------------------------------------------------------------------------------------------------------------")
                print("mwp: ",mwp[bidx])
                print("outputs_decoded_mwp: ",outputs_decoded_mwp[bidx])
                print("eq: ",eq[bidx])
                print("outputs_decoded_eq: ",outputs_decoded_eq[bidx])
            preds = list(zip(kw, mwp, outputs_decoded_mwp, eq, outputs_decoded_eq))
            pred_list_all.extend(preds)
    

    # save predictions for qualititative analysis
    util.save_preds(pred_list_all, record_dir, file_name="preds_all_test.csv")

    print('mwp2eq acc: {:.4f}'.format(sum(mwp2eq_acc) / float(len(mwp2eq_acc))))


# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())