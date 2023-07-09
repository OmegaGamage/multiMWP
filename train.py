
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import traceback
from pathlib import Path

import socket
from collections import OrderedDict
from typing import *

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
    BloomForCausalLM,
    BloomTokenizerFast,
    GPT2Model,
    GPT2Tokenizer,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    MBartForConditionalGeneration,
    MBartTokenizer,
    MBart50Tokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoModelForSeq2SeqLM, # for indicBART
    AlbertTokenizer, #https://huggingface.co/ai4bharat/IndicBART
    AutoTokenizer,
    XGLMTokenizer,
    XGLMModel
)

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
    "zh" : "Chinese",
    "dv" : "Dhivehi",
    "fa" : "Persian",
    "ar" : "Arabic",
    "ne" : "Nepali",
    "bn" : "Bengali"
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
    "indic-bartSS": "ai4bharat/IndicBARTSS",
    "xglm-564M" : 'facebook/xglm-564M',
    "bloom-560m": "bigscience/bloom-560m",
}

tokenizer_lang_dic = {
    "en" : "en_XX",
    "si" : "si_LK",
    "ta" : "ta_IN",
    "ur" : "ur_PK",
    "or" : "bn_IN",
    "hi" : "hi_IN",
    "as" : "bn_IN",
    "al" : "en_XX",
    "zh" : "zh_CN",
    "bn" : "bn_IN",
    "fa" : "fa_IR",
    "ar" : "ar_AR",
    "ne" : "ne_NP",
    "dv" : "si_LK"
}

def get_model(model_name, weight_path = ""):


    if weight_path =="":
        model_path = model_dict[model_name]
    else:
        model_path = weight_path

    if(model_name =="t5-base" or model_name =="t5-small"  or model_name =="t5-large" ):
        model = T5ForConditionalGeneration.from_pretrained(model_path)

    elif(model_name == "bart-base" or model_name == "bart-large"):

        # Model predictions are intended to be identical to the original implementation when forced_bos_token_id=0.
        # This only works, however, if the string you pass to fairseq.encode starts with a space.
        # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
        model = BartForConditionalGeneration.from_pretrained(model_path)

    elif(model_name == "gpt2" or model_name == "gpt2-medium" or model_name == "gpt2-large"):
        model = GPT2Model.from_pretrained(model_path)

    elif(model_name == "mt5-base" or model_name == "mt5-small" or model_name == "mt5-large" or model_name == "mt5-xl" or model_name == "mt5-xxl"):
        model = MT5ForConditionalGeneration.from_pretrained(model_path)

    elif(model_name == "mbart-large-50" or model_name == "mbart-large-50-one-to-many-mmt" or model_name == "mbart-large-50-many-to-many-mmt" or model_name == "mbart-large-50-many-to-one-mmt" or model_name == "mbart-large-cc25"):
        model = MBartForConditionalGeneration.from_pretrained(model_path)

    elif(model_name == "m2m100_418M" or model_name == "m2m100_1.2B"):
        model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    elif(model_name == "indic-bart" or model_name == "indic-bartSS"):
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        # Or use model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif(model_name == "xglm-564M"):
        model = XGLMModel.from_pretrained(model_path)
    elif model_name == "bloom-560m":
        model = BloomForCausalLM.from_pretrained(model_path)

    return model

def get_tokenizer(model_name, language = 'en', weight_path = ""):

    language_label = tokenizer_lang_dic[language]

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
        tokenizer = MBart50Tokenizer.from_pretrained(model_path, src_lang=language_label, tgt_lang=language_label)

    elif(model_name == "mbart-large-cc25"):
        tokenizer = MBartTokenizer.from_pretrained(model_path)

    elif(model_name == "m2m100_418M" or model_name == "m2m100_1.2B"):
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        tokenizer.src_lang = language
        tokenizer.tgt_lang = language

    elif(model_name == "xglm-564M"):
        tokenizer = XGLMTokenizer.from_pretrained(model_path)
    elif model_name == "bloom-560m":
        tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    return tokenizer

# A dataset for our inputs.
class MWPDataSet(Dataset):
    def __init__(self, tokenizer, data_dir: str, type_path, max_examples=-1,
                 max_src_len=200, max_tgt_len=500):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all

        max_src and max_tgt len refer to number of tokens in the input sequences
        # Note: these are not randomized. If they were we might need to collate.
        """

        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"

        self.example_path = Path(data_dir) / type_path
        self.max_examples = max_examples
        self.tokenizer = tokenizer

        self.max_src_len = max_src_len  # max num of tokens in tokenize()
        self.max_tgt_len = max_tgt_len

        self.inputs = []            # list of dict
        self.targets = []           # list of dict
        self.input_text = []        # list of str
        self.target_text = []       # list of str

        self._build()       # fill inputs, targets, max_lens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        src_text = self.input_text[index]
        tgt_text = self.target_text[index]

        # These will be cast to torch.long in forward
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "source_text": src_text, "target_text": tgt_text}

    def _build(self):
        source_path = self.example_path.with_suffix(".source")
        target_path = self.example_path.with_suffix(".target")

        with open(source_path, 'r') as f_source, \
                open(target_path, 'r') as f_target:

            source, target = f_source.readlines(), f_target.readlines()
            source_ct, target_ct = len(source), len(target)
            assert source_ct == target_ct , f"Lengths don't match"

            # Note we could batch encode
            log.warning(f'Using max_src_len, max_tgt_len = ({self.max_src_len}, {self.max_tgt_len})')

            inputs_out = []     # accumulate the output of batch_encode
            targets_out = []    # same
            inputs_text = []    # save the original text for evaluations
            targets_text = []   # same

            if self.max_examples > 0 :
                source_ct = min(self.max_examples, source_ct)

            for idx in range(source_ct):
                # append end of sequence tokens (not necessary) because handled by tokenize() call
                src = source[idx].strip()
                tgt = target[idx].strip()

                inputs_text.append(src)
                targets_text.append(tgt)

                # tokenize
                # padding="max_length" pads to max_len
                # otherwise (e.g. for batch), we could use padding=longest with truncation
                # note: don't need add_special_tokens since EOS added automatically and others are PAD
                # self.tokenizer returns a dict of input_ids and attention_masks (where attn masks corresponds to padding)
                # Note: padding could also be done via collate in dataloader
                #TODO we could actually batch encode these (i.e. multiple per)
                tokenized_inputs = self.tokenizer(
                    [src], max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
                )
                tokenized_targets = self.tokenizer(
                    [tgt], max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
                )
                inputs_out.append(tokenized_inputs)
                targets_out.append(tokenized_targets)
            self.inputs = inputs_out
            self.targets = targets_out
            self.input_text = inputs_text
            self.target_text = targets_text

"""
Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
# Note:
# - we default to not shuffling the dev set
"""
def get_dataloaders(tokenizer, batch_size, num_train, num_val, data_dir, num_workers, k_max_src_len, k_max_tgt_len,
                    shuffle_train=True, shuffle_dev=False,shuffle_test=False) -> Tuple[DataLoader, DataLoader, DataLoader]:

    #TODO should pass max src and max tgt len in as arguments
    train_data_set = MWPDataSet(tokenizer, type_path="train", data_dir=data_dir, max_examples=num_train,
                               max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    eval_data_set = MWPDataSet(tokenizer, type_path="val", data_dir=data_dir, max_examples=num_val,
                              max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    test_data_set = MWPDataSet(tokenizer, type_path="test", data_dir=data_dir, max_examples=num_val,
                              max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
    log.info(f'Datasets loaded with sizes: train: {len(train_data_set)}, dev: {len(eval_data_set)}, test: {len(test_data_set)}')

    return train_loader, eval_loader , test_loader

def forward(model, device, batch):
    src_ids = batch["source_ids"].to(device, dtype=torch.long)
    src_mask = batch["source_mask"].to(device, dtype=torch.long)
    tgt_ids = batch["target_ids"].to(device, dtype=torch.long)

    # padded ids (pad=0) are set to -100, which means ignore for loss calculation
    tgt_ids[tgt_ids[: ,:] == 0 ] = -100
    label_ids = tgt_ids.to(device)
    # when we call model() with labels, they will be
    # - automatically right shifted by 1 (for teacher forcing)
    # - prepended by BOS=Beginning of sequence which is a PAD token
    # - any token that was -100 will be masked_fill_ to <pad> for teacher forcing
    # return_dict means return as a dictionary
    out_dict = model(src_ids, attention_mask=src_mask, labels=label_ids, return_dict=True)
    loss, logits = out_dict['loss'], out_dict['logits']
    return loss, logits

def write_output_to_text(pred_list_all,save_path):
  with open(save_path, 'w') as f:
    for item in pred_list_all:
        f.write("{0}\n".format(item))

# def get_model(model_name):

# def get_tokenizer(model_name):


def main():
    parser = argparse.ArgumentParser(
        description="Running main script for Multilingual Math Word Problem generator ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model',
                        type=str,
                        help='NLP Model to use. Default: t5-base',
                        default='t5-base')
    parser.add_argument('--language',
                        help='Language of the model. Default: {}'.format('en'),
                        choices=['en', 'si', 'ta', 'as', 'ur','hi', 'or', 'al', 'zh', 'dv', 'bn', 'fa', 'ar', 'ne'],
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

    parser.add_argument('--weight_dir',
                        type=str,
                        help="Directory of the dataset"
                        "data/",
                        default="data/")
    parser.add_argument('--use_trained',
                        action='store_true',
                        help="use previously trained, saved model",
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
    Path(save_weights_dir).mkdir(parents=True, exist_ok=True)
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


    if(args.use_trained):
        weight_path = args.weight_dir
        model = get_model(args.model, weight_path)
        model.config.max_length = args.max_length
        tokenizer = get_tokenizer(args.model,  args.language, weight_path)
    else:
        model = get_model(args.model)
        model.config.max_length = args.max_length
        tokenizer = get_tokenizer(args.model, args.language)

    # print(model.config)
    print("----------------------------")
    print(model.config)
    print("----------------------------")

    data_dir = os.path.join(args.data_dir,  language, args.experiment, str(args.seed_len))
    train_loader, dev_loader, test_loader = \
        get_dataloaders(tokenizer, batch_size=args.batch_size, num_train=args.num_train, num_val=args.num_val,
                        data_dir=data_dir, num_workers=args.num_workers, k_max_src_len = args.max_src_len,
                        k_max_tgt_len = args.max_tgt_len)

    # reset in case we used the -1 flag for all
    num_train = len(train_loader.dataset)
    num_val = len(dev_loader.dataset)
    num_test = len(test_loader.dataset)

    total_steps = ( (num_train // args.batch_size) * args.epochs)     # num times that optim.step() will be called
    total_train = num_train * args.epochs

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_eps, weight_decay=args.weight_decay)

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
    while epoch < args.epochs:
        epoch += 1
        model.train()
        log.info(f'Training at epoch {epoch}... ')
        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                batch_size = len(batch["source_ids"])
                loss, logits = forward(model, device, batch)
                loss_val = loss.item()      # get the item since loss is a tensor

                # Backward
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()        # don't need to pass step to scheduler

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val)
                tbx.add_scalar('train/loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                log.info(f'train/loss at step {step} -> {loss_val} ')


        ###############
          # Evaluate
        ###############
        if(epoch == args.epochs):

          log.info(f'Evaluating at step {step}...')
          model.eval()        # put model in eval mode

          # See how the model is doing with exact match on tokens
          pred_list_all = []                      # accumulate for saving; list; one list per epoch
          pred_list_correct = []
          loss_meter = util.AverageMeter()    # NLL (default metric for model) (reset each time)

          prediction_list_for_write_to_text = []

          # set up two count variables
          total_matches_no_eos_ct = 0
          total_matches_with_eos_ct = 0

          with torch.no_grad(), \
              tqdm(total=num_val) as progress_bar:
              for batch_num, batch in enumerate(dev_loader):
                  batch_size = len(batch["source_ids"])

                  # evaluation for loss fcn
                  loss, _ = forward(model, device, batch)     # loss, logits, but don't need logits
                  loss_meter.update(loss.item(), batch_size)  # loss.item() since it's a tensor

                  # predict / generate for token matches
                  src_ids = batch["source_ids"].to(device, dtype=torch.long)
                  src_mask = batch["source_mask"].to(device, dtype=torch.long)
                  tgt_ids = batch["target_ids"].to(device, dtype=torch.long)
                  # note you could tweak the generation params. See huggingface details for generate
                  generated_ids = model.generate(src_ids, attention_mask=src_mask, max_length=None, min_length=None)       # (batch x seq length) -------Added min length

                  # collect some stats
                  total_matches_no_eos, total_matches_with_eos, correct_indices = \
                      util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                  total_matches_no_eos_ct += total_matches_no_eos
                  total_matches_with_eos_ct += total_matches_with_eos

                  # save for qualitative analysis
                  orig_text_input, orig_text_output = batch["source_text"], batch["target_text"]
                  #TODO this could break once skip_special_tokens is fixed
                  outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  #skip_special_tokens=False
                  preds = list(zip(orig_text_input, orig_text_output, outputs_decoded))
                  pred_list_all.extend(preds)

                  # ----------print-----------
                  print(outputs_decoded)
                  prediction_list_for_write_to_text.extend(outputs_decoded)

                  # we also store only the correct indices
                  for idx in correct_indices.tolist():    # tensor to list; these are the valid indices
                      pred_list_correct.append(preds[idx[0]])     # each item was a list of one element

                  # print one batch of generations for qualitative assessment
                  if batch_num == 0:
                      for orig_input, orig_target, actual_output in preds[:1]:
                          log.info(f'Source: {orig_input}\t Target: {orig_target}\n'
                                  f'\t Actual: {actual_output}')

                  # Log info
                  progress_bar.update(batch_size)
                  progress_bar.set_postfix(NLL=loss_meter.avg)

          # save predictions for qualititative analysis
          util.save_preds(pred_list_all, record_dir, file_name="preds_all_eval.csv")
          util.save_preds(pred_list_correct, record_dir, file_name="preds_correct_eval.csv")
          results_list = [('NLL', loss_meter.avg),
                          ('exact_match_with_eos', total_matches_with_eos_ct),
                          ('exact_match_no_eos', total_matches_no_eos_ct)]
          results = OrderedDict(results_list)


          # Save predictions to a output file
          write_output_to_text(prediction_list_for_write_to_text, save_predicted_eval_output_text_dir)


          # Log to console
          results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
          log.info(f'Dev {results_str}')

          # Log to TensorBoard
          for k, v in results.items():
              tbx.add_scalar(f'dev/{k}', v, step)
          util.visualize(tbx,
                        pred_dict=pred_list_all,     # will be truncated by num_visuals
                        step=step,
                        split='dev',
                        num_visuals=3)


        ###############
          # TEST (you might want to save checkpoints)
        ###############
        if(epoch == args.epochs and args.get_testing_results==True):

          log.info(f'Testing at step {step}...')
          model.eval()        # put model in eval mode

          # See how the model is doing with exact match on tokens
          test_pred_list_all = []                      # accumulate for saving; list; one list per epoch
          test_pred_list_correct = []
          loss_meter = util.AverageMeter()    # NLL (default metric for model) (reset each time)

          Test_prediction_list_for_write_to_text = []

          # set up two count variables
          total_matches_no_eos_ct = 0
          total_matches_with_eos_ct = 0

          with torch.no_grad(), \
              tqdm(total=num_test) as progress_bar:
              for batch_num, batch in enumerate(test_loader):
                  batch_size = len(batch["source_ids"])

                  # testing for loss fcn
                  loss, _ = forward(model, device, batch)     # loss, logits, but don't need logits
                  loss_meter.update(loss.item(), batch_size)  # loss.item() since it's a tensor

                  # predict / generate for token matches
                  src_ids = batch["source_ids"].to(device, dtype=torch.long)
                  src_mask = batch["source_mask"].to(device, dtype=torch.long)
                  tgt_ids = batch["target_ids"].to(device, dtype=torch.long)
                  # note you could tweak the generation params. See huggingface details for generate
                  generated_ids = model.generate(src_ids, attention_mask=src_mask, max_length=None, min_length=None)       # (batch x seq length) -------Added min length

                  # collect some stats
                  total_matches_no_eos, total_matches_with_eos, correct_indices = \
                      util.masked_token_match(tgt_ids, generated_ids, return_indices=True)
                  total_matches_no_eos_ct += total_matches_no_eos
                  total_matches_with_eos_ct += total_matches_with_eos

                  # save for qualitative analysis
                  orig_text_input, orig_text_output = batch["source_text"], batch["target_text"]
                  #TODO this could break once skip_special_tokens is fixed
                  outputs_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  #skip_special_tokens=False
                  preds = list(zip(orig_text_input, orig_text_output, outputs_decoded))
                  test_pred_list_all.extend(preds)

                  # ----------print-----------
                  print(outputs_decoded)
                  Test_prediction_list_for_write_to_text.extend(outputs_decoded)

                  # we also store only the correct indices
                  for idx in correct_indices.tolist():    # tensor to list; these are the valid indices
                      test_pred_list_correct.append(preds[idx[0]])     # each item was a list of one element

                  # print one batch of generations for qualitative assessment
                  if batch_num == 0:
                      for orig_input, orig_target, actual_output in preds[:1]:
                          log.info(f'Source: {orig_input}\t Target: {orig_target}\n'
                                  f'\t Actual: {actual_output}')

                  # Log info
                  progress_bar.update(batch_size)
                  progress_bar.set_postfix(NLL=loss_meter.avg)

          # save predictions for qualititative analysis
          util.save_preds(test_pred_list_all, record_dir, file_name="preds_all_test.csv")
          util.save_preds(test_pred_list_correct, record_dir, file_name="preds_correct_test.csv")
          results_list = [('NLL', loss_meter.avg),
                          ('exact_match_with_eos', total_matches_with_eos_ct),
                          ('exact_match_no_eos', total_matches_no_eos_ct)]
          results = OrderedDict(results_list)


          # Save predictions to a output file
          write_output_to_text(Test_prediction_list_for_write_to_text,save_predicted_test_output_text_dir)


          # Log to console
          results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
          log.info(f'test {results_str}')

    model.save_pretrained(save_weights_dir)
    tokenizer.save_pretrained(save_weights_dir)


# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())