import os
import argparse
import traceback
from pathlib import Path
from math import ceil

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#Chinese word segementer
import jieba

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
    "bn" : "Bengali",
    "fa" : "Persian",
    "ar" : "Arabic",
    "ne" : "Nepali",
    "dv" : "Dhivehi"
}


def dataset_split(file1, file2, out_path, train_size=0.6, test_size=0.3):
  test_size = test_size/(1-train_size)
  target,source = [],[]
  target = open(file1).read().splitlines()
  target = list(filter(None, target))
  source = open(file2).read().splitlines()
  source = list(filter(None, source))

  print("len(source): ",len(source))
  print("len(target): ",len(target))

  target_df = pd.DataFrame(target,columns=["Target"])
  source_df = pd.DataFrame(source,columns=["Source"])

  X_train, X_rem, y_train, y_rem = train_test_split(target_df,source_df, train_size=train_size, random_state=1)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=test_size, random_state=1)

  # print(X_train.shape), print(y_train.shape)
  # print(X_valid.shape), print(y_valid.shape)
  # print(X_test.shape), print(y_test.shape)

  np.savetxt(os.path.join(out_path, 'train.target'), X_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'train.source'), y_train.values, fmt='%s')

  np.savetxt(os.path.join(out_path, 'val.target'), X_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.source'), y_valid.values, fmt='%s')

  np.savetxt(os.path.join(out_path, 'test.target'), X_test.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.source'), y_test.values, fmt='%s')

def generate_seeded_dataset(input_file, output_file, ratio, language):
    if language == "Chinese":
        with open(input_file, 'r') as f_target, open(output_file, 'w') as f_source:
            for target_line in f_target:
                source_line = target_line.strip()
                source_line = jieba.lcut(source_line)
                source_line = "".join(source_line[:round(len(source_line) * ratio)])
                f_source.write('%s\n' % source_line)

    else:
        with open(input_file, 'r') as f_target, open(output_file, 'w') as f_source:
            for target_line in f_target:
                source_line = target_line.strip().split()
                source_line = " ".join(source_line[:round(len(source_line) * ratio)])
                f_source.write('%s\n' % source_line)


def main():
    parser = argparse.ArgumentParser(
        description="Running main script for Multilingual Math Word Problem generator ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir',
                        required=True,
                        type=str,
                        help="Path to original dataset directory",
                        default="data")
    parser.add_argument('--dataset_dir',
                        required=True,
                        type=str,
                        help="Path to prepared dataset directory",
                        default="./")
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
    parser.add_argument('--train_split',
                        type=float,
                        help="Training split. Default: {}".format(0.6),
                        default=0.6)
    parser.add_argument('--test_split',
                        type=float,
                        help="Test split. Default: {}".format(0.3),
                        default=0.3)
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

    print("INFO: Train-Split: {}, Test-Split: {}".format(args.train_split, args.test_split))
    # Create directory to store processed data
    output_dataset_path = os.path.join(args.dataset_dir,  language, args.experiment, str(args.seed_len))
    Path(output_dataset_path).mkdir(parents=True, exist_ok=True)

    datain_file = "{}-{}.txt".format(args.mwp_type, language)
    dataout_file = "target_{}_{}_{}.txt".format(args.experiment, args.seed_len, language)
    input_file = os.path.join(args.data_dir, datain_file)
    output_file = os.path.join(output_dataset_path, dataout_file)

    # Generate inputs and labels corresponding to model anhd dataset
    generate_seeded_dataset(input_file, output_file, args.seed_len, language)
    # Create train-val-test split
    dataset_split(input_file, output_file, output_dataset_path, train_size=args.train_split, test_size=args.test_split)

# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())