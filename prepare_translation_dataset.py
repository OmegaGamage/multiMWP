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



def dataset_split(
    tgt_lang, mwp_type, seed_len, experiment, data_dir, dataset_dir, train_size=0.6, test_size=0.3, src_lang = "English"
):
  test_size = test_size/(1-train_size)

  output_dataset_path = os.path.join(dataset_dir, tgt_lang, experiment,str(seed_len), "translation")
  Path(output_dataset_path).mkdir(parents=True, exist_ok=True)

  src_file = os.path.join(data_dir, "{}-{}.txt".format(mwp_type, src_lang))
  source = open(src_file).read().splitlines()
  source = list(filter(None, source))

  tgt_file = os.path.join(data_dir, "{}-{}.txt".format(mwp_type, tgt_lang))
  target = open(tgt_file).read().splitlines()
  target = list(filter(None, target))

  print("len(source): ",len(source))
  print("len(target): ",len(target))

  target_df = pd.DataFrame(target,columns=["Target"])
  source_df = pd.DataFrame(source,columns=["Source"])

  X_train, X_rem, y_train, y_rem = train_test_split(source_df,target_df, train_size=train_size, random_state=1)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=test_size, random_state=1)


  np.savetxt(os.path.join(output_dataset_path, 'train.source'), X_train.values, fmt='%s')
  np.savetxt(os.path.join(output_dataset_path, 'train.target'), y_train.values, fmt='%s')

  np.savetxt(os.path.join(output_dataset_path, 'val.source'), X_valid.values, fmt='%s')
  np.savetxt(os.path.join(output_dataset_path, 'val.target'), y_valid.values, fmt='%s')

  np.savetxt(os.path.join(output_dataset_path, 'test.source'), X_test.values, fmt='%s')
  np.savetxt(os.path.join(output_dataset_path, 'test.target'), y_test.values, fmt='%s')


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

    # Generate inputs and labels corresponding to model anhd dataset
    dataset_split(
        language,
        args.mwp_type,
        args.seed_len,
        args.experiment,
        args.data_dir,
        args.dataset_dir,
        train_size=args.train_split,
        test_size=args.test_split,
    )

# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())