import os
import json
import argparse
import traceback
from pathlib import Path
from collections import OrderedDict

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
    "zh" : "Chinese"
}


def create_json_file(equation_file, mwp_file, output_file, seed_len_ratio, language):

    equations = open(equation_file).read().splitlines()
    mwps = open(mwp_file).read().splitlines()

    num_eqs = len(equations)
    num_mwps = len(mwps)
    assert num_eqs == num_mwps , f"Lengths don't match"

    combined_data=[]

    if language == "Chinese":
        for i in range(len(equations)):
            od = OrderedDict()
            od['id'] = i
            od['mwp'] = mwps[i]
            od['equation'] = equations[i]
            mwp_temp = mwps[i].strip()
            source_line = jieba.lcut(mwp_temp)
            od['mwp_start'] = "".join(source_line[:round(len(source_line) * seed_len_ratio)])

            combined_data.append(od)
    else:

        for i in range(len(equations)):
            od = OrderedDict()
            od['id'] = i
            od['mwp'] = mwps[i]
            od['equation'] = equations[i]
            mwp_words = mwps[i].split()
            od['mwp_start'] = " ".join(mwp_words[:round(len(mwp_words) * seed_len_ratio)])

            combined_data.append(od)

    with open(output_file, "w") as outfile:
        json.dump(combined_data, outfile)

def dataset_split(json_file, out_path, train_size=0.6, test_size=0.3):
  test_size = test_size/(1-train_size)

  with open(json_file) as jf:
    samples = json.load(jf)

  mwps = [sample['mwp'] for sample in samples]
  equations = [sample['equation'] for sample in samples]
  source_mwp = [sample['mwp_start'] for sample in samples]

  source1_df = pd.DataFrame(source_mwp, columns=["Source1"])
  target1_df = pd.DataFrame(mwps, columns=["Target1"])

  source2_df = pd.DataFrame(mwps, columns=["Source2"])
  target2_df = pd.DataFrame(equations, columns=["Target2"])


  X1_train, X1_rem, y1_train, y1_rem = train_test_split(target1_df, source1_df, train_size=train_size, random_state=1)
  X1_valid, X1_test, y1_valid, y1_test = train_test_split(X1_rem,y1_rem, test_size=test_size, random_state=1)

  X2_train, X2_rem, y2_train, y2_rem = train_test_split(target2_df, source2_df, train_size=train_size, random_state=1)
  X2_valid, X2_test, y2_valid, y2_test = train_test_split(X2_rem,y2_rem, test_size=test_size, random_state=1)

  np.savetxt(os.path.join(out_path, 'train.target1'), X1_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'train.source1'), y1_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.target1'), X1_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.source1'), y1_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.target1'), X1_test.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.source1'), y1_test.values, fmt='%s')

  np.savetxt(os.path.join(out_path, 'train.target2'), X2_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'train.source2'), y2_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.target2'), X2_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.source2'), y2_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.target2'), X2_test.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.source2'), y2_test.values, fmt='%s')


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
    parser.add_argument('--equation_file',
                        required=False,
                        type=str,
                        help="Name of the file containing equations",
                        default="Simple-Equations.txt")
    parser.add_argument('--language',
                        help='Language of the model. Default: {}'.format('en'),
                        choices=['en', 'si', 'ta', 'as', 'ur','hi', 'or', 'al', 'zh'],
                        default='en')
    parser.add_argument('--mwp_type',
                        help='Language of the model.',
                        choices=['Simple', 'Algebraic', 'Combine'],
                        default='Simple')
    parser.add_argument('--equation_type',
                        help='Equations are word or algebraic.',
                        choices=['word', 'algebraic'],
                        default='algebraic')
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
    json_file = "{}-{}-{}.json".format(args.mwp_type, language, args.seed_len)
    json_output_file = os.path.join(output_dataset_path, json_file)


    if(args.equation_type == 'word'):
        equation_file = "{}-{}-{}.txt".format(args.mwp_type, "Equations", language)
    else:
        equation_file = "{}-{}.txt".format(args.mwp_type, "Equations")

    equation_file = os.path.join(args.data_dir, equation_file)
    mwp_file = os.path.join(args.data_dir, datain_file)
    # Generate inputs and labels corresponding to model anhd dataset
    create_json_file(equation_file, mwp_file, json_output_file, args.seed_len,language)

    # generate_seeded_dataset(input_file, output_file, args.seed_len)
    # Create train-val-test split
    dataset_split(json_output_file, output_dataset_path, train_size=args.train_split, test_size=args.test_split)

# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())