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

languages = ["Albanian", "Assamese", "Chinese", "English", "Hindi", "Odia", "Sinhala", "Tamil", "Urdu"]

def create_dictionary(equation_file, mwp_file, seed_len_ratio, language):

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

    return combined_data


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

def dataset_split(data_dir, out_path, mwp_type, equation_type, seed_len_ratio, train_size=0.6, test_size=0.3):
  test_size = test_size/(1-train_size)


  X1_train = pd.DataFrame()
  Y1_train = pd.DataFrame()
  X1_valid = pd.DataFrame()
  Y1_valid = pd.DataFrame()
  X1_test = pd.DataFrame()
  Y1_test = pd.DataFrame()

  X2_train = pd.DataFrame()
  Y2_train = pd.DataFrame()
  X2_valid = pd.DataFrame()
  Y2_valid = pd.DataFrame()
  X2_test = pd.DataFrame()
  Y2_test = pd.DataFrame()

  for language in languages:


    datain_file = "{}-{}.txt".format(mwp_type, language)
    if(equation_type == 'word'):
        equation_file = "{}-{}-{}.txt".format(mwp_type, "Equations", language)
    else:
        equation_file = "{}-{}.txt".format(mwp_type, "Equations")

    equation_file = os.path.join(data_dir, equation_file)
    mwp_file = os.path.join(data_dir, datain_file)

    samples = create_dictionary(equation_file, mwp_file, seed_len_ratio, language)



    mwps = [sample['mwp'] for sample in samples]
    equations = [sample['equation'] for sample in samples]
    source_mwp = [sample['mwp_start'] for sample in samples]

    source1_df = pd.DataFrame(source_mwp, columns=["Source1"])
    target1_df = pd.DataFrame(mwps, columns=["Target1"])

    source2_df = pd.DataFrame(mwps, columns=["Source2"])
    target2_df = pd.DataFrame(equations, columns=["Target2"])


    x1_train, x1_rem, y1_train, y1_rem = train_test_split(target1_df, source1_df, train_size=train_size, random_state=1)
    x1_valid, x1_test, y1_valid, y1_test = train_test_split(x1_rem,y1_rem, test_size=test_size, random_state=1)

    x2_train, x2_rem, y2_train, y2_rem = train_test_split(target2_df, source2_df, train_size=train_size, random_state=1)
    x2_valid, x2_test, y2_valid, y2_test = train_test_split(x2_rem,y2_rem, test_size=test_size, random_state=1)

    X1_train = pd.concat([X1_train, x1_train])
    Y1_train = pd.concat([Y1_train, y1_train])
    X1_valid = pd.concat([X1_valid, x1_valid])
    Y1_valid = pd.concat([Y1_valid, y1_valid])
    X1_test = pd.concat([X1_test, x1_test])
    Y1_test = pd.concat([Y1_test, y1_test])


    X2_train = pd.concat([X2_train, x2_train])
    Y2_train = pd.concat([Y2_train, y2_train])
    X2_valid = pd.concat([X2_valid, x2_valid])
    Y2_valid = pd.concat([Y2_valid, y2_valid])
    X2_test = pd.concat([X2_test, x2_test])
    Y2_test = pd.concat([Y2_test, y2_test])


    # X1_train = X1_train.sample(frac=1, random_state=42)
    # Y1_train = Y1_train.sample(frac=1, random_state=42)

    # X1_valid = X1_valid.sample(frac=1, random_state=42)
    # Y1_valid = Y1_valid.sample(frac=1, random_state=42)
    # X1_test = X1_test.sample(frac=1, random_state=42)
    # Y1_test = Y1_test.sample(frac=1, random_state=42)

    # X2_train = X2_train.sample(frac=1, random_state=42)
    # Y2_train = Y2_train.sample(frac=1, random_state=42)

    # X2_valid = X2_valid.sample(frac=1, random_state=42)
    # Y2_valid = Y2_valid.sample(frac=1, random_state=42)
    # X2_test = X2_test.sample(frac=1, random_state=42)
    # Y2_test = Y2_test.sample(frac=1, random_state=42)


  np.savetxt(os.path.join(out_path, 'train.target1'), X1_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'train.source1'), Y1_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.target1'), X1_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.source1'), Y1_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.target1'), X1_test.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.source1'), Y1_test.values, fmt='%s')

  np.savetxt(os.path.join(out_path, 'train.target2'), X2_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'train.source2'), Y2_train.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.target2'), X2_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'val.source2'), Y2_valid.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.target2'), X2_test.values, fmt='%s')
  np.savetxt(os.path.join(out_path, 'test.source2'), Y2_test.values, fmt='%s')


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
    output_dataset_path = os.path.join(args.dataset_dir, args.experiment, str(args.seed_len))
    Path(output_dataset_path).mkdir(parents=True, exist_ok=True)

    # generate_seeded_dataset(input_file, output_file, args.seed_len)
    # Create train-val-test split
    dataset_split(args.data_dir, output_dataset_path, args.mwp_type, args.equation_type , args.seed_len, train_size=args.train_split, test_size=args.test_split)

# If calling script then execute
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {0}".format(e))
        print(traceback.format_exc())