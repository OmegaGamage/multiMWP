import os
import argparse
import random
import time

from collections import OrderedDict
from operator import itemgetter

import pandas as pd

from datasets import load_metric

#Load metrics from huggingface library

bleu = load_metric("bleu")
# bleurt = load_metric("bleurt")
bleurt = load_metric('bleurt', 'bleurt-large-512')
rouge = load_metric('rouge')
meteor = load_metric('meteor')
sacrebleu = load_metric("sacrebleu")
bertscore = load_metric("bertscore")

from rouge_score import rouge_scorer
rouge_scorer2 = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# indic_glue_metric = load_metric('indic_glue', 'wnli3')  # 'wnli' or any of ["copa", "sna", "csqa", "wstp", "inltkh", "bbca", "iitp-mr", "iitp-pr", "actsa-sc", "md"]


lang_dict = {
    "English" : "en",
    "Sinhala" : "si",
    "Tamil" : "ta",
    "Urdu" : "ur",
    "Odia" : "or",
    "Hindi" : "hi",
    "Assamese" : "as",
    "Albanian" : "al",
}

bertscore_supported_langs = ['en', ]
def get_sentence_bleu_score(reference, candidate, n=4):
    reference = reference.strip().split()
    candidate = candidate.strip().split()

    if (len(reference)!=0 and len(candidate)!=0):
        score = bleu.compute(predictions=[candidate], references=[[reference]])["bleu"]
    elif(len(reference) == len(candidate)):
        score = 1.0
    else:
        score = 0.0
    return score

def get_bleu_score(ref_cand):
    ref_cand = [ [a.split()   for a in b] for b in ref_cand]

    references = ref_cand[0]
    references = [[r] for r in references]
    predictions = ref_cand[1]

    results = bleu.compute(predictions=predictions, references=references)
    # bleu_score = results["bleu"]
    bleu_score = results['precisions'][0]
    print("Finish Blue score....")
    return bleu_score

def get_sentence_rouge_score(reference, candidate, metric):
    # print("n = ", n)
    if n !=  "1" and n != "2" and n != "L":
        print("ERROR: Invalid n for rouge score")
        return -1.0
    if metric !=  "recall" and metric != "fmeasure" and metric != "precision":
        print("ERROR: Invalid n for rouge score")
        return -1.0

    results = rouge.compute(predictions=[candidate], references=[reference])

    return results['rouge{}'.format(n)].mid.fmeasure

def get_rouge_score(ref_cand, metric):

    references = ref_cand[0]
    predictions = ref_cand[1]

    results = rouge.compute(predictions=predictions, references=references)
    
    if (metric == 'fmeasure'):
        rouge_score = (results['rouge1'].mid.fmeasure, results['rouge2'].mid.fmeasure, results['rougeL'].mid.fmeasure)
    if (metric == 'precision'):
        rouge_score = (results['rouge1'].mid.precision, results['rouge2'].mid.precision, results['rougeL'].mid.precision)
    if (metric == 'recall'):
        rouge_score = (results['rouge1'].mid.recall, results['rouge2'].mid.recall, results['rougeL'].mid.recall)

    return rouge_score


def get_sentence_rouge_score2(reference, candidate, metric):

    results = rouge_scorer2.score(reference, candidate)

    if (metric == 'fmeasure'):
        return (results['rouge1'].fmeasure,results['rouge2'].fmeasure,results['rougeL'].fmeasure)
    if (metric == 'precision'):
        return (results['rouge1'].precision,results['rouge2'].precision,results['rougeL'].precision)
    if (metric == 'recall'):
        return (results['rouge1'].recall,results['rouge2'].recall,results['rougeL'].recall)

def get_rouge_score2(ref_cand, metric):

    if metric !=  "recall" and metric != "fmeasure" and metric != "precision":
        print("ERROR: Invalid n for rouge score")
        return -1.0
    cum_rouge_score_1 = 0
    cum_rouge_score_2 = 0
    cum_rouge_score_l = 0

    for i in range(len(ref_cand[0])):
        rouge_scores = get_sentence_rouge_score2(ref_cand[0][i], ref_cand[1][i], metric)

        cum_rouge_score_1 += rouge_scores[0]
        cum_rouge_score_2 += rouge_scores[1]
        cum_rouge_score_l += rouge_scores[2]

    cum_rouge_score_1 /= len(ref_cand[0])
    cum_rouge_score_2 /= len(ref_cand[0])
    cum_rouge_score_l /= len(ref_cand[0])

    return (cum_rouge_score_1, cum_rouge_score_2, cum_rouge_score_l)

def get_sentence_meteor_score(reference, candidate):
    results = meteor.compute(predictions=candidate, references=reference)
    return results["meteor"]

def get_meteor_score(ref_cand):
    cum_meteor_score = 0

    references = ref_cand[0]
    predictions = ref_cand[1]
    results = meteor.compute(predictions=predictions, references=references)
    meteor_score_2 = results["meteor"]
    print("meteor_score_2: ",meteor_score_2)
    return round(meteor_score_2,4)

def get_sentence_sacrebleu_score(reference, candidate):
    results = sacrebleu.compute(predictions=candidate, references=reference)
    return results["score"]

def get_sacrebleu_score(ref_cand):
    references = ref_cand[0]
    references = list(filter(None, references))
    references = [[r] for r in references]
    predictions = ref_cand[1]
    # predictions = list(filter(None, predictions))

    print("len(references): ",len(references))
    print("len(predictions): ",len(predictions))

    results = sacrebleu.compute(predictions=predictions, references=references)
    sacrebleu_score = results['score']

    return round(sacrebleu_score,1)


def get_sentence_bleurt_score(reference, candidate):
    results = bleurt.compute(predictions=candidate, references=reference)
    return results['scores'][0]

def get_bleurt_score(ref_cand):

    references = ref_cand[0]
    predictions = ref_cand[1]

    results = bleurt.compute(predictions=predictions, references=references)

    bleurt_score = results['scores']
    bleurt_score = sum(bleurt_score) / len(bleurt_score)

    return bleurt_score


def get_bertscore(ref_cand, lang, bertscore_model):
    references = ref_cand[0]
    predictions = ref_cand[1]

    results = bertscore.compute(predictions=predictions, references=references, lang=lang, model_type= bertscore_model)
    results_f1 = results['f1']
    results_f1 = sum(results_f1)/len(results_f1)

    return results_f1


def get_metrics(experiment_info, bertscore_model):

    print("\n--------------------------------\n")
    print(experiment_info["results_path"])
    is_dir_exist = os.path.isdir(experiment_info["results_path"])
    if(is_dir_exist):
        print("Path exist")
    else:
        print("ERROR : Path does not exist")

    preds_all_eval_file = os.path.join(experiment_info["results_path"],"preds_all_eval.csv")
    preds_all_eval = open(preds_all_eval_file).read().splitlines()
    preds_all_eval = [record.split("|") for record in preds_all_eval]

    for i in range(len(preds_all_eval)):
        if(len(preds_all_eval[i]) == 4):
            if(experiment_info["language"] == "Albanian"):
                preds_all_eval[i][1] = preds_all_eval[i][1] + "|" + preds_all_eval[i][2]
                del preds_all_eval[i][2]

            if(experiment_info["language"] == "Assamese"):
                preds_all_eval[i][1] = preds_all_eval[i][1] + "|"
                del preds_all_eval[i][2]
            if(experiment_info["language"] == "Hindi"):
                preds_all_eval[i][1] = preds_all_eval[i][1] + "|"
                del preds_all_eval[i][2]
            if(experiment_info["language"] == "Odia"):
                preds_all_eval[i][1] = preds_all_eval[i][1] + "|" + preds_all_eval[i][2]
                del preds_all_eval[i][2]
        if(len(preds_all_eval[i]) == 5):

            if(experiment_info["language"] == "Odia"):
                preds_all_eval[i][1] = preds_all_eval[i][1] + "|" + preds_all_eval[i][2]
                del preds_all_eval[i][2]
                preds_all_eval[i][2] = preds_all_eval[i][2] + "|" + preds_all_eval[i][3]
                del preds_all_eval[i][3]

    preds_all_eval = [ val for val in preds_all_eval if ( val[0] != "" and val[1] != "" and val[2] != "")]
    preds_all_eval = [record[1:] for record in preds_all_eval ]

    preds_all_eval = list(zip(*preds_all_eval))
    preds_all_eval = [list(p) for p in preds_all_eval]
    # preds_all_eval = [ [a.split()   for a in b] for b in preds_all_eval]


    preds_all_test_file = os.path.join(experiment_info["results_path"],"preds_all_test.csv")
    preds_all_test = open(preds_all_test_file).read().splitlines()
    preds_all_test = [record.split("|") for record in preds_all_test]


    for i in range(len(preds_all_test)):
        if(len(preds_all_test[i]) == 4):
            if(experiment_info["language"] == "Albanian"):
                preds_all_test[i][1] = preds_all_test[i][1] + "|" + preds_all_test[i][2]
                del preds_all_test[i][2]

            if(experiment_info["language"] == "Assamese"):
                preds_all_test[i][1] = preds_all_test[i][1] + "|"
                del preds_all_test[i][2]
            if(experiment_info["language"] == "Hindi"):
                preds_all_test[i][1] = preds_all_test[i][1] + "|"
                del preds_all_test[i][2]
            if(experiment_info["language"] == "Odia"):
                preds_all_test[i][1] = preds_all_test[i][1] + "|" + preds_all_test[i][2]
                del preds_all_test[i][2]
        if(len(preds_all_test[i]) == 5):

            if(experiment_info["language"] == "Odia"):
                preds_all_test[i][1] = preds_all_test[i][1] + "|" + preds_all_test[i][2]
                del preds_all_test[i][2]
                preds_all_test[i][2] = preds_all_test[i][2] + "|" + preds_all_test[i][3]
                del preds_all_test[i][3]


    preds_all_test = [ val for val in preds_all_test if ( val[0] != "" and val[1] != "" and val[2] != "")]
    preds_all_test = [record[1:] for record in preds_all_test ]
    preds_all_test = list(zip(*preds_all_test))
    preds_all_test = [list(p) for p in preds_all_test]
    # preds_all_test = [ [a.split()   for a in b] for b in preds_all_test]

    print("Getting Blue score....")
    experiment_info["eval BLEU"] = get_bleu_score(preds_all_eval)

    print("Getting ROUGE score....")
    for metric in ["fmeasure", "precision", "recall"]:
        temp_results = get_rouge_score(preds_all_eval, metric)
        experiment_info["eval ROUGE-{}-{}".format("1",metric)] = temp_results[0]
        experiment_info["eval ROUGE-{}-{}".format("2",metric)] = temp_results[1]
        experiment_info["eval ROUGE-{}-{}".format("L",metric)] = temp_results[2]

    print("Getting ROUGE score2....")
    for metric in ["fmeasure", "precision", "recall"]:
        temp_results = get_rouge_score2(preds_all_eval, metric)
        experiment_info["eval ROUGE2-{}-{}".format("1",metric)] = temp_results[0]
        experiment_info["eval ROUGE2-{}-{}".format("2",metric)] = temp_results[1]
        experiment_info["eval ROUGE2-{}-{}".format("L",metric)] = temp_results[2]

    print("Getting METEOR score....")
    experiment_info["eval METEOR"] = get_meteor_score(preds_all_eval) #Working

    print("Getting BLEURT score....")
    experiment_info["eval BLEURT"] = get_bleurt_score(preds_all_eval)

    print("Getting SCAREBLEU score....")
    experiment_info["eval SCAREBLEU"] = get_sacrebleu_score(preds_all_eval) #Working


    print("Getting BERTScore....")
    experiment_info["eval BERTSCORE "] = get_bertscore(preds_all_eval, lang_dict[experiment_info["language"]], bertscore_model) #Working


    print("Getting test Blue score....")
    experiment_info["test BLEU"] = get_bleu_score(preds_all_test) #Working

    print("Getting test ROUGE score....")
    for metric in ["fmeasure", "precision", "recall"]:
        temp_results = get_rouge_score(preds_all_eval, metric)
        experiment_info["test ROUGE-{}-{}".format("1",metric)] = temp_results[0]
        experiment_info["test ROUGE-{}-{}".format("2",metric)] = temp_results[1]
        experiment_info["test ROUGE-{}-{}".format("L",metric)] = temp_results[2]

    print("Getting test ROUGE score2....")
    for metric in ["fmeasure", "precision", "recall"]:
        temp_results = get_rouge_score2(preds_all_eval, metric)
        experiment_info["test ROUGE2-{}-{}".format("1",metric)] = temp_results[0]
        experiment_info["test ROUGE2-{}-{}".format("2",metric)] = temp_results[1]
        experiment_info["test ROUGE2-{}-{}".format("L",metric)] = temp_results[2]


    print("Getting test METEOR score....")
    experiment_info["test METEOR"] = get_meteor_score(preds_all_test) #Working

    print("Getting test BLEURT score....")
    experiment_info["test BLEURT"] = get_bleurt_score(preds_all_test)

    print("Getting test SCAREBLEU score....")
    experiment_info["test SCAREBLEU"] = get_sacrebleu_score(preds_all_test) #Working

    print("Getting BERTScore....")
    experiment_info["test BERTSCORE "] = get_bertscore(preds_all_eval, lang_dict[experiment_info["language"]], bertscore_model) #Working

    del experiment_info['results_path']
    # del experiment_info['data_path']
    return experiment_info

def main():

    parser = argparse.ArgumentParser(
        description="Running main script for calculating metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir',
                        required=True,
                        type=str,
                        help="Path to model output save directory",
                        default="data")

    parser.add_argument('--output_file',
                        required=True,
                        type=str,
                        help="Path to metric outputs",
                        default="data")

    parser.add_argument('--bertscore_model',
                        required=False,
                        type=str,
                        help="model to use in bertscore",
                        default="xlm-roberta-large")

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


    save_dir = args.save_dir
    output_file = args.output_file

    # experiments_dirs = os. listdir(save_dir)
    experiments_dirs = [dirpaths for dirpaths, dirnames, filenames in os.walk(save_dir) if not dirnames]

    experiment_info=[]

    start = time.time()
    for experiments_dir in experiments_dirs:
        experiment = experiments_dir.split("/")[-1]
        print("INFO: Running experiment {} ...".format(experiment))
        # model_name, language, mwp_type,train_split,test_split, seed_len, experiment_id = experiment.split("_")

        if(not os.path.exists( os.path.join(experiments_dir, "preds_all_eval.csv"))):
            continue

        if(not os.path.exists( os.path.join(experiments_dir, "preds_all_test.csv"))):
            continue

        model_name, language, mwp_type,train_split, seed_len, experiment_id = experiment.split("|")
        train_split,test_split= train_split.split("_")
        seed_len = seed_len.split()[0]
        # experiment_id = experiment_id.split("-")[0]
        test_summary =OrderedDict()
        test_summary["model"] = model_name
        test_summary["experiment"] =experiment
        test_summary["language"] = language
        test_summary["mwp_type"] = mwp_type
        test_summary["seed_len"] = float(seed_len)
        test_summary["experiment_id"] = experiment_id
        test_summary["train_split"] = int(train_split)
        test_summary["test_split"] = int(test_split)
        # test_summary["data_path"] = os.path.join(data_dir, language, experiment_id, seed_len)
        test_summary["results_path"] = experiments_dir

        experiment_info.append(test_summary)

    # experiment_info = sorted(experiment_info, key=lambda d: d['model'])
    experiment_info = sorted(experiment_info, key=itemgetter('model', 'language', 'mwp_type'))

    end = time.time()
    print("INFO: Elapsed time(pre=processing) {}....".format(end - start))

    print(experiment_info)

    print("------------------------------------------------")
    frames = []
    for i, experiment in enumerate(experiment_info):
        print("Prossing the file: {}".format(experiment['experiment']))
        start = time.time()
        ord_dict = get_metrics(experiment, args.bertscore_model)
        print(ord_dict)
        df = pd.DataFrame(ord_dict, index=[i])
        frames.append(df)
        end = time.time()
        print("INFO: Elapsed time {}....".format(end - start))


    print("Started pandas post-processing....")
    start = time.time()
    eval_table = pd.concat(frames)

    eval_table.to_csv(output_file, index=False)

    end = time.time()
    print("INFO: Elapsed time {}....".format(end - start))

# If calling script then execute
if __name__ == "__main__":
    main()