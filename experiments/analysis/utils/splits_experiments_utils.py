import os
import json
from tqdm import tqdm
from pathlib import Path
from glob import glob
from collections import defaultdict
from comet_ml import API

COVR_DATASET_VERSION = "9"


def get_model_name_from_tags(tags):
    options = ['bart_large', 't5_large', 'bart', 't5', 'lstm']
    for option in sorted(options, reverse=True, key=len):
        if option in tags or option.replace('_', '-') in tags:
            return option
    raise ValueError()


def get_experiments(dataset, allow_from_cache=True, instance_source='source', split_method="template", lstm=False):
    comet_api = API(api_key=os.getenv("COMET_API_KEY"))
    experiments = []
    if dataset != 'covr':
        if split_method == "template":
            anon_level = 2 if dataset in ["smcalflow", "spider"] else 1
            experiments += comet_api.get_experiments(workspace='benbogin', project_name='analyze-comp-gen',
                                                    pattern=f"{dataset}.*split.*anon_{anon_level}.*")
    else:
        if split_method == "cfg":
            experiments = comet_api.get_experiments(workspace='benbogin', project_name='analyze-comp-gen',
                                                    pattern=r"covr - .* - split\(" + COVR_DATASET_VERSION)
    experiments += comet_api.get_experiments(workspace='benbogin', project_name='analyze-comp-gen',
                                             pattern=fr"{dataset} .* - split\(" + split_method + "/")

    experiments_results = defaultdict(lambda: defaultdict(dict))
    for exp in tqdm(experiments):
        if lstm and "lstm" not in exp.name:
            continue
        if not lstm and "lstm" in exp.name:
            continue
        cache_filename = f"cache/{exp.id}.json"
        need_reload = True
        if allow_from_cache and os.path.exists(cache_filename):
            exp_maybe = json.load(open(cache_filename, "rt"))
            if exp_maybe['metric_summary'] and 'instance_source' in exp_maybe:
                need_reload = False
                exp = exp_maybe
        if need_reload:
            tags = exp.get_tags()
            try:
                if exp.get_metadata()['running']:
                    continue
                exp_instance_source = exp.get_parameters_summary('dataset_reader.instance_source')
                exp = {
                    'tags': tags,
                    'split_path': exp.get_parameters_summary('dataset_reader.split_path')['valueCurrent'],
                    'random_seed': exp.get_parameters_summary('dataset_reader.sample_random_seed')['valueCurrent'],
                    'metric_summary': exp.get_metrics_summary(metric='best_validation_accuracy'),
                    'experiment_dir': exp.get_system_details()['logAdditionalSystemInfoList'][0]['value'],
                    'instance_source': exp_instance_source['valueCurrent'] if exp_instance_source else 'source'
                }
                if not exp['metric_summary']:
                    continue
            except Exception as e:
                print(e)
                continue
            json.dump(exp, open(cache_filename, "wt"))

        if exp['instance_source'] != instance_source:
            continue

        model_name = get_model_name_from_tags(exp['tags'])
        split_key = get_split_key_from_path(exp['split_path'], dataset)

        if not exp['metric_summary']:
            print("no metrics for", exp.get_name())
            continue

        experiments_results[model_name][split_key][exp['random_seed']] = {
            'accuracy': float(exp['metric_summary']['valueMax']),
            'epochs': exp['metric_summary']['stepCurrent'],
            'experiment_dir': exp['experiment_dir']
        }
    return experiments_results


def get_split_id_from_path(split_path, dataset):
    return split_path.split('/')[-3] + "_" + split_path.split('/')[-2]


def get_sub_dataset_from_path(p):
    return p.split('/')[-1].split('.json')[0]


def get_split_key_from_path(p, dataset):
    # fix old experiment names (we might still get these from old experiments drawn from comet)
    if '_anon' in p:
        p = p.replace('_anon_1', '').replace('_anon_2', '')
        p = p.replace('/split_', '/splits/template/split_')
    if 'splits/9' in p:
        p = p.replace('splits/9', 'splits/cfg')

    split_id = get_split_id_from_path(p, dataset)
    sub_dataset = get_sub_dataset_from_path(p)
    if sub_dataset != dataset:
        return f"{dataset}_{sub_dataset}_{split_id}"
    else:
        return f"{dataset}_{split_id}"


def get_splits(dataset, split_method='template'):
    split_files = glob(f'../../datasets/{dataset}/splits/{split_method}/*/*.json')
    split_files += glob(f'../../datasets/{dataset}/splits/{split_method}/*/*/*.json')

    splits_info = {}
    for fpath in split_files:
        if "/full/" in fpath:
            continue
        key = get_split_key_from_path(fpath, dataset)
        splits_info[key] = json.load(open(fpath, 'rt'))

    return splits_info


def get_examples_by_qid(dataset, base_dir="..", sub_dataset=None):
    # sub_dataset_string = sub_dataset if sub_dataset else "*"
    sub_dataset = sub_dataset if sub_dataset else dataset
    path_pattern = Path(base_dir) / f'datasets/{dataset}/{sub_dataset}.all.jsonl'
    print(path_pattern)
    datasets_files = glob(str(path_pattern))
    print(f'loading data from {datasets_files}')
    examples_by_qid = {}
    for fpath in datasets_files:
        examples = [json.loads(l) for l in tqdm(open(fpath, 'rt'))]

        for i, ex in enumerate(examples):
            qid = ex['qid']
            examples_by_qid[qid] = ex

    return examples_by_qid


def verify_split(train_programs, test_programs, return_culprit=False):
    """
    Verify that all tokens in test programs appear in train programs
    """
    def tokenize(t):
        t = t.replace("(", " ").replace(")", " ").replace(".", " ").replace(",", " ")
        return t.split()

    train_tokens = set()
    for program in train_programs:
        train_tokens.update(tokenize(program))
    for program in test_programs:
        if set(tokenize(program)).difference(train_tokens):
            if return_culprit:
                return False, set(tokenize(program)).difference(train_tokens)
            else:
                return False
    return True, None