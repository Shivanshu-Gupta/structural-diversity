# create program splits
# python datasets/scripts/split_datasets.py --base_dir '' --dataset 'covr' --split-method template
# python datasets/scripts/split_datasets.py --base_dir '' --dataset 'covr' --split-method template --percentage_programs_test 0.5 --limit-test 5000


import json
import os, sys
import typer
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path({
    'ben': '',
    'shivanshu': ''
}[os.getenv('RUNNER')])
sys.path.insert(0, BASE_DIR.absolute().__str__())

from data_scripts.utils.anon_fns import *
from data_scripts.splitters.split_by_bigrams import splits_by_bigrams
from data_scripts.splitters.split_by_template import splits_by_template
from experiments.analysis.utils.splits_experiments_utils import get_examples_by_qid

app = typer.Typer()

def load_dataset_old(ds):
    if ds == "overnight":
        output = []
        overnight_sub_datasets = [
            'basketball', 'blocks', 'calendar', 'calendarplus', 'geo880', 'housing',
            'publications', 'recipes', 'regex', 'restaurants', 'socialnetwork']
        for sub_dataset in overnight_sub_datasets:
            output.append((get_examples_by_qid(ds, base_dir=BASE_DIR, sub_dataset=sub_dataset), sub_dataset))

        return output
    else:
        return [(get_examples_by_qid(ds, base_dir=BASE_DIR), ds)]

def load_dataset(dataset, pool_type):
    from glob import glob
    path_pattern = Path(BASE_DIR) / f'datasets/{dataset}/{dataset}-{pool_type}.jsonl'
    print(path_pattern)
    datasets_files = glob(str(path_pattern))
    print(f'loading data from {datasets_files}')
    examples_by_qid = {}
    for fpath in datasets_files:
        examples = [json.loads(l) for l in open(fpath, 'rt')]

        for i, ex in enumerate(examples):
            qid = ex['qid']
            examples_by_qid[qid] = ex

    return [(examples_by_qid, dataset)]

def iid_splits(examples_by_qid, n_splits, limit_train=None, limit_test=None):
    from sklearn.model_selection import train_test_split
    splits = []
    for seed in range(n_splits):
        trn_qids, tst_qids = train_test_split(
            list(examples_by_qid.keys()),
            train_size=limit_train, test_size=limit_test,
            random_state=seed)
        split = dict(
            train=trn_qids,
            test=tst_qids,
            split_name=f'split_{seed}')
        splits.append(split)
    return splits

def subtree_splits(examples_by_qid, ds_name, n_splits, limit_train=None, limit_test=None):
    import pandas as pd
    from data_scripts.subsample_pool import get_subtrees, get_templates, iterative_subsample
    trn_pool = pd.DataFrame([ex for ex in examples_by_qid.values()])
    subtree2ex, ex2subtree, _ = get_subtrees(
        trn_pool, ds_name, max_size=4, context_type='', anon=False)
    template2ex, ex2template, _ = get_templates(trn_pool)
    ex2template = {ex: list(d.keys())[0] for ex, d in ex2template.items()}

    if not isinstance(limit_test, int): limit_test = int(len(ex2subtree) * limit_test)
    splits = []
    for seed in range(n_splits):
        tst_qids = iterative_subsample(
            '5_cyc_fix', subtree2ex, ex2subtree, None, template2ex, ex2template,
            n_trn=limit_test, seed=seed, ex_sel='mf_new_template', show_progress=True)
        trn_qids = [qid for qid in examples_by_qid if qid not in set(tst_qids)]
        split = dict(
            train=trn_qids,
            test=tst_qids,
            split_name=f'split_{seed}')
        splits.append(split)
    return splits

def nosplit_splits(examples_by_qid, ds_name):
    import pandas as pd
    trn_pool = pd.DataFrame([ex for ex in examples_by_qid.values()])
    split = dict(
        train=list(examples_by_qid.keys()),
        test=[],
        split_name='')
    return [split]


ANON_FN_MAP = {
    'covr': anonymize_covr_target,
    'covr_scfg_gen': anonymize_covr_target,
    'overnight': anonymize_overnight_target,
    'atis': anonymize_atis_target,
    'thingtalk': anonymize_thingtalk,
    'smcalflow': anonymize_smcalflow_target,
    'smcalflow-nostr': anonymize_smcalflow_target
}

@app.command()
def split(dataset: str = 'covr_scfg_gen', percentage_programs_test: float = 0.2,
          n_random_seeds: int = 5, split_method: str = 'template',
          limit_train: float = None, limit_test: float = None):
    if split_method == 'bigram' and n_random_seeds > 1:
        n_random_seeds = 1
        print("Only one random seed will be used since there is no randomness in bigram split method")
    if limit_train is not None and int(limit_train) == limit_train:
        limit_train = int(limit_train)
    if limit_test is not None and int(limit_test) == limit_test:
        limit_test = int(limit_test)

    loaded_datasets = load_dataset_old(dataset)
    # loaded_datasets = load_dataset(dataset, pool_type)

    anon_fn = ANON_FN_MAP[dataset.split('_')[0]]
    for example_by_qid, ds_name in loaded_datasets:
        for ex in example_by_qid.values():
            ex['anonymized_target'] = anon_fn(ex['target'])

        if split_method == 'template':
            splits = splits_by_template(
                example_by_qid,
                anon_fn,
                n_random_seeds,
                percentage_programs_test,
                limit_train=limit_train, limit_test=limit_test
            )
        elif split_method == 'bigram':
            splits = splits_by_bigrams(
                example_by_qid, limit_train=limit_train, limit_test=limit_test,
                add_siblings=False, p_similar_bigrams=0.5, easiness_threshold=1
            )
        elif split_method == 'iid':
            splits = iid_splits(example_by_qid, n_random_seeds, limit_train, limit_test)
        elif split_method == 'subtree':
            splits = subtree_splits(example_by_qid, ds_name, n_random_seeds, limit_train, limit_test)
        elif split_method == 'nosplit':
            splits = nosplit_splits(example_by_qid, ds_name)
        else:
            raise ValueError(f'Unknown split method: {split_method}')

        for split in splits:
            if ds_name == dataset:
                split_output_path = BASE_DIR / f"datasets/{dataset}/splits/{split_method}" / split['split_name']
            else: # subdataset
                split_output_path = BASE_DIR / f"datasets/{dataset}/{ds_name}/splits/{split_method}" / split['split_name']
            # split_output_path = BASE_DIR / f"datasets/{dataset}/splits/{pool_type}-{split_method}" / split['split_name']

            print(split_output_path)
            os.makedirs(split_output_path, exist_ok=True)
            # os.makedirs(split_output_path / "full", exist_ok=True)

            with open(split_output_path / f"{ds_name}.json", "wt") as f_out:
                json.dump(split, f_out)

            with open(split_output_path / f"{ds_name}.train.jsonl", "wt") as f_out:
                for ex_id in split['train']:
                    f_out.write(json.dumps(example_by_qid[ex_id]) + "\n")

            with open(split_output_path / f"{ds_name}.test.jsonl", "wt") as f_out:
                for ex_id in split['test']:
                    f_out.write(json.dumps(example_by_qid[ex_id]) + "\n")

if __name__ == "__main__":
    app()