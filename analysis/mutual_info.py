from ast import Sub
from genericpath import exists
from logging import root
import os, sys
import json
from unittest import result
import typer
import numpy as np
import pandas as pd

from itertools import product
from functools import cache
from pathlib import Path
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

from dotenv import load_dotenv
load_dotenv()
BASE_DIR = Path({
    'ben': '../..',
    'shivanshu': ''
}[os.getenv('RUNNER')])
sys.path.insert(0, BASE_DIR.absolute().__str__())

from utils.progress import track
from data_scripts.subsample.subsamples import get_subsample_params_single_split
from data_scripts.subsample_pool import SubsampleParams, get_data

app = typer.Typer()

def get_subtrees(trn_pool, ds_name, max_size, context_type, anon):
    from generation.scfg.utils import target_to_ast, target_to_ast_calflow, get_subtrees as get_subtrees_fn, tuple_to_tree

    if ds_name == 'smcalflow':
        target_to_ast = target_to_ast_calflow
    # trees = [target_to_ast(target) for target in trn_pool.target]
    all_subtrees = set()
    subtree2ex = defaultdict(Counter)
    ex2subtree = {}
    targets = trn_pool.anonymized_target if anon else trn_pool.target
    for qid, target in track(zip(trn_pool.qid, targets),
                            total=trn_pool.shape[0],
                            description="Extracting subtrees",
                            disable=True):
        tree = target_to_ast(target)
        _, ex_subtrees = get_subtrees_fn(tree, max_size, context_type)
        ex2subtree[qid] = {st: 1 for st in ex_subtrees}
        for subtree in ex_subtrees:
            all_subtrees.add(subtree)
            subtree2ex[subtree][qid] += 1
    # print(f'{len(all_subtrees)} subtrees with max size {max_size} and context type {context_type} and anon={anon}')
    # all_subtrees = [tuple_to_tree(st) for st in all_subtrees]
    return dict(subtree2ex), ex2subtree, None

def pool_mutual_info(trnset, comp2ex, return_all=False, show_progress=False):
    bags = [set(ex_d.keys()) for ex_d in comp2ex.values()]  # compounds as bags of instances
    mutual_info_arr = np.zeros((len(bags), len(bags)))
    total_entropy = 0
    for i in track(range(len(bags)), description="Computing pairwise MIs...",
                   disable=not show_progress):
        bag = bags[i]
        if len(bag) > 0:
            # p = np.array([len(trnset) - len(bag), len(bag)]) / len(trnset)
            # total_entropy += -(p * np.log(p)).sum()
            total_entropy += -(len(bag)/len(trnset)) * np.log((len(bag)/len(trnset)))
            # from scipy.stats import entropy
            # total_entropy += entropy([len(trnset) - len(bag), len(bag)])

        for j in range(i, len(bags)):
            common = bags[i].intersection(bags[j])
            # if len(common) > 0:
            #     pointwise_mutual_info += (np.log(len(common)) + np.log(len(trnset)) - np.log(len(bags[i])) - np.log(len(bags[j])))
            table = np.array([
                [len(common), len(bags[i]) - len(common)],
                [len(bags[j]) - len(common), len(trnset) - len(bags[i]) - len(bags[j]) + len(common)],
            ])
            pmi = 0
            for x in range(2):
                for y in range(2):
                    joint_prob = table[x, y] / len(trnset)
                    if joint_prob == 0:
                        # pmi += -1
                        pmi += 0
                    else:
                        marg_prob0 = table[x, :].sum() / len(trnset)
                        marg_prob1 = table[:, y].sum() / len(trnset)
                        # pmi += np.log(joint_prob / (marg_prob0 * marg_prob1))
                        pmi += joint_prob * np.log(joint_prob / (marg_prob0 * marg_prob1))
                        # mi += np.log((joint_prob * joint_prob) / (marg_prob0 * marg_prob1)) / (-np.log(joint_prob))
            mutual_info_arr[i, j] = pmi

    copy_to_lower_tri = lambda X: X + X.T - np.diag(np.diag(X))
    mutual_info_arr = copy_to_lower_tri(mutual_info_arr)

    mutual_info = mutual_info_arr[np.triu_indices(len(bags), 1)].sum()
    total_mutual_info = mutual_info_arr.sum()

    if not return_all: return total_mutual_info, mutual_info, total_entropy
    else: return total_mutual_info, mutual_info, total_entropy, mutual_info_arr

def ds_mutual_info(dataset, ds_name, sub_dataset, split_path):
    from data_scripts.subsample_pool import get_data
    trn_pool, _ = get_data(dataset, sub_dataset, split_path.parent.parent)
    trn_qids = json.load(open(split_path))['train']
    trnset = trn_pool[trn_pool.qid.isin(trn_qids)]
    comp2ex, _, _ = get_subtrees(trnset, ds_name, max_size=4, context_type='', anon=False)
    all_mutual_info, mutual_info, entropy = pool_mutual_info(trnset, comp2ex)
    print(f'{split_path} {mutual_info} {entropy} {mutual_info / (entropy ** 2)}')
    return len(comp2ex), all_mutual_info, mutual_info, entropy

def get_entropy(trnset, comp2ex):
    bags = [set(ex_d.keys()) for ex_d in comp2ex.values()]
    entropy = 0
    for bag in bags:
        assert len(bag) > 0
        if len(bag) > 0: entropy += -(len(bag)/len(trnset)) * np.log((len(bag)/len(trnset)))
        # if len(bag) < len(trnset): entropy += -((len(trnset) - len(bag))/len(trnset)) * np.log(((len(trnset) - len(bag))/len(trnset)))
    return entropy

@cache
def get_data(dataset, subdataset, split_dir):
    split_info = json.load(open(split_dir / f'{subdataset or dataset}.json', 'rt'))
    trn_pool = pd.read_json(split_dir / f'{subdataset or dataset}.train.jsonl',
                            lines=True)

    return trn_pool, split_info

@app.command()
def get_split_info(dataset: str = 'covr', sub_dataset: str = '', debug: bool = False, root_dir: str = ''):
    ds_name = dataset.split('_')[0]
    dataset_l, split_type_l, split_seed_l = [dataset], ['iid', 'template'], range(4)
    subsample_type_l = ['random', 'subtree']
    get_subdataset = lambda dataset: '' if dataset != 'overnight' else 'socialnetwork'
    def full_split_type(dataset, split_type):
        if split_type != 'template': return split_type
        else: return f'template/{0.5 if dataset.startswith("covr") else 0.2}'
    n_trn_ls = defaultdict(lambda: [100, 300, 600, 1000], {
        ('covr', 'iid'): [50, 100, 300],
        ('covr', 'template/0.5'): [50, 100, 300],
        ('covr', 'subtree'): [50, 100, 300],
        ('thingtalk', 'iid'): [50, 100, 300, 600],
        ('overnight', 'iid'): [50, 100, 300],
        ('smcalflow', 'iid'): [1000, 3000, 6000],
        ('smcalflow', 'template/0.2'): [1000, 3000, 6000],
        ('smcalflow', 'subtree'): [1000, 3000, 6000],
    })
    params_l: list[SubsampleParams] = []
    for dataset, split_type, split_seed in product(dataset_l, split_type_l, split_seed_l):
        split_type = full_split_type(dataset, split_type)
        params_l.extend(get_subsample_params_single_split(
            dataset, get_subdataset(dataset), split_type,
            split_seed, subsample_type_l, n_trn_l=n_trn_ls[(dataset, split_type)]))
    rows = [p.to_dict() for p in params_l]
    midf = pd.DataFrame(rows, columns=rows[0].keys())
    split_paths = [p.get_split_file() for p in params_l]

    if 0:
        # if subdataset: split_type = '/'.join(split_dir.parts[4:-1])
        # else: split_type = '/'.join(split_dir.parts[3:-1])
        # print(ds_name, split_type)
        arg_names = ['split_seed', 'split_type', 'gen_algo', 'subsample_params']
        split_types = [
            f'template/{0.5 if dataset.startswith("covr") else 0.2}',
            'iid',
            # 'subtree',
        ]
        # splits = range(2)
        splits = range(4)
        gen_algos = ['standard']
        n_trn_ls = defaultdict(lambda: [100, 300, 600, 1000], {
            ('covr', 'iid'): [50, 100, 300],
            ('covr', 'template/0.5'): [50, 100, 300],
            ('covr', 'subtree'): [50, 100, 300],
            ('thingtalk', 'iid'): [50, 100, 300, 600],
            ('overnight', 'iid'): [50, 100, 300],
            ('smcalflow', 'iid'): [1000, 3000, 6000],
            ('smcalflow', 'template/0.2'): [1000, 3000, 6000],
            ('smcalflow', 'subtree'): [1000, 3000, 6000],
        })
        max_st_size_l = defaultdict(lambda: [2, 3, 4, 5])[(ds_name)]
        # seeds = list(range(2))
        seeds = list(range(3))

        args_l = []
        for split_type in split_types:
            n_trn_l = n_trn_ls[(ds_name, split_type)]
            params_l = []
            params_l += SubsampleParams(n_trn=n_trn_l, algo='rand', seed=seeds).get_settings(SubsampleParams.get_key_order())
            # params_l += SubsampleParams(
            #     n_trn=n_trn_l,  anon=[True, False][1 if ds_name == 'smcalflow' else 1:], compound='ngram',
            #     no_sibs=False,
            #     algo=['1', '5_cyc_fix'],
            #     ex_sel='random', seed=seeds).get_settings(SubsampleParams.get_key_order())
            params_l += SubsampleParams(
                n_trn=n_trn_l, anon=[True, False][1 if ds_name == 'smcalflow' else 1::], compound='subtree', min_freq=[1, 2][:1],
                context_type=[None, 'nt', 'rule'][:1], max_size=max_st_size_l[2:3],
                algo=['5_cyc_fix'],
                ex_sel=['random', 'new_template', 'mf_new_template'], seed=seeds).get_settings(SubsampleParams.get_key_order())
            # params_l += SubsampleParams(
            #     n_trn=n_trn_l, compound='template',
            #     algo=['1_cyc_fix', '5_cyc_fix'][0 if ds_name in ['thingtalk', 'overnight'] else 0:],
            #     ex_sel='random', seed=seeds).get_settings((SubsampleParams.get_key_order())
            if False: # entropy + subtree
                params_l += SubsampleParams(
                    n_trn=n_trn_l, anon=[False, True][:1], compound='subtree', min_freq=[1, 2, 3][:1],
                    context_type=[None, 'nt', 'rule'][:1], max_size=max_st_size_l[2:3],
                    algo='entropy').get_settings(SubsampleParams.get_key_order())
            args_l += [dict(zip(arg_names, args)) for args in
                        list(product(splits, [split_type], gen_algos, params_l))]
        # args_l = args_l[:1]
        print(f'{len(args_l)} train sets')
        print(args_l)
        if debug: return
        flat_args_l = [{k: args[k] for k in args if k != 'subsample_params'}
                | args['subsample_params'].to_dict()
                for args in args_l]
        midf = pd.DataFrame(flat_args_l, columns=flat_args_l[0].keys())

        split_paths = []
        for i, args in enumerate(params_l):
            split_name = f'{args["split_type"]}/split_{args["split_seed"]}/subsamples/{args["subsample_params"].get_name()}'
            split_path = Path(root_dir) / 'datasets' / dataset / (sub_dataset or "") / f'splits/{split_name}.json'
            split_paths.append(split_path)

    with Parallel(n_jobs=40, verbose=10) as parallel:
        result_cols = ['comp_count', 'all_mutual_info', 'mutual_info', 'entropy']
        midf[result_cols] = parallel(delayed(ds_mutual_info)(
            dataset, ds_name, sub_dataset, split_path)
                for split_path in split_paths)
    midf['ncmi0'] = midf.mutual_info / (midf.entropy ** 2)
    midf['ncmi1'] = midf.all_mutual_info / (midf.entropy ** 2)
    midf['ncmi2'] = midf.mutual_info / (midf.comp_count ** 2)
    midf['nentropy'] = midf.entropy / midf.comp_count

    print(midf.groupby(by=['n_trn', 'split_type', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi0.mean().unstack(level=0))
    print(midf.query('algo == "rand"').groupby(['n_trn']).ncmi0.min())
    print(midf.query('compound == "subtree" and algo == "5_cyc_fix"').groupby(['n_trn']).ncmi0.max())
    print(midf.groupby(by=['n_trn', 'split_type', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi2.mean().unstack(level=0))
    return midf

@app.command()
def get_ds_info(dataset: str = 'covr', sub_dataset: str = '', debug: bool = False, root_dir: str = '.'):
    ds_name = dataset.split('_')[0]
    # if subdataset: split_type = '/'.join(split_dir.parts[4:-1])
    # else: split_type = '/'.join(split_dir.parts[3:-1])
    # print(ds_name, split_type)
    arg_names = ['gen_algo', 'subsample_params']
    gen_algos = ['standard']
    n_trn_ls = {
        ('covr', 'nosplit'): [50, 100, 300, 600, 1000, 3000],
        ('atis', 'nosplit'): [50, 100, 300, 600, 1000, 3000],
        ('thingtalk', 'nosplit'): [50, 100, 300, 600, 1000, 3000],
        ('overnight', 'nosplit'): [50, 100, 300, 600, 1000, 3000],
        ('smcalflow', 'nosplit'): [1000, 3000, 6000, 10000],
    }
    max_st_size_l = defaultdict(lambda: [2, 3, 4, 5])[(ds_name)]
    seeds = list(range(3 if ds_name == 'smcalflow' else 3))

    args_l = []
    n_trn_l = n_trn_ls[(ds_name, 'nosplit')][:3]
    key_order = ['n_trn', 'seed', 'anon', 'compound', 'min_freq',
                    'no_sibs', 'context_type', 'max_size',
                    'algo', 'ex_sel', 'freq_obj_lbd']
    params_l = []
    params_l += SubsampleParams(n_trn=n_trn_l, algo='rand', seed=seeds).get_settings(key_order)
    params_l += SubsampleParams(
        n_trn=n_trn_l, anon=[True, False][1 if ds_name == 'smcalflow' else 1::], compound='subtree', min_freq=[1, 2][:1],
        context_type=[None, 'nt', 'rule'][:1], max_size=max_st_size_l[2:3],
        algo=['5_cyc_fix'],
        ex_sel=['random', 'new_template', 'mf_new_template'], seed=seeds).get_settings(key_order)
    args_l += [dict(zip(arg_names, args)) for args in
                list(product(gen_algos, params_l))]
    print(f'{len(args_l)} train sets')
    print(args_l)
    if debug: return

    flat_args_l = [{k: args[k] for k in args if k != 'subsample_params'}
              | args['subsample_params'].to_dict()
              for args in args_l]
    midf = pd.DataFrame(flat_args_l, columns=flat_args_l[0].keys())
    split_paths = []
    for i, args in enumerate(args_l):
        split_name = f'nosplit/subsamples/{args["subsample_params"].get_name()}'
        split_path = Path(root_dir) / 'datasets' / dataset / (sub_dataset or "") / f'splits/{split_name}.json'
        split_paths.append(split_path)

    with Parallel(n_jobs=40, verbose=10) as parallel:
        result_cols = ['comp_count', 'all_mutual_info', 'mutual_info', 'entropy']
        midf[result_cols] = parallel(delayed(ds_mutual_info)(
            dataset, ds_name, sub_dataset, split_path)
                for split_path in split_paths)
    midf['ncmi0'] = midf.mutual_info / (midf.entropy ** 2)
    midf['ncmi1'] = midf.all_mutual_info / (midf.entropy ** 2)
    midf['ncmi2'] = midf.mutual_info / (midf.comp_count ** 2)
    midf['nentropy'] = midf.entropy / midf.comp_count

    print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi0.mean().unstack(level=0))
    print(midf.query('algo == "rand"').groupby(['n_trn']).ncmi0.min())
    print(midf.query('compound == "subtree" and algo == "5_cyc_fix"').groupby(['n_trn']).ncmi0.max())
    print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi2.mean().unstack(level=0))

    return midf

@app.command()
def get_ds_infos(root_dir: str = '.'):
    """
    Info-theoretic analysis of subsamples of different datasets (subsample from dataset not train pool)
    """
    # covr_ds_info = get_ds_info(dataset='covr', root_dir=root_dir)_dir
    # atis_ds_info = get_ds_info(dataset='atis', root_dir=root_dir)
    # thingtalk_ds_info = get_ds_info(dataset='thingtalk', root_dir=root_dir)
    overnight_ds_info = get_ds_info(dataset='overnight', sub_dataset='socialnetwork', root_dir=root_dir)
    # smcalflow_ds_info = get_ds_info(dataset='smcalflow', root_dir=root_dir)
    ds_infos = {
        # 'covr': covr_ds_info,
        # 'atis': atis_ds_info,
        # 'thingtalk': thingtalk_ds_info,
        # 'smcalflow': smcalflow_ds_info,
        'overnight': overnight_ds_info,
    }
    outdir = Path(root_dir) / 'experiments/outputs/ds_info_fixed/'
    os.makedirs(outdir, exist_ok=True)
    for k, midf in ds_infos.items():
        print(k)
        midf.to_pickle(outdir / f'{k}.pkl')
        print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi0.agg(['mean', 'std']).unstack(level=0))
        print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi1.agg(['mean', 'std']).unstack(level=0))
        print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi2.agg(['mean', 'std']).unstack(level=0))
        # print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).nentropy.agg(['mean', 'std']).unstack(level=0))
        print()
    return ds_infos

@app.command()
def get_split_infos(root_dir: str = '.'):
    """
    Info-theoretic analysis of subsamples of train sets of different splits
    """
    covr_split_info = get_split_info(dataset='covr', root_dir=root_dir)
    # atis_split_info = get_split_info(dataset='atis', root_dir=root_dir)
    thingtalk_split_info = get_split_info(dataset='thingtalk', root_dir=root_dir)
    overnight_split_info = get_split_info(dataset='overnight', sub_dataset='socialnetwork', root_dir=root_dir)
    # smcalflow_split_info = get_split_info(dataset='smcalflow', root_dir=root_dir)
    split_infos = {
        'covr': covr_split_info,
        # 'atis': atis_split_info,
        'thingtalk': thingtalk_split_info,
        # 'smcalflow': smcalflow_split_info,
        'overnight': overnight_split_info,
    }
    outdir = Path(root_dir) / 'experiments/outputs/split_info_fixed/'
    os.makedirs(outdir, exist_ok=True)
    for k, midf in split_infos.items():
        print(k)
        # midf.to_pickle(outdir / f'{k}.pkl')
        print(midf.groupby(by=['n_trn', 'split_type', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).ncmi0.mean().unstack(level=0).unstack(level=0))
    return split_infos

if __name__ == '__main__':
    app()
