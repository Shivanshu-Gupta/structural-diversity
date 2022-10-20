import os, sys
import json
import typer
import pandas as pd

from rich import print
from collections import defaultdict
from functools import partial, cache
from joblib import Parallel, delayed, parallel_backend
from copy import deepcopy
from tqdm import tqdm

from data_scripts.subsample.subsamples import SubsampleParams, get_n_trn_l, get_subsample_params, get_subsample_params_single_split, get_vals
from data_scripts.utils.anon_fns import anonymize_covr_target, anonymize_atis_target, anonymize_overnight_target, anonymize_thingtalk, anonymize_smcalflow_target
from data_scripts.subsample.substructs import get_ngrams, get_templates, get_subtrees
from data_scripts.subsample.algos import iterative_subsample
# from generation.scfg.utils import get_subtrees as get_subtrees_fn

tqdm.pandas()
# app = typer.Typer(rich_markup_mode="rich")
app = typer.Typer()

@cache
def get_split_info(dataset, subdataset, split_dir):
    filename = f"{dataset if dataset != 'overnight' else subdataset}.json"
    split_info = json.load(open(split_dir / filename, 'rt'))
    return split_info

def get_trn_pool(dataset, subdataset, split_dir):
    ds_name = dataset.split('-')[0]
    filename = f"{dataset if dataset != 'overnight' else subdataset}.train.jsonl"
    trn_pool = pd.read_json(split_dir / filename, lines=True)
    if 'anonymized_target' not in trn_pool.columns:
        anon_fn = {'covr': anonymize_covr_target, 'atis': anonymize_atis_target, 'thingtalk': anonymize_thingtalk, 'overnight': anonymize_overnight_target, 'smcalflow': anonymize_smcalflow_target, 'smcalflow-nostr': anonymize_smcalflow_target}[ds_name]
        trn_pool['anonymized_target'] = trn_pool.target.apply(anon_fn)
    return trn_pool

@cache
def get_trn_qids(dataset, subdataset, split_dir):
    trn_pool = get_trn_pool(dataset, subdataset, split_dir)
    return trn_pool.qid

@cache
def get_subtrees_fn(dataset, subdataset, split_dir, max_size, context_type, anon):
    trn_pool = get_trn_pool(dataset, subdataset, split_dir)
    ds_name = dataset.split('-')[0]
    return get_subtrees(trn_pool, ds_name=ds_name, max_size=max_size, context_type=context_type, anon=anon)

@cache
def get_ngrams_fn(dataset, subdataset, split_dir, no_sibs, anon, return_sims=False):
    trn_pool = get_trn_pool(dataset, subdataset, split_dir)
    return get_ngrams(trn_pool, no_sibs, anon, return_sims)

@cache
def get_templates_fn(dataset, subdataset, split_dir):
    trn_pool = get_trn_pool(dataset, subdataset, split_dir)
    return get_templates(trn_pool)

def get_max_n_trn(dataset, split_type, key=None):
    n_trn_l = get_n_trn_l(dataset, split_type)
    max_n_trn = [
        max(n_trn_l),
        # max(n_trn_l_d[ds_name][split_type][None]),
        # 1000,
    ][0]
    return max_n_trn

def subsample(i, params: SubsampleParams, show_progress=True):
    print(f'{i}: {params}')
    # output_path = split_dir / 'subsamples'
    # if subdataset: output_path /= subdataset
    # output_path /= f'{params.get_name()}.json'
    dataset, subdataset = params.dataset, params.subdataset
    split_dir = params.get_split_dir()
    split_info = get_split_info(dataset, subdataset, split_dir)
    output_path = params.get_split_file()

    max_n_trn = get_max_n_trn(dataset, params.split_type)
    max_n_trn_params = deepcopy(params)
    max_n_trn_params.n_trn = max_n_trn
    max_output_path = output_path.parent / f'{max_n_trn_params.get_name()}.json'

    n_trn, seed, algo = params.n_trn, params.seed, params.algo
    if params.n_trn != max_n_trn and max_output_path.exists() and algo != 'set-cover':
        print('loading from cache')
        split_info['train'] = json.load(open(max_output_path))['train'][:params.n_trn]
    else:
        if algo == 'rand':
            trn_qids = trn_qids.sample(n_trn, random_state=seed).values
        else:
            common_kwargs = dict(dataset=dataset, subdataset=subdataset, split_dir=split_dir)
            max_size, context_type, anon = params.max_size, params.context_type, params.anon
            compounds_fn = {
                'subtree': partial(get_subtrees_fn, max_size=max_size,
                                   context_type=context_type, anon=anon, **common_kwargs),
                'ngram': partial(get_ngrams_fn, no_sibs=params.no_sibs, anon=anon,
                                 return_sims=('4' in algo), **common_kwargs),
                'template': partial(get_templates_fn, **common_kwargs),
            }[params.compound]
            comp2ex, ex2comp, comp_sims = compounds_fn()
            template2ex, ex2template, _ = get_templates_fn(**common_kwargs)
            ex2template = {ex: list(d.keys())[0] for ex, d in ex2template.items()}
            trn_qids = iterative_subsample(
                algo, comp2ex, ex2comp, comp_sims, template2ex, ex2template,
                min_freq=params.min_freq, n_trn=n_trn, seed=seed,
                ex_sel=params.ex_sel, show_progress=show_progress)
        split_info['train'] = list(trn_qids)
    assert(len(split_info['train']) == params.n_trn)
    # exit()
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'wt') as f:
        print(output_path)
        json.dump(split_info, f, indent=2)

def comp_counts():
    import pandas as pd
    from pathlib import Path
    from data_scripts.subsample_pool import get_subtrees, get_ngrams, get_templates
    trn_paths = []
    # trn_paths += [Path(f'datasets/ben/{dataset}/splits/template/split_{split}/{dataset}.train.jsonl') for dataset in ['covr', 'atis', 'thingtalk'][:-1] for split in range(1)]
    trn_paths += [
        Path(f'datasets/{dataset}/splits/{split_type}/split_{split}/{dataset}.train.jsonl')
        for dataset in ['covr', 'atis', 'thingtalk', 'smcalflow']
        for split_type in ['iid', f'template/{0.5 if dataset.startswith("covr") else 0.2}']
        for split in range(1)
    ]
    trn_paths += [
        Path(f'datasets/overnight/{subdataset}/splits/{split_type}/split_{split}/{subdataset}.train.jsonl')
        for subdataset in ['socialnetwork'] for split_type in ['iid', f'template/0.2'] for split in range(1)
    ]
    trn_paths = [
        Path(f'datasets/{dataset}/splits/{split_type}/split_{split}/{dataset}.train.jsonl')
        for dataset in ['smcalflow'] for split_type in ['iid'] for split in range(1)
    ]
    print(trn_paths)
    records = []
    for trn_path in trn_paths:
        # print(trn_path, trn_path.exists())
        assert trn_path.exists(), f'{trn_path} does not exist'
        trn_pool = pd.read_json(trn_path, lines=True)
        if 'anonymized_target' not in trn_pool.columns:
            anon_fn = {'covr': anonymize_covr_target, 'atis': anonymize_atis_target, 'thingtalk': anonymize_thingtalk, 'overnight': anonymize_overnight_target}[str(trn_path).split('/')[1]]
            trn_pool['anonymized_target'] = trn_pool.target.apply(anon_fn)

        template2ex, _, _ = get_templates(trn_pool)
        print(trn_path, f'{len(template2ex)} templates')
        records.append({'compound': 'template', 'path': str(trn_path), 'count': len(template2ex)})
        for anon in [True, False]:
            ngram2ex, _, _ = get_ngrams(trn_pool, False, anon=anon)
            print(trn_path, f'{len(ngram2ex)} ngrams')
            records.append({'compound': 'ngram', 'anon': anon, 'path': str(trn_path), 'count': len(ngram2ex)})
            for max_size in range(2, 5):
                subtree2ex, _, _ = get_subtrees(trn_pool, max_size=max_size, context_type='', anon=anon)
                records.append({'compound': 'subtree', 'max_size': max_size, 'anon': anon, 'path': str(trn_path), 'count': len(subtree2ex)})
                print(trn_path, f'{len(subtree2ex)} subtrees of max_size {max_size}')

# @app.command()
# def single_subsample(
#         dataset: str = 'covr', subdataset: str = '',
#         split_type: str = 'iid', split_seed: str = 0,
#         n_trn: int = 1000, compound: str = None, algo: str = 'rand',
#         no_sibs: bool = False, ex_sel: str = 'random', seed: int = 0,
#         trn_pool=None, debug: bool = False, print_params: bool = False,):
#     params_l = [SubsampleParams(
#         dataset=dataset, subdataset=subdataset, split_type=split_type, split_seed=split_seed,
#         compound=compound, n_trn=n_trn, algo=algo, no_sibs=no_sibs, ex_sel=ex_sel, seed=seed)]
#     if print_params:
#         for i, p in enumerate(params_l):
#             print(f'{i}\t{p.get_name()}\t{p}')
#     if debug: return params_l
#     run(params_l, parallel=False)

# @app.command()
# def main(
#         dataset: list[str] = typer.Option(['covr']),
#         split_type: list[str] = typer.Option('all'),
#         split_seed: list[int] = typer.Option(-1),
#         subsample_type: list[str] = typer.Option('all'),
#         print_params: bool = False, print_paths: bool = False,
#         only_incomplete: bool = False, parallel: bool = False, debug: bool = False,):
@app.command()
def main(
        dataset: str = 'covr', split_type: str = 'all',
        split_seed: str = '-1', subsample_type: str = 'all',
        print_params: bool = False, print_paths: bool = False,
        only_incomplete: bool = False, parallel: bool = False, debug: bool = False,):
    dataset_l = get_vals(dataset, str, ['covr', 'atis', 'overnight', 'thingtalk', 'smcalflow-uncommon'])
    split_type_l = get_vals(split_type, str, ['iid', 'template', 'subtree'])
    split_seed_l = get_vals(split_seed, int, range(4))
    subsample_type_l = get_vals(subsample_type, str, ['random', 'ngram', 'template', 'subtree'])
    params_l = get_subsample_params(dataset_l, split_type_l, split_seed_l, subsample_type_l,
                                    reverse_n_trn_l=True)
    print(f'{len(params_l)} params')
    if only_incomplete:
        params_l = [p for p in params_l if not p.get_split_file().exists()]
    print(f'{len(params_l)} incomplete params')
    if print_params:
        for i, p in enumerate(params_l):
            print(f'{i}\t{p.get_split_file().exists()}\t{p}')
    if print_paths:
        for i, p in enumerate(params_l):
            print(f'{i}\t{p.get_split_file().exists()}\t{p.get_split_file()}')
    if debug: return
    if not parallel:
        for i, p in enumerate(params_l):
            subsample(i, p, show_progress=True)
    else:
        n_jobs = 10 if parallel else 1
        with Parallel(n_jobs=n_jobs, verbose=1, max_nbytes=1e6) as parallel:
            parallel(delayed(subsample)(i, p, show_progress=False)
                     for i, p in enumerate(params_l))

# @app.command()
# def subsample_status(
        # dataset: list[str] = typer.Option(['covr']),
        # split_type: list[str] = typer.Option(['iid']),
        # split_seed: list[int] = typer.Option([0,1,2,3]),
        # subsample_type: list[str] = typer.Option(['all']),
        # print_params: bool = False, print_paths: bool = False,
        # print_summary: bool = False,):
    # dataset_l = dataset if dataset != ['all'] else ['covr', 'atis', 'overnight', 'thingtalk', 'smcalflow-uncommon']
    # split_type_l = split_type if split_type != ['all'] else ['iid', 'template', 'subtree']
    # split_seed_l = split_seed if split_seed != [-1] else range(4)
    # subsample_type_l = subsample_type if subsample_type != ['all'] else ['random', 'ngram', 'template', 'subtree']
@app.command()
def subsample_status(
        dataset: str = 'covr', split_type: str = 'all',
        split_seed: str = '-1', subsample_type: str = 'all',
        print_params: bool = False, print_paths: bool = False):
    dataset_l = get_vals(dataset, str, ['covr', 'atis', 'overnight', 'thingtalk', 'smcalflow-uncommon'])
    split_type_l = get_vals(split_type, str, ['iid', 'template', 'subtree'])
    split_seed_l = get_vals(split_seed, int, range(4))
    subsample_type_l = get_vals(subsample_type, str, ['random', 'ngram', 'template', 'subtree'])
    get_subdataset = lambda dataset: '' if dataset != 'overnight' else 'socialnetwork'
    def full_split_type(dataset, split_type):
        if split_type != 'template': return split_type
        else: return f'template/{0.5 if dataset.startswith("covr") else 0.2}'

    params_l_store: dict[(str, str), list[SubsampleParams]] = defaultdict(list)
    for dataset in dataset_l:
        for split_type in split_type_l:
            for split_seed in split_seed_l:
                params_l = get_subsample_params_single_split(
                    dataset, get_subdataset(dataset), full_split_type(dataset, split_type),
                    split_seed, subsample_type_l, verbose=False,
                    print_params=print_params, print_paths=print_paths)
                params_l_store[(dataset, split_type)].extend(params_l)
                existence = [p.get_split_file().exists() for p in params_l]
                print(f'Total {sum(existence)}/{len(existence)} subsamples done for [yellow]{split_type}[/yellow] split {split_seed} of [red]{dataset}[/red]')
            existence = [p.get_split_file().exists() for p in params_l_store[(dataset, split_type)]]
            print(f'Total {sum(existence)}/{len(existence)} subsamples done for [yellow]{split_type}[/yellow] splits of [red]{dataset}[/red]')
        existence = [p.get_split_file().exists() for s in split_type_l
                     for p in params_l_store[(dataset, s)]]
        print(f'Total {sum(existence)}/{len(existence)} subsamples done for [red]{dataset}[/red]')
    existence = [p.get_split_file().exists() for d in dataset_l
                 for s in split_type_l for p in params_l_store[(d, s)]]
    print(f'Total {sum(existence)}/{len(existence)} subsamples done')
    rows = [p.to_dict() | {'completed': p.get_split_file().exists()}
            for k in params_l_store for p in params_l_store[k]]
    df = pd.DataFrame(rows, columns=rows[0].keys())
    print(df)
    for dataset in dataset_l:
        print(dataset)
        print(df.query(f'completed == False and dataset == "{dataset}"').groupby(by=['n_trn', 'split_type', 'split_seed', 'compound', 'anon', 'min_freq', 'max_size', 'algo', 'ex_sel'], dropna=False).seed.count().unstack(level=1).unstack(level=0).dropna(axis=1, how='all'))

if __name__ == '__main__':
    app()
