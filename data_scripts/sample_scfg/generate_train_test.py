import os, sys
import json
import typer
import re
import pandas as pd

from rich import print
from itertools import product
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed, parallel_backend
from contextlib import nullcontext
from copy import deepcopy
from tqdm import tqdm
from rich.progress import Progress

tqdm.pandas()

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path({
    'ben': '../..',
    'shivanshu': ''
}[os.getenv('RUNNER')])
sys.path.insert(0, BASE_DIR.absolute().__str__())

from utils.progress import track, get_progress_bar
from utils.param_impl import Parameters
from data_scripts.utils.anon_fns import *
from experiments.analysis.utils.lf_utils import jaccard, tokenize_lf, TOKENS_TO_IGNORE
from generation.scfg.grammar import NonTerminal, Production, SyncGrammar, make_grammar
from scfg.covr.grammars import grammar1
from generation.scfg.generate import Generation, Generator, DiverseGenerator, rule_seq_to_parse_tree
from generation.scfg.subtree_extractor import SubtreeExtractor
from generation.scfg.utils import is_subtree
from data_scripts.sample_scfg.sample_grammar import depth_balanced_sample, depth_balanced_sample_name, unifrules_sample, unifrules_sample_name, compounds_sample, compounds_sample_name, compounds_name, compound_sims

app = typer.Typer()
# grammar = grammar1

if 1:
    def get_generator(dataset: str, diverse: bool = False,
                      return_subtree_extractor: bool = False):
        if dataset.startswith('covr_scfg'):
            from scfg.covr.grammars import grammar1
            scfg = make_grammar(grammar1)
            kwargs = dict(
                max_depth = 21,
                max_freq = tuple({'filter_object': 2, 'with_relation_ref': 2}.items())
            )
        elif dataset.startswith('geoquery_scfg'):
            from generation.scfg.geoquery.grammar import get_scfg
            scfg = get_scfg()
            kwargs = dict(max_depth=6)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        gen_cls = DiverseGenerator if diverse else Generator
        generator = gen_cls(scfg, **kwargs)
        subtree_extractor = SubtreeExtractor(scfg, **kwargs)

        if return_subtree_extractor: return generator, subtree_extractor
        else: return generator

    def make_df_row(g: Generation, id) -> dict:
        return dict(
            qid=id, source=g.sents[1].capitalize(), target=g.sents[0],
            productions=[r.single_lang_repr(0) for r in g.rule_seq],
            depth=g.depth,
            anonymized_target=anonymize_covr_target(g.sents[0]))

    def dump_gens_df(df, path):
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        print(path)
        with open(path, 'w') as f:
            df.to_json(f, lines=True, orient='records')

    def get_settings(d):
        from collections.abc import Iterable
        keys = []
        value_lists = []
        for k, v in d.items():
            if isinstance(v, dict):
                value_lists.append(get_settings(v))
                keys.append(k)
            elif isinstance(v, Iterable):
                if not isinstance(v, list): v = list(v)
                if len(v) == 0:
                    raise ValueError(f"Empty settings list for {k}")
                if isinstance(v[0], dict):    # Each value in settings list itself extends GridMixin so needs to be explored in the search space
                    value_lists.append([s for _v in v for s in get_settings(_v)])
                else:
                    value_lists.append(v)
                keys.append(k)
        _setting = deepcopy(d)
        settings = []
        for values in product(*value_lists):
            for k, v in zip(keys, values):
                _setting[k] = v
            settings.append(deepcopy(_setting))
        return settings

def compounds_samples(
    dataset: str = 'covr_scfg_gen', subdataset: str = '',
    subtree_size: int = None, add_terms: bool = False, prepend_path: bool = True,
    size: int = 100, sort_key:str = 'freq', seed: int = 0, is_test_fn=None,
    run_all_subtree: bool = False, run_all_sel: bool = False, use_parallel: bool = False):

    if not run_all_subtree:
        subtree_args_l = [dict(max_size=subtree_size,
                                add_terms=add_terms,
                                prepend_path=prepend_path)]
    else:
        subtree_sizes = dict(covr=[8, 10, 12][1:2], geoquery=[5])[dataset.split('_')[0]]
        subtree_args_l = get_settings(dict(max_size=subtree_sizes,
                                            add_terms=[False],
                                            prepend_path=[True]))
    gens_l = []
    for subtree_args in subtree_args_l:
        print(subtree_args)
        generator, subtree_extractor = get_generator(dataset, return_subtree_extractor=True)

        # enumerate subtrees
        subtrees = subtree_extractor.extract_subtrees(**subtree_args)
        print(f'{len(subtrees)} subtrees')

        # generate examples
        if not run_all_sel:
            args_l = [dict(size=size, sort_key=sort_key, seed=seed)]
        else:
            size_l = [100, 300, 600, 1000]
            sort_keys = []
            sort_keys += ['', 'freq']
            # sort_keys += ['stsim', 'freq_stsim'][:1]
            # sort_keys += ['sstsim', 'freq_sstsim'][:1]
            args_l = get_settings(dict(size=size_l, sort_key=sort_keys,
                                        seed=range(1)))

        def _compounds_sample_wrapper(i, args, show_progress=True):
            print(f'{i}: {args}')
            name = compounds_sample_name(**subtree_args, **args)
            sims_fn = partial(compound_sims, subtrees,
                            compounds_name(**subtree_args),
                            dataset, subdataset)
            gens = compounds_sample(
                generator, subtrees, **args,
                is_test_fn=is_test_fn, sims_fn=sims_fn,
                show_progress=show_progress)
            for g in gens: assert not is_test_fn(g)
            return name, gens

        n_jobs = 4 if use_parallel else 1
        with Parallel(n_jobs=n_jobs) as parallel:
            _gens_l = parallel(delayed(_compounds_sample_wrapper)(i, args, show_progress=not use_parallel)
                    for i, args in enumerate(args_l))
        gens_l += _gens_l
    return gens_l

@app.command()
def unifrules_test(
    dataset: str = 'covr_scfg_gen_v3', size: int = 2000,
    show_progress: bool = True):

    for seed in range(3):
        seed = 1000 + seed
        generator = get_generator(dataset)
        gens = unifrules_sample(generator, size=size, seed=seed,
                                       show_progress=show_progress)
        name = unifrules_sample_name(size, seed)
        df = pd.DataFrame([make_df_row(g, f'test-{name}-{i}')
                           for i, g in enumerate(gens)])
        dump_gens_df(df, f'datasets/{dataset}/splits/{name}-iid/test.jsonl')
        dump_gens_df(df, f'datasets/{dataset}/splits/{name}-template/test.jsonl')

@app.command()
def depth_balanced_test(
    dataset: str = 'covr_scfg_gen', size: int = 2000,
    show_progress: bool = True):
    generator = get_generator(dataset)
    for seed in range(3):
        seed = 1000 + seed
        gens = depth_balanced_sample(generator, size=size, seed=seed,
                                     show_progress=show_progress)
        name = depth_balanced_sample_name(seed, size)
        df = pd.DataFrame([make_df_row(g, f'test-{name}-{i}')
                        for i, g in enumerate(gens)])
        dump_gens_df(df, f'datasets/{dataset}/splits/{name}-iid/test.jsonl')
        dump_gens_df(df, f'datasets/{dataset}/splits/{name}-template/test.jsonl')

@app.command()
def compounds_test(
    dataset: str = 'covr_scfg_gen', size: int = 2000,
    subtree_size: int = 10, add_terms: bool = False, prepend_path: bool = True,
    sort_key:str = ''):
    generator, subtree_extractor = get_generator(
        dataset, return_subtree_extractor=True)
    subtree_args = dict(max_size=subtree_size, add_terms=add_terms,
                        prepend_path=prepend_path)
    # enumerate subtrees
    subtrees = subtree_extractor.extract_subtrees(**subtree_args)
    print(f'{len(subtrees)} subtrees')
    sims_fn = partial(compound_sims, subtrees, compounds_name(**subtree_args), dataset)
    for seed in range(3):
        seed = 1000 + seed
        args = dict(size=size, sort_key=sort_key, seed=seed)
        gens = compounds_sample(generator, subtrees, **args,
                                sims_fn=sims_fn, show_progress=True)
        name = compounds_sample_name(**subtree_args, **args)
        df = pd.DataFrame([make_df_row(g, f'test-{name}-{i}')
                        for i, g in enumerate(gens)])
        dump_gens_df(df, f'datasets/{dataset}/splits/{name}-iid/test.jsonl')
        dump_gens_df(df, f'datasets/{dataset}/splits/{name}-template/test.jsonl')

def get_test_fn(test_path):
    test_path = Path(test_path)
    test_set = pd.read_json(test_path, lines=True)
    if 'template' in str(test_path):
        print('creating template split')
        test_templates = set(test_set.anonymized_target)
        return lambda gen: anonymize_covr_target(gen.sents[0]) in test_templates
    elif 'iid' in str(test_path):
        print('creating iid split')
        test_targets = set(test_set.target)
        return lambda gen: gen.sents[0] in test_targets
    else:
        raise ValueError(f'Unknown test path {test_path}')

@app.command()
def unifrules_train(
    dataset: str = 'covr_scfg_gen', subdataset: str = '',
    test_path: str = 'datasets/covr_scfg_gen/splits/unifrules-template/split_0/test.jsonl',
    size: int = 100, seed: int = 0, run_all: bool = False, use_parallel: bool = False):
    is_test_fn = get_test_fn(test_path)
    split_dir = Path(test_path).parent
    generator = get_generator(dataset)

    if not run_all:
        args_l = [dict(size=size, seed=seed)]
    else:
        size_l = [100, 300, 600, 1000]
        args_l = get_settings(dict(size=size_l, seed=range(3)))

    def _unifrules_sample_wrapper(i, args, show_progress=True):
        print(f'{i}: {args}')
        name = unifrules_sample_name(**args)
        trn_gens = unifrules_sample(generator, **args,
                                    is_test_fn=is_test_fn,
                                    show_progress=show_progress)
        for g in trn_gens: assert not is_test_fn(g)
        df = pd.DataFrame([make_df_row(g, f'train-{name}-{i}')
                        for i, g in enumerate(trn_gens)])
        dump_gens_df(df, split_dir / f'unifrules' / f'{name}.jsonl')

    n_jobs = 4 if use_parallel else 1
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(delayed(_unifrules_sample_wrapper)(i, args, show_progress=not use_parallel)
            for i, args in enumerate(args_l))

    # dump_gens_df(test_set, split_dir / f'unifrules-sample' / 'test.jsonl')

@app.command()
def compounds_train(
    dataset: str = 'covr_scfg_gen', subdataset: str = '',
    test_path: str = 'datasets/covr_scfg_gen/splits/unifrules-template/split_0/test.jsonl',
    subtree_size: int = None, add_terms: bool = False, prepend_path: bool = True,
    size: int = 100, sort_key:str = 'freq', seed: int = 0,
    run_all_subtree: bool = False, run_all_sel: bool = False,
    use_parallel: bool = False):
    is_test_fn = get_test_fn(test_path)
    split_dir = Path(test_path).parent

    gens_l = compounds_samples(dataset, subdataset,
        subtree_size, add_terms, prepend_path,
        size, sort_key, seed, is_test_fn,
        run_all_subtree, run_all_sel, use_parallel)

    for name, gens in gens_l:
        df = pd.DataFrame([make_df_row(g, f'train-{name}-{i}')
                        for i, g in enumerate(gens)])
        dump_gens_df(df, split_dir / f'compounds' / f'{name}.jsonl')
    # dump_gens_df(test_set, split_dir / f'compounds-sample' / 'test.jsonl')

@app.command()
def depth_balanced_train(
    dataset: str = 'covr_scfg_gen', subdataset: str = '',
    test_path: str = 'datasets/covr_scfg_gen/splits/unifrules-template/split_0/test.jsonl',
    size: int = 100, seed: int = 0, run_all: bool = False, use_parallel: bool = False):

    is_test_fn = get_test_fn(test_path)
    split_dir = Path(test_path).parent
    generator = get_generator(dataset)

    if not run_all:
        args_l = [dict(size=size, seed=seed)]
    else:
        size_l = [100, 300, 600, 1000]
        args_l = get_settings(dict(size=size_l, seed=range(3)))

    def _depth_balanced_sample_wrapper(i, args, show_progress=True):
        print(f'{i}: {args}')
        name = depth_balanced_sample_name(**args)
        trn_gens = depth_balanced_sample(generator, **args,
                                    is_test_fn=is_test_fn,
                                    show_progress=show_progress)
        for g in trn_gens: assert not is_test_fn(g)
        df = pd.DataFrame([make_df_row(g, f'train-{name}-{i}')
                        for i, g in enumerate(trn_gens)])
        dump_gens_df(df, split_dir / f'depth_balanced' / f'{name}.jsonl')

    n_jobs = 4 if use_parallel else 1
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(delayed(_depth_balanced_sample_wrapper)(i, args, show_progress=not use_parallel)
            for i, args in enumerate(args_l))

if 1:
    @app.command()
    def unifrules_pool(dataset: str = 'covr_scfg_gen_v3', size: int = 1000,
                     show_progress: bool = True):
        generator = get_generator(dataset)
        generations = unifrules_sample(generator, size=size, show_progress=show_progress)
        df = pd.DataFrame([make_df_row(g, f'unifrules-{i}')
                        for i, g in enumerate(track(generations))])
        dump_gens_df(df, Path(f'datasets/{dataset}/{dataset}-unifrules.jsonl'))

    @app.command()
    def diverse_pool(dataset: str = 'covr_scfg_gen_v3',  max_len: int = 250):
        generator = get_generator(dataset, diverse=True)
        print(generator.key_count())
        print(generator.diverse_count())
        generations = generator.diverse_generate()
        print(len(generations))
        generations = [g for g in generations if len(tokenize_lf(g.sents[0])) <= max_len]
        df = pd.DataFrame([make_df_row(g, f'diverse-{i}')
                        for i, g in enumerate(track(generations))])
        print(df.shape[0])
        dump_gens_df(df, Path(f'datasets/{dataset}/{dataset}-diverse.jsonl'))

    if 0:
        @app.command()
        def old_compounds_pool(dataset: str = 'covr_scfg_gen_v3',
                            subtree_size: int = 10, add_terms: bool = False,
                            prepend_path: bool = True, run_all: bool = False):
            subtree_argnames = ['max_size', 'add_terms', 'prepend_path']
            if not run_all:
                subtree_args_l = [[subtree_size, add_terms, prepend_path]]
            else:
                subtree_sizes = [8, 10, 12]
                subtree_args_l = list(product(subtree_sizes, [False], [True]))
            subtree_args_l = [dict(zip(subtree_argnames, args))
                            for args in subtree_args_l]

            def work(i, subtree_args, show_progress=True):
                print(f'{i}: {subtree_args}')
                generator, subtree_extractor = get_generator(dataset,
                                                            return_subtree_extractor=True)
                # enumerate subtrees
                subtrees = subtree_extractor.extract_subtrees(**subtree_args)
                if show_progress: print(f'{len(subtrees)} subtrees')

                # generate samples
                generations = [generator.guided_sample_key(subtree=subtree, uniform=True)
                                for subtree in track(subtrees, disable=not show_progress,
                                                    desc='Generating examples for subtrees')]
                if show_progress: print(f'{len(generations)} generations')
                name = compounds_pool_name(**subtree_args)
                df = pd.DataFrame([make_df_row(g, f'{name}-{i}')
                                for i, g in enumerate(generations)])
                print(df.shape[0], 'examples')
                dump_gens_df(df, Path(f'datasets/{dataset}/{dataset}-{name}.jsonl'))

            with Parallel(n_jobs=12) as parallel:
                parallel(delayed(work)(i, subtree_args, show_progress=False)
                        for i, subtree_args in enumerate(subtree_args_l))

    @app.command()
    def iid_split(dataset: str = 'covr_scfg_gen_v3', pool_type: str = 'unifrules', n_splits: int = 5):
        data = pd.read_json(f'datasets/{dataset}/{dataset}-{pool_type}.jsonl', lines=True)
        from sklearn.model_selection import train_test_split
        for seed in range(n_splits):
            train_data, test_data = train_test_split(data, test_size=2000, random_state=seed)
            split = dict(
                train=train_data.qid.values.tolist(),
                test=test_data.qid.values.tolist(),
                split_name=f'{dataset}/iid/split_{seed}')
            split_dir = Path(f'datasets/{dataset}/splits/{pool_type}-iid/split_{seed}')
            dump_gens_df(train_data, split_dir / f'{dataset}.train.jsonl')
            dump_gens_df(test_data, split_dir / f'{dataset}.test.jsonl')
            json.dump(split, open(split_dir / f'{dataset}.json', 'w'), indent=2)

    @app.command()
    def split_generated(dataset: str = 'covr_scfg_gen', subdataset: str = '',
                        test_path: str = 'datasets/covr_scfg_gen/splits/unifrules-template/split_0/covr_scfg_gen.test.jsonl', gen_algo='diverse', run_all: bool = False):
        if not run_all:
            gen_algos = [gen_algo]
        else:
            gen_algos = []
            gen_algos += ['unifrules']
            gen_algos += ['diverse']
            subtree_argnames = ['max_size', 'add_terms', 'prepend_path']
            subtree_args_l = list(product([8, 10, 12][1:2], [False], [True]))
            gen_algos += [compounds_name(**dict(zip(subtree_argnames, subtree_args)))
                        for subtree_args in subtree_args_l]
        for gen_algo in gen_algos:
            print(gen_algo)
            test_path = Path(test_path)
            # split_info = json.load(open(split_dir / f'{dataset}.json', 'rt'))
            test_set = pd.read_json(test_path, lines=True)
            train_pool = pd.read_json(f'datasets/{dataset}/{dataset}-{gen_algo}.jsonl',
                                    lines=True)
            print(f'{len(train_pool)} examples in train pool')
            split_dir = test_path.parent
            if 'template' in str(split_dir):
                test_templates = set(test_set.anonymized_target)
                train_pool = train_pool.query('anonymized_target not in @test_templates')
                print(f'{len(train_pool)} examples in train pool after removing test templates')
                assert test_templates.intersection(set(train_pool.anonymized_target)) == set()
            elif 'iid' in str(split_dir):
                test_targets = set(test_set.target)
                train_pool = train_pool.query('target not in @test_targets')
                print(f'{len(train_pool)} examples in train pool after removing test examples')
                assert test_targets.intersection(set(train_pool.target)) == set()
            else:
                raise ValueError(f'Unknown split dir {split_dir}')
            split = dict(train=train_pool.qid.values.tolist(),
                        test=test_set.qid.values.tolist(),
                        # split_name=f'{split_info["split_name"]}/{gen_algo}'
                        )

            new_split_dir = split_dir / f'{gen_algo}-splits'
            os.makedirs(new_split_dir, exist_ok=True)
            # dump_gens_df(train_pool, new_split_dir / f'{dataset}.train.jsonl')
            # dump_gens_df(test_set, new_split_dir / f'{dataset}.test.jsonl')
            json.dump(split, open(new_split_dir / f"{dataset}.json", 'w'), indent=2)

            from subsample_trn_pool import main
            main(dataset=dataset, subdataset=subdataset, split_dir=new_split_dir,
                run_all=True, parallel=False, trn_pool=train_pool)

usage = """
ds=covr_scfg_gen_v4
split_type=iid

python data_scripts/sample_grammar.py pool --dataset $ds --size 100000

python data_scripts/sample_grammar.py diverse-pool --dataset $ds --depth 21
python data_scripts/sample_grammar.py compounds-pool --dataset $ds --depth 21 --run-all
cat datasets/$ds/$ds.all.jsonl datasets/$ds/$ds-*.jsonl > datasets/$ds/$ds.combined.jsonl

python data_scripts/sample_grammar.py iid-split --dataset $ds
python data_scripts/split_datasets.py --dataset $ds --split-method template --percentage_programs_test 0.5 --limit-test 2000

for split in 0 1 2 3 4; python data_scripts/subsample_trn_pool.py --run-all --dataset $ds --split-dir datasets/$ds/splits/$split_type/split_$split/
for split in 0 1 2 3 4; python data_scripts/sample_grammar.py compounds-sample --dataset $ds --split-dir datasets/$ds/splits/$split_type/split_$split/ --depth 21 --run-all-subtree --run-all-sel

for split in 0 1 2 3 4; python data_scripts/sample_grammar.py split-generated --dataset $ds --split-dir datasets/$ds/splits/$split_type/split_$split/ --gen-algo compounds-sz10-ntpath
for split in 0 1 2 3 4; python data_scripts/sample_grammar.py split-generated --dataset $ds --split-dir datasets/$ds/splits/$split_type/split_$split/ --run-all


for split in 0; python data_scripts/sample_grammar.py compounds-sample --dataset $ds --split-dir datasets/$ds/splits/$split_type/split_$split/ --depth 21
for split in 0; python data_scripts/subsample_trn_pool.py --dataset $ds --split-dir datasets/$ds/splits/$split_type/split_$split/ --size 1000 --algo 4
"""

usage = """
python datasets/scripts/sample_grammar.py sample --dataset covr --size 100000
python datasets/scripts/split_datasets.py --dataset covr_scfg_gen --split-method template --percentage_programs_test 0.5 --limit-test 2000
for ds in covr_scfg_gen; for split in 1 2 3 4; python datasets/scripts/subsample_trn_pool.py --run-all --dataset $ds --split-dir datasets/$ds/splits/template/0.5/split_$split/
for ds in covr_scfg_gen; for split in 1 2 3 4; python datasets/scripts/sample_grammar.py split --dataset $ds --split-dir datasets/$ds/splits/template/0.5/split_$split/
for ds in covr_scfg_gen; for split in 0; python datasets/scripts/subsample_trn_pool.py --run-all --dataset $ds --split-dir datasets/$ds/splits/template/0.5/split_$split/diverse_splits
"""

if __name__ == '__main__':
    app()
