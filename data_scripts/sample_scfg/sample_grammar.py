import os, sys
import json
import random
import typer
import re
import nltk
import numpy as np
import numpy.random as npr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from collections import defaultdict, Counter
from itertools import product
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed, parallel_backend
from contextlib import nullcontext
from copy import deepcopy
from random import Random
from tqdm import tqdm
from shutil import copyfile
from rich.progress import Progress

tqdm.pandas()

from dotenv import load_dotenv

load_dotenv()

# BASE_DIR = Path({
#     'ben': '../..',
#     'shivanshu': ''
# }[os.getenv('RUNNER')])
# sys.path.insert(0, BASE_DIR.absolute().__str__())
# sys.path.insert(0, os.path.dirname(os.getcwd()))

from utils.progress import track, get_progress_bar
from generation.scfg.covr.grammars import grammar1
from generation.scfg.generate import Generation, Generator
from generation.scfg.utils import is_subtree

app = typer.Typer()
grammar = grammar1

if 0:
    def anonymize_covr_target(target):
        anonymized = target
        numbers = ['2', '3', '4']
        anonymized = re.sub(f"(\\b)({'|'.join(numbers)})(\\b)", r'ANON_NUMBER', anonymized)
        entities = ['dog', 'cat', 'mouse', 'animal']
        anonymized = re.sub(f"(\\b)({'|'.join(entities)})(\\b)", r'ANON_ENTITY', anonymized)
        relations = ['chasing', 'playing with', 'looking at']
        anonymized = re.sub(f"(\\b)({'|'.join(relations)})(\\b)", r'ANON_RELATION', anonymized)
        types = ['color', 'shape']
        anonymized = re.sub(f"(\\b)({'|'.join(types)})(\\b)", r'ANON_TYPE', anonymized)
        types_values = ['black', 'white', 'brown', 'gray', 'round', 'square', 'triangle']
        anonymized = re.sub(f"(\\b)({'|'.join(types_values)})(\\b)", r'ANON_TYPE_VALUE', anonymized)
        symbols = ['or', 'and']
        anonymized = re.sub(f"(\\b)({'|'.join(symbols)})(\\b)", r'ANON_LOGIC', anonymized)
        return anonymized

if 1:
    def anonymize_covr_target(target):
        anonymized = target
        entities = ['2', '3', '4', 'dog', 'cat', 'mouse', 'animal', 'chasing', 'playing with', 'looking at', 'color',
                    'shape', 'black', 'white', 'brown', 'gray', 'round', 'square', 'triangle']
        anonymized = re.sub(f"(\\b)({'|'.join(entities)})(\\b)", r'ANON_ENTITY', anonymized)
        symbols = ['or', 'and', 'all', 'most', 'some', 'none', 'eq', 'gt', 'lt']
        anonymized = re.sub(f"(\\b)({'|'.join(symbols)})(\\b)", r'ANON_SYMBOL', anonymized)
        return anonymized

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

def depth_balanced_sample_name(seed, size):
    return f'depth_balanced-sz{size}-{seed}'

def depth_balanced_sample(generator: Generator, size: int = None, is_test_fn = None,
                          seed: int = None, show_progress=False):
    if seed is not None:
        random.seed(seed)
        npr.seed(seed)
    def sample_count(depth):
        return generator.key_count(max_depth=depth) - generator.key_count(max_depth=depth - 1)

    def template_count(depth):
        return generator.template_count(max_depth=depth) - generator.template_count(max_depth=depth - 1)

    def get_budget(depth, template=False):
        if template: return min(depth * depth, template_count(depth))
        else: return min(depth * depth, sample_count(depth))

    budgets = {d: get_budget(d) for d in range(1, generator.max_depth + 1) if get_budget(d, template=False) > 0}
    if size:
        total = sum(budgets.values())
        budgets = {d: int(budgets[d] * size / total) for d in budgets}
        deficit = size - sum(budgets.values())
        for d in list(sorted(budgets.keys()))[:deficit]:
            budgets[d] += 1
        assert sum(budgets.values()) == size
    depth_to_templates = {d: set() for d in budgets}
    template_to_gens = defaultdict(set)
    depths_to_go = len(budgets)
    max_d = max(budgets.keys())

    print(f'{depths_to_go}\t{budgets}')
    for i in track(range(100000), disable=not show_progress):
        gen = generator.sample_key(uniform=True)
        if is_test_fn and is_test_fn(gen): continue
        # template = anonymize_covr_target(gen.sents[0])
        template = gen.sents[0]
        if template not in depth_to_templates[gen.depth] and budgets[gen.depth] > 0:
            template_to_gens[template].add(gen)
            depth_to_templates[gen.depth].add(template)
            budgets[gen.depth] -= 1
            if budgets[gen.depth] == 0:
                depths_to_go -= 1
                # print(f'{depths_to_go}\t{budgets}')
            if depths_to_go == 0 or budgets[max_d] == 0:
                break
    # print(f'{depths_to_go}\t{budgets}')
    generations = [g for t in template_to_gens for g in template_to_gens[t]]
    return generations

def unifrules_sample_name(size, seed):
    return f'unifrules-sz{size}-{seed}'

def unifrules_sample(generator: Generator, size: int = 1000, is_test_fn = None,
                     seed: int = None, show_progress=False,):
    if seed is not None:
        random.seed(seed)
        npr.seed(seed)

    generations = set()
    from contextlib import nullcontext
    progress = get_progress_bar() if show_progress else nullcontext()
    with progress:
        if show_progress: task_id = progress.add_task(f'Generating {size} samples', total=size)
        while len(generations) < size:
            # npr.seed(i)
            gen = generator.sample_key(uniform=True)
            if is_test_fn and is_test_fn(gen): continue
            generations.add(gen)
            progress.update(task_id=task_id, completed=len(generations))
            if show_progress: progress.update(task_id=task_id, completed=len(generations))
    return generations

def compounds_name(max_size, add_terms, prepend_path):
        name = f'compounds-sz{max_size}'
        if add_terms: name += '-terms'
        if prepend_path: name += '-ntpath'
        return name

def compounds_sample_name(max_size, add_terms, prepend_path, size, sort_key, seed):
    name = f'compounds-sz{max_size}'
    if add_terms: name += '-terms'
    if prepend_path: name += '-ntpath'
    if sort_key: name += f'-sort_{sort_key}'
    name += f'-sz{size}-{seed}'
    return name

def compounds_sample(generator: Generator, subtrees: list[nltk.Tree],
                     size: int, sort_key: str, sims_fn, is_test_fn=None,
                     seed: int = None, show_progress=False):
    if seed is not None:
        random.seed(seed)
        npr.seed(seed)
    generations = set()
    subtree_strings = [str(t) for t in subtrees]
    trn_subtree_freqs = {s: 0 for s in subtree_strings}
    trn_subtree_idxs = set()

    sims = None
    if sort_key.split('_')[-1] == 'stsim': sims = sims_fn(kernel='st')
    elif sort_key.split('_')[-1] == 'sstsim': sims = sims_fn(kernel='sst')

    from contextlib import nullcontext
    progress = get_progress_bar() if show_progress else nullcontext()
    with progress:
        if show_progress:
            task_id = progress.add_task(f'Generating {size} samples', total=size)
        while len(generations) < size:
            # order in which to try subtrees
            subtree_idxs = list(range(len(subtrees)))
            random.shuffle(subtree_idxs)
            if sort_key and len(generations) > 0:
                if sort_key == 'freq':
                    subtree_idxs = sorted(subtree_idxs, key=lambda i: trn_subtree_freqs[subtree_strings[i]])
                elif sort_key == 'stsim' or sort_key == 'sstsim':
                    max_sims = sims[list(trn_subtree_idxs)].max(axis=0)
                    subtree_idxs = sorted(subtree_idxs, key=lambda i: max_sims[i])
                elif sort_key == 'freq_stsim' or sort_key == 'freq_sstsim':
                    max_sims = sims[list(trn_subtree_idxs)].max(axis=0)
                    subtree_idxs = sorted(subtree_idxs, key=lambda i: (trn_subtree_freqs[subtree_strings[i]], max_sims[i]))
                else:
                    raise ValueError(f'Unknown sort key {sort_key}')
            # sample example
            found = False
            for idx in subtree_idxs:
                n_trials = 1 if sort_key == '' else 10
                for _ in range(n_trials):
                    subtree = subtrees[idx]
                    gen = generator.guided_sample_key(subtree=subtree, uniform=True)
                    if is_test_fn and is_test_fn(gen): continue
                    if gen in generations: continue
                    found = True
                    break
                if found: break
            assert found
            generations.add(gen)

            # update "counters"
            tree = rule_seq_to_parse_tree(gen.rule_seq)
            for i, subtree in enumerate(subtrees):
                if is_subtree(tree, subtree):
                    trn_subtree_freqs[subtree_strings[i]] += 1
                    trn_subtree_idxs.add(i)
            if show_progress:
                progress.update(task_id=task_id, completed=len(generations))
            # print(dict(sorted(Counter(list(trn_subtree_freqs.values())).items())[:10]))
        print(dict(sorted(Counter(list(trn_subtree_freqs.values())).items())[:10]))
    return generations

def compound_sims(
    subtrees, name: str, dataset: str = 'covr_scfg_gen', kernel: str = 'st'):
    import re
    from generation.scfg.treekernel import tree
    from generation.scfg.treekernel.tree_kernels import KernelST, KernelSST
    def get_kernel_sims(all_parse_trees, kernel):
        sims = np.empty((len(all_parse_trees), len(all_parse_trees)))
        trees = [tree.Tree.fromPrologString('1' + re.sub(r"\s+", '', str(t)))
                for t in track(all_parse_trees)]
        for t in track(trees): kernel.preProcess(t)
        for i in track(range(len(all_parse_trees))):
            for j in range(i, len(all_parse_trees)):
                a, b = trees[i], trees[j]
                sims[i][j] = sims[j][i] = kernel.evaluate(a, b)
        return sims
    if kernel == 'st':
        path = Path(f'datasets/{dataset}/{name}-stsims.npy')
        if not path.exists():
            stsims = get_kernel_sims(subtrees, KernelST(0.3))
            np.save(path, stsims)
        else:
            stsims = np.load(path)
        return stsims
    elif kernel == 'sst':
        path = Path(f'datasets/{dataset}/{name}-sstsims.npy')
        if not path.exists():
            sstsims = get_kernel_sims(subtrees, KernelSST(0.3))
            np.save(path, sstsims)
        else:
            sstsims = np.load(path)
        return sstsims
    else:
        raise ValueError(f'Unknown kernel {kernel}')
