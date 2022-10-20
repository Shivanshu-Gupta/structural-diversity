import attr

from rich import print
from collections import defaultdict
from pathlib import Path
from itertools import product

from utils.param_impl import Parameters

@attr.s(auto_attribs=True)
class SubsampleParams(Parameters):
    dataset: str = 'covr'
    subdataset: str = ''
    split_type: str = 'iid'
    split_seed: int = 0
    n_trn: int = 100
    seed: int = 0
    anon: bool = False
    compound: str = None # None or 'ngram' or 'subtree' or 'depst' or 'template'
    min_freq: int = 1

    # subtree params
    context_type: str = None  # ['nt', 'rule', None]
    max_size: int = None

    # compound selection
    algo: str = 'rand'  # ['rand', '0', '1', '2', '4', '24', '25', 'entropy', 'weighted-entropy','set-cover']
    no_sibs: bool = False
    ex_sel: str = 'random'
    freq_obj_lbd: float = 0

    def get_old_name(self):
        name_parts = []
        if self.anon: name_parts.append('anon')
        # name_parts.append(self.compound)
        assert self.compound in [None, 'ngram', 'template', 'subtree', 'depst', 'subtree-depst']
        if self.compound == 'subtree' or self.compound == 'depst':
            name_parts.append(self.compound)
            if self.context_type:
                name_parts.append(f'{self.context_type}-tsz_{self.max_size}')
            else:
                name_parts.append(f'tsz_{self.max_size}')
        elif self.compound == 'subtree-depst':
            name_parts.append(self.compound)
        elif self.compound == 'ngram':
            name_parts.append(self.compound)
            if self.no_sibs:
                name_parts.append('no_sibs')
        if self.min_freq > 1:
            name_parts.append(f'minfreq_{self.min_freq}')
        name_parts.append(self.algo)
        name_parts.append(str(self.n_trn))
        if self.algo not in ['rand', 'entropy', 'weighted-entropy', 'set-cover']:
            name_parts.append(self.ex_sel)
        if self.algo.startswith('entropy'):
            if self.freq_obj_lbd: name_parts.append(str(self.freq_obj_lbd))
        name_parts.append(str(self.seed))
        return '-'.join(name_parts)

    def get_name(self):
        name_parts = []
        if self.anon: name_parts.append('anon')
        # name_parts.append(self.compound)
        assert self.compound in [None, 'ngram', 'template', 'subtree', 'depst', 'subtree-depst']
        if self.compound == 'subtree' or self.compound == 'depst':
            name_parts.append(self.compound)
            if self.context_type:
                name_parts.append(f'{self.context_type}-tsz_{self.max_size}')
            else:
                name_parts.append(f'tsz_{self.max_size}')
        elif self.compound == 'subtree-depst':
            name_parts.append(self.compound)
        elif self.compound == 'ngram':
            name_parts.append(self.compound)
            if self.no_sibs:
                name_parts.append('no_sibs')
        elif self.compound == 'template':
            name_parts.append(self.compound)
        if self.min_freq > 1:
            name_parts.append(f'minfreq_{self.min_freq}')
        name_parts.append(self.algo)
        name_parts.append(str(self.n_trn))
        if self.algo not in ['rand', 'entropy', 'weighted-entropy', 'set-cover']:
            name_parts.append(self.ex_sel)
        if self.algo.startswith('entropy'):
            if self.freq_obj_lbd: name_parts.append(str(self.freq_obj_lbd))
        name_parts.append(str(self.seed))
        return '-'.join(name_parts)

    def get_split_dir(self):
        return Path('datasets') / self.dataset / self.subdataset / 'splits' / self.split_type / f'split_{self.split_seed}'

    def get_split_file(self):
        if self.n_trn > 0:
            return self.get_split_dir() / 'subsamples' / (self.compound or self.algo) / f'{self.get_name()}.json'
        else:
            return self.get_split_dir() / f'{(self.subdataset or self.dataset)}.json'

    def get_exp_dir(self, exp_dir='', name = '', model_name = 'bart'):
        exp_dir = Path(exp_dir) / name / self.dataset / self.subdataset / model_name / self.split_type / f'split_{self.split_seed}'
        if self.n_trn > 0:
            return  exp_dir / 'subsamples' / (self.compound or self.algo) / self.get_name()
        else:
            return exp_dir / (self.subdataset or self.dataset)


    @classmethod
    def get_key_order(cls):
        return ['dataset', 'subdataset', 'split_type', 'split_seed',
                'n_trn', 'seed', 'anon', 'compound', 'min_freq', 'no_sibs',
                'context_type', 'max_size', 'algo', 'ex_sel', 'freq_obj_lbd']

def get_split_dir(dataset, subdataset, split_type, split_seed):
    return Path('datasets') / dataset / subdataset / 'splits' / split_type / f'split_{split_seed}'

def get_split_file(split_dir, params):
    return split_dir / 'subsamples' / (params.compound or params.algo) / f'{params.get_name()}.json'

def get_n_trn_l(ds_name, split_type, key=None):
    n_trn_l_d = {
        'covr': {
            'iid': defaultdict(lambda: [50, 100, 200, 300, 400], {}),
            'template/0.5': defaultdict(lambda: [50, 100, 200, 300, 400], {}),
            'subtree': defaultdict(lambda: [50, 100, 300], {}),
            'nosplit': defaultdict(lambda: [50, 100, 300, 600, 1000, 3000], {}),
        },
        'atis': {
            'iid': defaultdict(lambda: [100, 300, 600, 1000, 2000, 3000], {}),
            'template/0.2': defaultdict(lambda: [100, 300, 600, 1000, 2000, 3000], {}),
            'subtree': defaultdict(lambda: [100, 300, 600, 1000], {}),
            'nosplit': defaultdict(lambda: [50, 100, 300, 600, 1000, 3000], {}),
        },
        'overnight': {
            'iid': defaultdict(lambda: [50, 100, 200, 300, 400], {}),
            'template/0.2': defaultdict(lambda: [100, 300, 600, 1000, 1500], {}),
            'subtree': defaultdict(lambda: [100, 300, 600, 1000], {}),
            'nosplit': defaultdict(lambda: [50, 100, 300, 600, 1000, 3000], {}),
        },
        'thingtalk': {
            'iid': defaultdict(lambda: [50, 100, 300, 600], {}),
            'template/0.2': defaultdict(lambda: [100, 300, 600, 1000], {}),
            'subtree': defaultdict(lambda: [100, 300, 600, 1000], {}),
            'nosplit': defaultdict(lambda: [50, 100, 300, 600, 1000, 3000], {}),
        },
        'smcalflow': {
            'iid': defaultdict(lambda: [1000, 3000, 6000, 15000, 30000, 50000], {}),
            'template/0.2': defaultdict(lambda: [1000, 3000, 6000, 15000, 30000, 50000], {}),
            'subtree': defaultdict(lambda: [1000, 3000, 6000], {}),
            'nosplit': defaultdict(lambda: [1000, 3000, 6000, 10000], {}),
        },
        'smcalflow-uncommon': {
            'iid': defaultdict(lambda: [1000, 3000, 6000, 15000], {}),
            'template/0.2': defaultdict(lambda: [1000, 3000, 6000, 15000], {}),
            'subtree': defaultdict(lambda: [1000, 3000, 6000], {}),
            'nosplit': defaultdict(lambda: [1000, 3000, 6000, 10000], {}),
        },
        'smcalflow-nostr': {
            'iid': defaultdict(lambda: [1000, 3000], {}),
            'template/0.2': defaultdict(lambda: [1000, 3000], {}),
            'subtree': defaultdict(lambda: [1000, 3000], {}),
        }
    }
    n_trn_l = n_trn_l_d[ds_name][split_type][key]
    return n_trn_l

def _get_subsample_params(subsample_type, n_trn_l, seeds, **kwargs):
    key_order = SubsampleParams.get_key_order()
    max_st_size_l = [2, 3, 4, 5]
    anon_l = [True, False][1:]
    subsample_params_l_d = {
        'full': SubsampleParams(**kwargs, n_trn=-1, algo='rand'),
        'random': SubsampleParams(**kwargs, n_trn=n_trn_l, algo='rand', seed=seeds),
        'ngram': SubsampleParams(**kwargs,
            n_trn=n_trn_l,  anon=anon_l, compound='ngram', no_sibs=False,
            algo=['1', '5_cyc_fix'], ex_sel='random', seed=seeds),
        'template': SubsampleParams(**kwargs,
            n_trn=n_trn_l, compound='template',
            algo=['0', '1_cyc_fix', '5_cyc_fix'], ex_sel='random', seed=seeds),
        'subtree': SubsampleParams(**kwargs,
            n_trn=n_trn_l, anon=anon_l, compound='subtree', min_freq=[1, 2][:1],
            context_type=[None, 'nt', 'rule'][:1], max_size=max_st_size_l[2:3],
            algo=['5_cyc_fix'],
            ex_sel=['random', 'new_template', 'mf_new_template'][:2], seed=seeds
        ),
    }[subsample_type].get_settings(key_order)
    return subsample_params_l_d

def get_subsample_params_single_split(
        dataset: str = 'covr', subdataset: str = '', split_type: str = 'iid',
        split_seed: str = 0, subsample_type_l: list[str] = ['random'], n_trn_l: list[int] = None,
        reverse_n_trn_l: bool = False, verbose: bool = True,
        print_params: bool = False, print_paths: bool = False,):
    seeds = list(range(3))
    n_trn_l = n_trn_l or get_n_trn_l(dataset, split_type)
    if reverse_n_trn_l: n_trn_l = n_trn_l[::-1]
    common_kwargs = dict(dataset=dataset, subdataset=subdataset, split_type=split_type, split_seed=split_seed)
    params_l = [p for type in subsample_type_l
                for p in _get_subsample_params(type, n_trn_l, seeds, **common_kwargs)]
    if verbose: print(f'Total {len(params_l)} subsamples for [yellow]{split_type}[/yellow] split {split_seed} of [red]{dataset}[/red]')
    if print_params:
        for i, p in enumerate(params_l):
            print(f'{i}\t{p.get_split_file().exists()}\t{p.get_name()}\t{p}')
    if print_paths:
        for i, p in enumerate(params_l):
            print(f'{i}\t{p.get_split_file().exists()}\t{p.get_split_file()}')
    return params_l

def get_subsample_params(
        dataset_l: list[str], split_type_l: list[str], split_seed_l: int,
        subsample_type_l: list[str], reverse_n_trn_l: bool = False,
        print_params: bool = False, print_paths: bool = False):
    # dataset_l = dataset.split(';') if dataset != 'all' \
    #     else ['covr', 'atis', 'overnight', 'thingtalk', 'smcalflow']
    # split_type_l = split_type.split(';') if split_type != 'all' \
    #     else ['iid', 'template', 'subtree']
    # split_seed_l = [int(s) for s in split_seed.split(';')] if split_seed != '-1' else range(4)
    # subsample_type_l = subsample_type.split(';') if subsample_type != 'all' \
    #     else ['random', 'ngram', 'template', 'subtree']
    get_subdataset = lambda dataset: '' if dataset != 'overnight' else 'socialnetwork'
    def full_split_type(dataset, split_type):
        if split_type != 'template': return split_type
        else: return f'template/{0.5 if dataset.startswith("covr") else 0.2}'
    params_l = []
    for dataset, split_type, split_seed in product(dataset_l, split_type_l, split_seed_l):
        params_l.extend(get_subsample_params_single_split(
            dataset, get_subdataset(dataset), full_split_type(dataset, split_type),
            split_seed, subsample_type_l, reverse_n_trn_l=reverse_n_trn_l,
            verbose=False, print_params=print_params, print_paths=print_paths))
    return params_l

def get_vals(vals_str, type, all):
    all_val = {str: 'all', int: '-1'}[type]
    if vals_str == all_val: return all
    return [type(v) for v in vals_str.split(';')]
