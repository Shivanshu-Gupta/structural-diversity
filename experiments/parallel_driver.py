import os, sys
import queue
import subprocess
import shlex
import json
import typer
import pandas as pd

from glob import glob
from rich import print
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend


from data_scripts.subsample.subsamples import SubsampleParams, get_subsample_params
from data_scripts.subsample_pool import get_vals

app = typer.Typer()

q = queue.Queue()
palette = ['r', 'g', 'b', 'y', 'c']
pd.set_option("display.precision", 2)
is_notebook = False
df_print = print

def get_run_dir(root_dir, expname):
    return Path(root_dir) / 'runs' / expname

def is_completed(root_dir, expname):
    run_dir = get_run_dir(root_dir, expname)
    logfile = run_dir / 'out.log'
    # print(run_dir)
    if not logfile.exists(): return False
    log = '\n'.join(logfile.open().readlines())
    # os.path.getmtime(logfile)
    return logfile.exists() and ('archiving weights and vocabulary' in log
                                 or 'Exception: Reached stopping accuracy' in log)

def run_train(idx, root_dir, expname, cmd, debug=False):
    gpu = q.get(block=True)
    logfile = Path(root_dir) / 'outputs' / expname / 'driver.log'
    cmd += f' --gpu {gpu}'
    # print(idx, cmd, '>', logfile)
    print(idx, cmd)
    os.makedirs(logfile.parent, exist_ok=True)
    if not debug:
        args = shlex.split(cmd)
        process = subprocess.Popen(args, stdout=logfile.open('w'), stderr=subprocess.STDOUT)
        ret = process.wait()
    q.put(gpu)

def get_metrics(root_dir, expname):
    # print(expname)
    completed = is_completed(root_dir, expname)
    val_accs = []
    for epoch in range(50):
        metricsfile: Path = get_run_dir(root_dir, expname) / f'metrics_epoch_{epoch}.json'
        if metricsfile.exists():
            metrics = json.load(metricsfile.open())
            val_accs.append(metrics['validation_accuracy'] * 100)
    return max(val_accs) if val_accs else None, val_accs, len(val_accs), completed

def load_results(root_dir, params_l, get_expname):
    rows = [p.to_dict() for p in params_l]
    resultsdf = pd.DataFrame(rows, columns=rows[0].keys())

    with Parallel(n_jobs=50) as parallel:
        result_cols = ['best_val_acc', 'val_accs', 'n_epochs', 'completed']
        resultsdf[result_cols] = parallel(delayed(get_metrics)(
            root_dir, get_expname(p)) for p in params_l)
    # print(resultsdf)
    # resultsdf.to_pickle(root_dir / 'outputs'/ 'result.pkl')
    return resultsdf

def train_all(root_dir, params_l, get_expname, get_cmd, debug=False, n_cocurrent_jobs=8):
    def train_wrapper(i, params):
        print(f'  > {i}/{len(params_l)} {params}')
        expname, cmd = get_expname(params), get_cmd(params)
        run_dir = get_run_dir(root_dir, expname)
        run_train(i, root_dir, expname, cmd, debug)
        if (run_dir / 'model.tar.gz').exists(): os.remove(run_dir / 'model.tar.gz')
        if (run_dir / 'best.th').exists(): os.remove(run_dir / 'best.th')
        print(f'  < {i}/{len(params_l)} {params}')

    while params_l:
        with Parallel(n_jobs=n_cocurrent_jobs, require='sharedmem', verbose=True) as parallel:
            parallel(delayed(train_wrapper)(i, params) for i, params in enumerate(params_l))
        if debug: break
        completed_l = [is_completed(root_dir, get_expname(p))
                       for p in params_l]
        if not any(completed_l):
            print('all jobs failed')
            break
        params_l = [p for p, c in zip(params_l, completed_l) if not c]

# @app.command()
# def exp_1(
#         name, dataset: list[str] = typer.Option(['covr']),
#         split_type: list[str] = typer.Option('all'),
#         split_seed: list[int] = typer.Option(-1),
#         subsample_type: list[str] = typer.Option('all'),
#         print_params: bool = False, root_dir: str = '',
#         only_incomplete: bool = False, debug: bool = False,):
@app.command()
def exp_1(
        name, root_dir: str = 'experiments',
        dataset: str = 'covr', split_type: str = 'all',
        split_seed: str = '-1', subsample_type: str = 'all',
        only_incomplete: bool = False, debug: bool = False,
        print_params: bool = False, print_cmds: bool = False,
        train: bool = False, results: bool = False,
        n_gpus: int = 8, n_jobs_per_gpu: int = 1):
    dataset = get_vals(dataset, str, ['covr', 'atis', 'overnight', 'thingtalk', 'smcalflow-uncommon'])
    split_type = get_vals(split_type, str, ['iid', 'template', 'subtree'])
    split_seed = get_vals(split_seed, int, range(4))
    subsample_type = get_vals(subsample_type, str, ['random', 'ngram', 'template', 'subtree'])
    params_l = get_subsample_params(dataset, split_type, split_seed, subsample_type)
    model_name = 'bart'

    def get_expname(p: SubsampleParams):
        expname = Path(name) / p.dataset / p.subdataset / model_name / p.split_type / f'split_{p.split_seed}'
        if p.n_trn > 0:
            expname = expname / 'subsamples' / (p.compound or p.algo) / p.get_name()
        else:
            expname = expname / (p.subdataset or p.dataset)
        return str(expname)

    def get_cmd(p: SubsampleParams):
        data_path = Path('datasets') / p.dataset  / f'{p.subdataset or p.dataset}.combined.jsonl'
        assert os.path.exists(data_path), f'{data_path} does not exist'

        split_path = p.get_split_file()
        assert os.path.exists(split_path), f'{split_path} does not exist'

        # read_loops = {100: 30, 300: 20, 600: 15, 1000: 10, 5000: 8}[n_trn]
        # read_loops = {50: 15, 100: 12, 300: 10, 600: 8, 1000: 8, 5000: 8}[subsample_params.n_trn]
        ds_name = p.dataset.split('-')[0]
        if not ds_name.startswith('smcalflow'):
            n_epochs = {-1: 10, 50: 20, 100: 16, 200: 14, 300: 12, 400: 12, 600: 10, 1000: 10, 1500: 10, 2000: 10, 3000: 10}[p.n_trn]
        else:
            # n_epochs = {50: 20, 100: 20, 300: 18, 600: 16, 1000: 20}[subsample_params.n_trn]
            n_epochs = 30 if p.n_trn > 0 else 15
        cmd = f'python experiments/train.py --name "{name}" --dataset "{ds_name}"'
        if p.dataset == 'overnight':
            cmd += f' --sub-dataset "{p.subdataset}"'
        cmd += f' --model-name "{model_name}"'
        cmd += f' --data-path "{str(data_path)}"'
        cmd += f' --split-path "{str(split_path)}"'
        cmd += f' --serialization-dir "{str(get_expname(p))}"'
        cmd += f' --n-epochs {n_epochs}'
        # cmd += f" --overrides '{json.dumps({'dataset_reader.read_loops': read_loops})}'"
        cmd += ' '.join([f' --tag {tag}' for tag in []])
        cmd += ' --force'
        return cmd
    incomplete_params_l = [p for p in params_l if not is_completed(root_dir, get_expname(p))]
    print(f'exp_1: {len(incomplete_params_l)}/{len(params_l)} still to be completed')
    if only_incomplete:
        print('Processing only incomplete jobs')
        params_l = incomplete_params_l
    for i, p in enumerate(params_l):
        if print_params: print(f'{i}/{len(params_l)} {p}')
        if print_cmds: print(f'{i}/{len(params_l)} {get_cmd(p)}')

    if train:
        for _ in range(n_jobs_per_gpu):
            for i in range(n_gpus):
                q.put(i)
        train_all(
            root_dir, params_l, get_expname, get_cmd, debug=debug,
            n_cocurrent_jobs=n_gpus*n_jobs_per_gpu + 1)

    if results:
        resultsdf = load_results(root_dir, params_l, get_expname)
        return resultsdf

@app.command()
def exp_1_results(
        name, root_dir: str = 'experiments',
        dataset: str = 'covr', split_type: str = 'all', subsample_type: str = 'all',
        df_print=print):
    df_print = df_print or print
    dataset = get_vals(dataset, str, ['covr', 'atis', 'overnight', 'thingtalk', 'smcalflow-uncommon'])
    resultsdf_l = []
    for _dataset in dataset:
        print(f'exp_1_results: {_dataset}')
        resultsdf = exp_1(name, root_dir, _dataset, split_type, "-1", subsample_type, results=True)
        df_print(resultsdf.query(f'completed == True').groupby(by=['n_trn', 'split_type', 'compound', 'anon', 'min_freq', 'max_size', 'algo', 'ex_sel'], dropna=False).best_val_acc.mean().unstack(level=1).unstack(level=0).dropna(axis=1, how='all'))
        df_print(resultsdf.query(f'completed == True').groupby(by=['n_trn', 'split_type', 'compound', 'anon', 'min_freq', 'max_size', 'algo', 'ex_sel'], dropna=False).seed.count().unstack(level=1).unstack(level=0).dropna(axis=1, how='all'))
        df_print(resultsdf.query(f'completed == False').groupby(by=['n_trn', 'split_type', 'compound', 'anon', 'min_freq', 'max_size', 'algo', 'ex_sel'], dropna=False).seed.count().unstack(level=1).unstack(level=0).dropna(axis=1, how='all'))
        resultsdf_l.append(resultsdf)

    # df_print(resultsdf.query(f'completed == True').groupby(by=['n_trn', 'split_type', 'compound', 'anon', 'min_freq', 'max_size', 'algo', 'ex_sel', 'split_seed'], dropna=False).best_val_acc.agg(['mean']).unstack(level=1).unstack(level=0).unstack(level=6).dropna(axis=1, how='all'))

    # print(latex_table(resultsdf))
    return resultsdf_l if len(resultsdf_l) > 1 else resultsdf_l[0]

if __name__ == '__main__':
    os.makedirs('outputs/parallel_exps', exist_ok=True)
    app()
