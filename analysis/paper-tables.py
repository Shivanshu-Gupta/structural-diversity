import os
from sys import implementation
import typer
import pandas as pd

from pathlib import Path
from functools import partial

from data_scripts.subsample_pool import get_subtrees, get_ngrams, get_templates
from data_scripts.utils.anon_fns import anonymize_covr_target, anonymize_atis_target, anonymize_overnight_target, anonymize_thingtalk, anonymize_smcalflow_target
from experiments.parallel_driver import exp_1, latex_table
from analysis.mutual_info import get_ds_info as get_midf

app = typer.Typer()

def make_singlecol_table(tabular_string, caption_string, label_string):
    return f"""
\\begingroup
\setlength{{\\tabcolsep}}{{4pt}} % Default value: 6pt
\\renewcommand{{\\arraystretch}}{{0.8}} % Default value: 1
\\begin{{table}}
\centering
\small
{tabular_string.strip()}
\end{{table}}
\endgroup
            """

def make_doublecol_table(tabular_string, caption_string, label_string):
    return f"""
\\begingroup
\setlength{{\\tabcolsep}}{{4pt}} % Default value: 6pt
\\renewcommand{{\\arraystretch}}{{0.8}} % Default value: 1
\\begin{{table*}}
\centering
\small
{tabular_string.strip()}
\caption{{{caption_string}}}
\label{{tab:{label_string}}}
\end{{table*}}
\endgroup
            """

@app.command()
def main_results(output_file: str = 'tables/results.tex'):
    # os.chdir('experiments/')
    print(output_file)
    exp_fn = partial(exp_1, name='final0', dump=True, ben=False, root_dir='experiments')
    res_dfs = {
        'covr': exp_fn(dataset='covr'),
        'atis': exp_fn(dataset='atis'),
        'thingtalk': exp_fn(dataset='thingtalk'),
        'overnight': exp_fn(dataset='overnight', sub_dataset='socialnetwork'),
        'smcalflow': exp_fn(dataset='smcalflow'),
    }
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, 'w') as f:
        for ds, df in res_dfs.items():
            tabular_string = latex_table(df)
            caption_string = f'Complete subsampling results for {ds}.'
            label_string = f'res-{ds.lower()}'
            table_string = make_doublecol_table(tabular_string, caption_string, label_string)
            f.write(table_string)

@app.command()
def stats(output_file: str = 'tables/stats.tex'):
    def get_data_path(dataset, sub_dataset=None):
        return Path('datasets') / dataset  / f'{sub_dataset or dataset}.combined.jsonl'
    ds_paths = {
        'covr': get_data_path('covr'),
        'atis': get_data_path('atis'),
        'thingtalk': get_data_path('thingtalk'),
        'overnight': get_data_path('overnight', 'socialnetwork'),
        'smcalflow': get_data_path('smcalflow'),
    }
    records = []
    for dataset, ds_path in ds_paths.items():
        print(ds_path)
        assert ds_path.exists(), f'{ds_path} does not exist'
        trn_pool = pd.read_json(ds_path, lines=True)
        if 'anonymized_target' not in trn_pool.columns:
            anon_fn = {
                'covr': anonymize_covr_target,
                'atis': anonymize_atis_target,
                'thingtalk': anonymize_thingtalk,
                'overnight': anonymize_overnight_target,
                'smcalflow': anonymize_smcalflow_target,
            }[dataset]
            trn_pool['anonymized_target'] = trn_pool.target.apply(anon_fn)

        record = dict(dataset=dataset)

        records.append(record | dict(compound='instance', count=len(trn_pool)))
        print(dataset, f'{len(trn_pool)} instances')

        template2ex, _, _ = get_templates(trn_pool)
        print(dataset, f'{len(template2ex)} templates')
        records.append(record | dict(compound='template', count=len(template2ex)))

        ngram2ex, _, _ = get_ngrams(trn_pool, False, anon=False)
        print(dataset, f'{len(ngram2ex)} ngrams')
        records.append(record | dict(compound='ngram', count=len(ngram2ex)))

        subtree2ex, _, _ = get_subtrees(trn_pool, ds_name=dataset, max_size=4, context_type='', anon=False)
        records.append(record | dict(compound='subtree', count=len(subtree2ex)))
        print(dataset, f'{len(subtree2ex)} subtrees')

    statsdf = pd.DataFrame(records).rename(columns={'dataset': 'Dataset', 'compound': 'Compound', 'count': 'Count'})
    print(statsdf)
    print(statsdf.pivot(index='Dataset', columns='Compound', values='Count')
          .rename_axis(mapper=None, axis=1))
    tabular_string = statsdf.pivot(index='Dataset',columns='Compound',values='Count')\
        .reset_index()\
        .rename_axis(mapper=None, axis=1)\
        .rename(columns={'Dataset': 'Dataset', 'instance': 'Instances', 'template': 'Templates', 'ngram': 'Bigrams', 'subtree': 'Subtrees'})\
        .to_latex(multirow=True, escape=False, float_format='%0.2f', multicolumn_format='c', index=False)
    table_string = make_singlecol_table(tabular_string)
    print(table_string)
    with open(output_file, 'w') as f:
        f.write(table_string)
    return statsdf

# @app.command()
# def mutual_info(output_file: str = 'tables/mi.tex'):
#     covrdf = get_midf(dataset='covr')
#     atisdf = get_midf(dataset='atis')
#     thingtalkdf = get_midf(dataset='thingtalk')
#     overnightdf = get_midf(dataset='overnight', sub_dataset='socialnetwork')
#     smcalflowdf = get_midf(dataset='smcalflow')
#     dfs = {
#         'covr': covrdf,
#         'atis': atisdf,
#         'thingtalk': thingtalkdf,
#         'smcalflow': smcalflowdf,
#         'overnight': overnightdf,
#     }
#     for k, midf in dfs.items():
#         print(k)

#         print(midf.groupby(by=['n_trn', 'compound', 'max_size', 'algo', 'ex_sel'], dropna=False).mutual_info.mean().unstack(level=0))
#         # print(midf.query('algo == "rand"').groupby(['n_trn']).mutual_info.min())
#         # print(midf.query('compound == "subtree" and algo == "5_cyc_fix"').groupby(['n_trn']).mutual_info.max())
#         print()

if __name__ == '__main__':
    app()

