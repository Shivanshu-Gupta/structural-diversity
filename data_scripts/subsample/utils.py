import pandas as pd
from functools import partial

from data_scripts.subsample_pool import get_subtrees, get_templates, get_sent_ngrams, iterative_subsample

def get_samples(ds_name, sample_size):
    from data_scripts.utils.anon_fns import anonymize_covr_target, anonymize_atis_target, anonymize_overnight_target, anonymize_thingtalk, anonymize_smcalflow_target
    if ds_name != 'overnight':
        trn_pool = full_df = pd.read_json(f'../datasets/{ds_name}/{ds_name}.combined.jsonl', lines=True)
    else:
        trn_pool = full_df = pd.read_json(f'../datasets/{ds_name}/socialnetwork.combined.jsonl', lines=True)
    if 'anonymized_target' not in trn_pool.columns:
        anon_fn = {'covr': anonymize_covr_target,
                   'atis': anonymize_atis_target,
                   'thingtalk': anonymize_thingtalk,
                   'overnight': anonymize_overnight_target,
                   'smcalflow': anonymize_smcalflow_target,
                   'smcalflow-uncommon': anonymize_smcalflow_target,
                   'smcalflow-nostr': anonymize_smcalflow_target}[ds_name]
        trn_pool['anonymized_target'] = trn_pool.target.apply(anon_fn)
    max_size = 4
    anon = False
    subtree2ex, ex2subtree, _ = get_subtrees(
        trn_pool, ds_name, max_size=max_size, context_type='', anon=anon)
    template2ex, ex2template, _ = get_templates(trn_pool)
    ex2template = {ex: list(d.keys())[0] for ex, d in ex2template.items()}

    diverse_qids = iterative_subsample(
        '5_cyc_fix', subtree2ex, ex2subtree, None, template2ex, ex2template,
        n_trn=sample_size, seed=0, ex_sel='new_template', show_progress=True)

    from sklearn.model_selection import train_test_split
    _, random_qids = train_test_split(list(full_df.qid), test_size=sample_size, random_state=0)

    diverse_df = full_df.query('qid in @diverse_qids')
    random_df = full_df.query('qid in @random_qids')
    dfs = dict(full=full_df, random=random_df, diverse=diverse_df)

    get_subtrees_fn = partial(get_subtrees, ds_name=ds_name, max_size=max_size, context_type='', anon=anon)
    subtree2ex_d = {name: subtree2ex if name == 'full' else get_subtrees_fn(df)[0]
                    for name, df in dfs.items()}

    return dfs, subtree2ex_d