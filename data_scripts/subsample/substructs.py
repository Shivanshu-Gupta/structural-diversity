import numpy as np
from collections import defaultdict, Counter

from utils.progress import track

def get_ngrams(trn_pool, no_sibs, anon, return_sims=False):
    from experiments.analysis.ngrams.ngram_extractor_ast import NgramsExtractorAST
    from experiments.analysis.tokens_similarity import TokenSimilarityExtractor, compute_ngrams_similarity
    from experiments.analysis.utils.lf_utils import tokenize_lf

    ngrams_extractor = NgramsExtractorAST({
        'type': 'ast',
        'n': 2,
        'add_parent_child': True,
        'add_adjacent_sibling': not no_sibs,
    })
    all_ngrams = set()
    ngram2ex = defaultdict(Counter)
    ex2ngram = {}
    targets = trn_pool.anonymized_target if anon else trn_pool.target
    for qid, target in track(zip(trn_pool.qid, targets),
                            total=trn_pool.shape[0],
                            description="Extracting ngrams"):
        tokens = tokenize_lf(target, add_sos=True)
        ex_ngrams = ngrams_extractor.get_ngrams_from_target(tokens, freqs=True)
        ex2ngram[qid] = ex_ngrams
        for ngram, freq in ex_ngrams.items():
            all_ngrams.add(ngram)
            ngram2ex[ngram][qid] += freq
    ngram_sims = None
    if return_sims:
        tok_sims = TokenSimilarityExtractor(all_ngrams).find_all_similarities_between_tokens()
        ngram_sims = np.array([[compute_ngrams_similarity(ng1, ng2, tok_sims)
                                for ng2 in ngram2ex]
                            for ng1 in track(ngram2ex, description='Computing ngrams similarities')])
    print(f'{len(all_ngrams)} ngrams with anon={anon}')
    return {k: dict(v) for k, v in ngram2ex.items()}, ex2ngram, ngram_sims

def get_templates(trn_pool):
    all_templates = set()
    template2ex = defaultdict(dict)
    ex2template = {}
    targets = trn_pool.anonymized_target
    for qid, target in track(zip(trn_pool.qid, targets),
                            total=trn_pool.shape[0],
                            description="Extracting templates"):
        ex2template[qid] = {target: 1}
        all_templates.add(target)
        template2ex[target][qid] = 1
    print(f'{len(all_templates)} templates')
    return dict(template2ex), ex2template, None

def get_subtrees(trn_pool, ds_name, max_size, context_type, anon):
    from data_scripts.utils.anon_fns import anonymize_smcalflow_target
    from generation.scfg.utils import target_to_ast, target_to_ast_calflow, get_subtrees as get_subtrees_fn, tuple_to_tree

    if ds_name == 'smcalflow':
        target_to_ast = target_to_ast_calflow
    # trees = [target_to_ast(target) for target in trn_pool.target]
    all_subtrees = set()
    subtree2ex = defaultdict(dict)
    ex2subtree = {}
    if not anon:
        targets = trn_pool.target
    elif ds_name != 'smcalflow':
        targets = trn_pool.anonymized_target
    else:
        targets = [anonymize_smcalflow_target(tgt, anonymize_level=0.5)
                   for tgt in trn_pool.target]
    for qid, target in track(zip(trn_pool.qid, targets),
                            total=trn_pool.shape[0],
                            description="Extracting subtrees"):
        tree = target_to_ast(target)
        _, ex_subtrees = get_subtrees_fn(tree, max_size, context_type)
        ex2subtree[qid] = {st: 1 for st in ex_subtrees}
        for subtree in ex_subtrees:
            all_subtrees.add(subtree)
            subtree2ex[subtree][qid] = 1
    print(f'{len(all_subtrees)} subtrees with max size {max_size} and context type {context_type} and anon={anon}')
    # all_subtrees = [tuple_to_tree(st) for st in all_subtrees]
    # return {k: dict(v) for k, v in subtree2ex.items()}, ex2subtree, None
    return dict(subtree2ex), ex2subtree, None

def to_nltk_tree(node):
    from nltk import Tree
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_
