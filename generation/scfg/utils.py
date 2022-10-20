import nltk
import numpy.random as npr
from functools import cache
from itertools import product
from collections import Counter, defaultdict
from utils.progress import track

def remove_puncts_from_parse_tree(parse_tree):
    children = [c for c in parse_tree if c != '(' and c != ')' and c != ',']
    children = [remove_puncts_from_parse_tree(c) if isinstance(c, nltk.Tree) else c for c in children]
    return nltk.Tree(parse_tree.label(), children)

def random_infinite_generator(l):
    assert(l)
    while 1:
        idxes = list(range(len(l)))
        npr.shuffle(idxes)
        for idx in idxes:
            yield l[idx]

@cache
def all_partitions(size):
    import more_itertools
    return [[len(part) for part in p] for p in more_itertools.partitions(range(size))]

@cache
def partitions(size, n_parts):
    return [p for p in all_partitions(size) if len(p) == n_parts]

# get_tree_size = lambda t: len(t.productions())
get_tree_size = lambda t: 1 + sum([get_tree_size(c) for c in t]) if isinstance(t, nltk.Tree) else 1

get_tuple_size = lambda t: 1 + sum([get_tuple_size(c) for c in t[1]]) if isinstance(t, tuple) else 1

get_tree_root = lambda t: t.label() if isinstance(t, nltk.Tree) else t

def target_to_ast(target, verbose=False):
    from experiments.analysis.utils.lf_utils import tokenize_lf
    from experiments.analysis.ast_parser import ASTParser
    from nltk import Tree
    tokens = tokenize_lf(target)
    ast_parser = ASTParser(config={})
    ast = ast_parser.get_ast(tokens)
    def post_process(_ast):
        if isinstance(_ast[0], list):
            # will ignore ast[1:]: eg. will ignore '[ NUMBER_VAL ]' in thingtalk
            return post_process(_ast[0]) if len(_ast[0]) > 0 else []
        elif isinstance(_ast, list):
            if len(_ast) == 1:
                return _ast[0]
            else:
                return Tree(_ast[0], [
                    post_process(c)
                    # if isinstance(c, list) and len(c) > 0 else c
                    if isinstance(c, list) else c
                    for c in _ast[1:] if c != []])
            # return [post_process(a) if isinstance(a, list) else a for a in _ast]
        else:
            raise ValueError(_ast)

    tree = post_process(ast)
    # tree = Tree.fromlist(post_process(ast))
    # tree = Tree.fromlist(ast[0])
    if verbose:
        tree.pretty_print(unicodelines=True)
    return tree

def target_to_ast_calflow(target):
    from experiments.analysis.utils.lf_utils import tokenize_lf
    from nltk import Tree
    def concat_strings(tree):
        children = []
        parts = []
        in_string = False
        for child in tree:
            if isinstance(child, Tree):
                children.append(concat_strings(child))
            else:
                assert isinstance(child, str)
                if child[0] == child[-1] == '"':
                    children.append(child)
                elif child[0] == '"':
                    parts.append(child)
                    in_string = True
                elif child[-1] == '"':
                    parts.append(child)
                    children.append(' '.join(parts))
                    parts = []
                    in_string = False
                elif in_string:
                    assert len(parts) > 0, f'{tree}\t{target}'
                    parts.append(child)
                else:
                    children.append(child)
        assert not in_string, f'{tree}\t{target}'
        return Tree(tree.label(), children)
    tree_ini = Tree.fromstring(f"(<s> ( {target} ))")
    tree_fin = concat_strings(tree_ini)
    # assert str(tree_ini) == str(tree_fin), target
    return tree_fin

def get_ngrams(t, result=None):
    if result is None: result = Counter()
    assert isinstance(t, nltk.Tree)
    for i in range(len(t)):
        result[(f'p:{t.label()}', f'c:{get_tree_root(t[i])}')] += 1
        if i > 0:
            result[(f's:{get_tree_root(t[i-1])}', f'c:{get_tree_root(t[i])}')] += 1
    for c in t:
        if isinstance(c, nltk.Tree):
            get_ngrams(c, result)
    return result

def is_subtree(t1: nltk.Tree, t2: nltk.Tree) -> bool:
    if type(t1) != type(t2): return False
    if type(t1) == str: return t1 == t2
    if t1.label() != t2.label(): return False
    if len(t2) == 0: return True
    elif len(t2) == 1:
        for i in range(len(t1)):
            if is_subtree(t1[i], t2[0]): return True
        return False
    elif len(t1) == len(t2):
        for i in range(len(t1)):
            if not is_subtree(t1[i], t2[i]): return False
        return True
    else:
        return False

def tree_to_tuple(t):
    if isinstance(t, nltk.Tree):
        return (t.label(), tuple([tree_to_tuple(c) for c in t]))
    else:
        return t

def tuple_to_tree(t):
    if isinstance(t, tuple):
        return nltk.Tree(t[0], [tuple_to_tree(c) for c in t[1]])
    else:
        return t

def get_subtrees(t: nltk.Tree, max_size: int, context_type: str = None,
                 path: nltk.Tree = None, node: nltk.Tree = None,
                 result: list[nltk.Tree] = None) -> tuple[list[nltk.Tree], list[nltk.Tree]]:
    assert isinstance(t, nltk.Tree)

    if context_type in ['rule', 'nt'] and (path == None or node == None):
        assert path == None and node == None
        path = node = nltk.Tree(t.label(), [])

    if result == None:
        # result = set()
        result = defaultdict(lambda: 0)

    subtrees = defaultdict(lambda: 0, {(t.label(), tuple()): 1})
    # subtrees = set()
    all_child_trees = []
    for i in range(len(t)):
        if isinstance(t[i], nltk.Tree):
            if context_type == 'rule':
                node[:] = [nltk.Tree(c.label(), []) if isinstance(c, nltk.Tree) else c
                           for c in t]
                c_node = node[i]
            elif context_type == 'nt':
                node[:] = [nltk.Tree(t[i].label(), [])]
                c_node = node[0]
            else:
                c_node = None
            c_trees = get_subtrees(t[i], max_size, context_type,
                                   path, c_node, result)[0]
        else:
            c_trees = [t[i]]
            # c_trees = []
        all_child_trees.append(c_trees)
    for c_sizes in partitions(max_size + len(t) - 1, len(t)):
        c_trees = [[t for t in c_trees if get_tuple_size(t) < c_sz]
                    for c_sz, c_trees in zip(c_sizes, all_child_trees) if c_sz > 1]
        for c_tree_comb in product(*c_trees):
            # subtrees.add(tree_to_tuple(nltk.Tree(t.label(), c_tree_comb)))
            subtrees[(t.label(), c_tree_comb)] += 1
    if context_type in ['rule', 'nt']:
        # subtrees = [tuple_to_tree(st) for st in subtrees]
        for st in subtrees:
            node[:] = tuple_to_tree(st)[:]
            result[tree_to_tuple(path)] += 1
    else:
        result |= subtrees
        # result.update(subtrees)
        # subtrees = [tuple_to_tree(st) for st in subtrees]
    # print(subtrees)

    if node is not None: node[:] = []
    return list(subtrees.keys()), list(result.keys())

def get_atis_rels(tree):
    # trees = [target_to_ast(tgt) for tgt in tgts]
    if tree.label() != 'and':
        if isinstance(tree[0], nltk.Tree): return get_atis_rels(tree[0])
        else: return []
    else:
        return [c.split(' ')[0] if isinstance(c, str) else c.label() for c in tree]

def percolate_rel_args(tree, rels):
    # if tree.label().split()[0] in rels and len(tree) == 0:
    if isinstance(tree, str) and tree.split()[0] in rels and tree.split()[1].startswith('$'):
        assert len(tree.split()) in [2, 3, 5], tree
        if len(tree.split()) == 3: print(tree)
        if len(tree.split()) == 2:
            children = [tree.split()[1]]
        else:
            children = [tree.split()[1],
                        ' '.join(tree.split()[2:])]
        return nltk.Tree(tree.split()[0], children)
    elif isinstance(tree, nltk.Tree):
        return nltk.Tree(tree.label(), [percolate_rel_args(c, rels) for c in tree])
    else:
        return tree

if 0:
    import pandas as pd
    from utils.anon_fns import anonymize_covr_target, anonymize_atis_target, anonymize_thingtalk, anonymize_overnight_target
    ds = ['covr', 'atis', 'thingtalk', 'smcalflow', 'geoquery', ('overnight', 'restaurants')][3]
    if type(ds) == str: df = pd.read_json(f'datasets/{ds}/{ds}.all.jsonl', lines=True)
    else: df = pd.read_json(f'datasets/{ds[0]}/{ds[1]}.all.jsonl', lines=True)
    trees = [target_to_ast(target) for target in df['target']]
    anon_fn = {'covr': anonymize_covr_target, 'atis': anonymize_atis_target, 'thingtalk': anonymize_thingtalk, ('overnight', 'restaurants'): anonymize_overnight_target}[ds]
    df['anonymized_target'] = df.target.apply(lambda t: anon_fn(t))
    anon_trees = [target_to_ast(t) for t in df.anonymized_target]
    all_subtrees = set()
    for t in track(trees):
        get_subtrees(t, 4, None, result=all_subtrees)
    all_subtrees = [tuple_to_tree(st) for st in all_subtrees]

    rels = set()
    for t in track(trees):
        rels.update(get_atis_rels(t))

    rels = [
        'approx_arrival_time',
        'air_taxi_operation',
        'day_arrival',
        'meal',
        'days_from_today',
        'equals',
        'daily',
        'day_after_tomorrow',
        'day_number',
        'arrival_time',
        'month_return',
        'services',
        'connecting',
        'class_type',
        'loc',
        'round_trip',
        'flight',
        'jet',
        'oneway',
        'day_number_return',
        'exists $2',
        'fare_basis_code',
        'minutes_distant',
        'approx_return_time',
        'time_elapsed',
        'before_day',
        'named',
        'meal_code',
        'during_day_arrival',
        'approx_departure_time',
        'turboprop',
        'next_days',
        'stops',
        'today',
        'to',
        'city',
        'or',
        'and',
        'during_day',
        'time_zone_code',
        'tomorrow',
        'aircraft',
        'day_number_arrival',
        'airline',
        'day_return',
        'exists $1',
        'overnight',
        'taxi',
        'departure_time',
        'flight_number',
        'ground_transport',
        'equals $1',
        'tomorrow_arrival',
        'from',
        'after_day',
        'weekday',
        'tonight',
        'stop',
        'airport',
        'has_stops',
        'fare',
        'manufacturer',
        'not',
        'economy',
        'month_arrival',
        'from_airport',
        'arrival_month',
        'miles_distant',
        'rapid_transit',
        'limousine',
        'booking_class',
        'equals $0',
        'aircraft_code',
        'rental_car',
        'has_meal',
        'nonstop',
        'discounted',
        'to_city',
        'month',
        'day',
        'year'
    ]
    perc_trees = [percolate_rel_args(t, rels) for t in all_subtrees]
    all_subtrees2 = set()
    for t in track(perc_trees):
        get_subtrees(t, 4, None, result=all_subtrees2)
    all_subtrees2 = [tuple_to_tree(st) for st in all_subtrees2]
