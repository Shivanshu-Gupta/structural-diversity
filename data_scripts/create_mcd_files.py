import argparse
import json
import os
from os import listdir

from experiments.analysis.ast_parser import ASTParser
from data_scripts.utils.constants import OVERNIGHT_DOMAINS
from data_scripts.split_datasets import ANON_FN_MAP
from experiments.analysis.utils.lf_utils import tokenize_lf

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default='covr')
    args.add_argument('--no-anon', action='store_true')
    args = args.parse_args()

    if args.dataset == 'covr':
        rows = [json.loads(l) for l in open("../covr/train_9.jsonl")][:5000]
        splits_dir_base = f"../{args.dataset}/splits/cfg/seed_0"
        splits_names = listdir(splits_dir_base)
    elif args.dataset == 'overnight':
        rows = []
        for domain in OVERNIGHT_DOMAINS:
            rows += [json.loads(l) for l in open(f"../{args.dataset}/{domain}.all.jsonl")]
        splits_dir_base = f"../{args.dataset}/splits/template"
        splits_names = [f"split_{i}/{domain}.json" for i in range(5) for domain in OVERNIGHT_DOMAINS]
    else:
        raise ValueError()

    ast_parser = ASTParser(config={})
    anon_fn = ANON_FN_MAP[args.dataset]
    target_to_tree = {}
    for ex in rows:
        try:
            if not args.no_anon:
                anonyimzed_target = anon_fn(ex['target']).replace('_', '')  # remove underscores from anonymization since we don't want anonymization to affect the parsing
            else:
                anonyimzed_target = ex['target']
            target_to_tree[ex['qid']] = ast_parser.get_ast(tokenize_lf(anonyimzed_target))
        except Exception as e:
            print(ex['target'])
            print(e)

    dataset = args.dataset
    if args.no_anon:
        dataset += "_no_anon"

    out_dir_base = f"../../../mcd-splitter/data/{dataset}"
    os.makedirs(out_dir_base, exist_ok=True)
    json.dump(target_to_tree, open(f"{out_dir_base}/all.json", "wt"))

    splits_files_pointers = {}

    for split in splits_names:
        split_name = split.split('.json')[-2].replace('/', '_')
        with open(f"{splits_dir_base}/{split}") as f:
            split_info = json.load(f)

        train_file_name = f"{split_name}_train.json"
        test_file_name = f"{split_name}_test.json"
        with open(f"{out_dir_base}/{train_file_name}", "wt") as f_out:
            train_examples = split_info.get('train_examples') or split_info.get('train')
            json.dump({qid: target_to_tree[qid] for qid in train_examples if qid in target_to_tree}, f_out)
        with open(f"{out_dir_base}/{test_file_name}", "wt") as f_out:
            test_examples = split_info.get('test_examples') or split_info.get('test')
            json.dump({qid: target_to_tree[qid] for qid in test_examples if qid in target_to_tree}, f_out)

        splits_files_pointers[split_name] = {
            'train': f'../data/{dataset}/' + train_file_name,
            'test': f'../data/{dataset}/' + test_file_name
        }
    with open(f"{out_dir_base}/splits.json", "wt") as f_out:
        json.dump(splits_files_pointers, f_out)
