from collections import defaultdict
from random import Random

import numpy as np
from setuptools._vendor.ordered_set import OrderedSet
from tqdm import tqdm

from experiments.analysis.legacy.easiness_predictor import EasinessPredictor
from experiments.analysis.tokens_similarity import TokenSimilarityExtractor, compute_ngrams_similarity
from experiments.analysis.utils.lf_utils import tokenize_lf
from experiments.analysis.utils.splits_experiments_utils import verify_split
from experiments.analysis.ngrams.ngram_extractor_ast import NgramsExtractorAST


def compute_easiness_score(train_targets, test_targets, add_siblings):
    predictor = EasinessPredictor(train_targets, {
        'ngrams': {
            'type': 'ast',
            'n': 2,
            'add_parent_child': True,
            'add_adjacent_sibling': add_siblings,
        },
        'find_similar': '01',
        'position': False,
        'length': False,
        'random': False
    })

    all_predictions = []

    for ex_target in test_targets:
        easiness_prediction = predictor.predict_easiness_for_examples(ex_target)
        all_predictions.append(easiness_prediction['final_score'])
    return np.mean(all_predictions)


def splits_by_bigrams(example_by_qid,
                      limit_train=None,
                      limit_test=None,
                      random_seed=0,
                      easiness_threshold=0.3,
                      add_siblings=False,
                      p_similar_bigrams=1):
    random = Random(random_seed)

    ngrams_extractor = NgramsExtractorAST({
        'type': 'ast',
        'n': 2,
        'add_parent_child': True,
        'add_adjacent_sibling': add_siblings,
    })

    all_ngrams = OrderedSet()
    ngram_to_qid = defaultdict(set)
    examples = list(example_by_qid.values())
    all_qids = set([ex['qid'] for ex in examples])
    all_targets = set([ex['target'] for ex in examples])
    for ex in tqdm(examples):
        tokens = tokenize_lf(ex['target'], add_sos=True)
        ex_ngrams = ngrams_extractor.get_ngrams_from_target(tokens)
        all_ngrams.update(ex_ngrams)
        for ngram in ex_ngrams:
            ngram_to_qid[ngram].add(ex['qid'])

    similarities = TokenSimilarityExtractor(all_ngrams).find_all_similarities_between_tokens()

    seen_splits = set()
    output_splits = []

    for ngram in all_ngrams:
        similar_ngrams = {}
        for other_ngram in all_ngrams:
            sim = compute_ngrams_similarity(ngram, other_ngram, similarities)
            if sim > 0:
                similar_ngrams[other_ngram] = sim

        for threshold in sorted(set(similar_ngrams.values())):
            ngrams_above_thresholds = selected_ngrams = [sim_ngram for sim_ngram, similarity in similar_ngrams.items() if similarity >= threshold]

            split_key = tuple(sorted(ngrams_above_thresholds))
            if split_key in seen_splits:
                break
            seen_splits.add(split_key)

            if len(selected_ngrams) == 1:
                continue

            if p_similar_bigrams < 1:
                if len(ngrams_above_thresholds) < 4:
                    continue
                selected_ngrams = random.sample(ngrams_above_thresholds, int(p_similar_bigrams*len(ngrams_above_thresholds)))

            all_test_examples = test_examples = {qid for sim_ngram in selected_ngrams for qid in ngram_to_qid[sim_ngram]}
            all_train_examples = train_examples = {qid for qid in all_qids if qid not in all_test_examples}

            if not all_train_examples or not all_test_examples:
                continue
            if len(all_test_examples) > 0.3 * len(examples):
                continue
            if len(all_test_examples) < 100:
                continue

            n_tries = 0
            found_valid_split = False
            while n_tries < 20:  # we try multiple times in case the split was invalid just because of "bad" sampling
                n_tries += 1
                if limit_train and len(all_train_examples) > limit_train:
                    train_examples = random.sample(list(all_train_examples), limit_train)
                if limit_test and len(all_test_examples) > limit_test:
                    test_examples = random.sample(list(all_test_examples), limit_test)

                train_targets = [example_by_qid[qid]['target'] for qid in train_examples]
                test_targets = [example_by_qid[qid]['target'] for qid in test_examples]

                valid_split, culprit = verify_split(train_targets, test_targets, return_culprit=True)
                if valid_split:
                    easiness_score = compute_easiness_score(train_targets, test_targets, add_siblings)
                    print("Easiness score:", easiness_score)
                    if easiness_score > easiness_threshold:
                        print("Skipping")
                        break
                    split_name = f'split_{len(output_splits)}'
                    if not add_siblings:
                        split_name = f'no_sib_{split_name}'
                    if p_similar_bigrams != 1:
                        split_name = f'sim_{p_similar_bigrams}_{split_name}'

                    output_splits.append({
                        'train': list(train_examples),
                        'test': list(test_examples),
                        'split_name': split_name,
                        'split_info': {
                            'unseen_bigrams': ngrams_above_thresholds,
                            'n_unseen_templates': len(all_test_examples)
                        },
                        'easiness_score': easiness_score
                    })
                    found_valid_split = valid_split
                    print("- valid split found")
                    break
            if found_valid_split:
                break
        else:
            print("no valid split")

    if p_similar_bigrams < 1:
        # in this case bigrams are easier, so we'll take the hardest K
        output_splits.sort(key=lambda x: x['easiness_score'])
        output_splits = output_splits[:1]
    print("Total splits:", len(output_splits))
    return output_splits
