import os, sys
from collections import defaultdict
from random import Random

sys.path.append(os.getcwd())
from experiments.analysis.utils.splits_experiments_utils import verify_split

LIMIT_PER_PROGRAM = 1000


def splits_by_template(example_by_qid, anon_fn, n_random_seeds, percentage_programs_test, anon_using_prods=False, limit_train=None, limit_test=None):
    output_splits = []
    for random_seed in range(n_random_seeds):
        random = Random(random_seed)

        examples_per_program = defaultdict(list)

        for ex in example_by_qid.values():
            ex['anonymized_target'] = anon_fn(ex['productions']) if anon_using_prods else anon_fn(ex['target'])
            examples_per_program[ex['anonymized_target']].append(ex['qid'])

        print(f"Unique #templates: {len(examples_per_program)}")

        for program, examples in examples_per_program.items():
            # limit number of examples per program if too much
            if len(examples) > LIMIT_PER_PROGRAM:
                examples_per_program[program] = random.sample(examples, LIMIT_PER_PROGRAM)

        all_programs = list(examples_per_program.keys())

        n_tries = 0
        while n_tries < 50000:
            random.shuffle(all_programs)
            n_test_programs = int(percentage_programs_test * len(all_programs))
            train_programs = all_programs[n_test_programs:]
            test_programs = all_programs[:n_test_programs]

            train_examples = [ex for p in train_programs for ex in examples_per_program[p]]
            # no need for more than a small number of test examples per program
            n_test_examples_per_program = max(10, int(1000/len(test_programs)))
            test_examples = [ex for p in test_programs
                             for ex in random.sample(examples_per_program[p],
                                                     min(n_test_examples_per_program,
                                                         len(examples_per_program[p])))]

            if limit_train and len(train_examples) > limit_train:
                train_examples = random.sample(train_examples, limit_train)
            if limit_test and len(test_examples) > limit_test:
                test_examples = random.sample(test_examples, limit_test)

            n_tries += 1
            if verify_split(train_programs, test_programs):
                break
        else:
            raise Exception("Could not make split")

        print(f"Train programs before sub-sampling: {len(train_programs)}")
        print(f"Test programs before sub-sampling: {len(test_programs)}")

        output_splits.append({
            'train': train_examples,
            'test': test_examples,
            'split_name': f'{percentage_programs_test}/split_{random_seed}'
        })
    return output_splits
