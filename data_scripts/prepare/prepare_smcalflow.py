import os
import sys
import itertools
import json
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


BASE_DIR = Path({
    'ben': '../..',
    'shivanshu': ''
}[os.getenv('RUNNER')])
sys.path.insert(0, BASE_DIR.absolute().__str__())

from experiments.analysis.ast_parser import ASTParser
from experiments.analysis.utils.lf_utils import tokenize_lf
from generation.scfg.utils import target_to_ast_calflow
# sys.path.append('../scripts')
from data_scripts.utils.anon_fns import anonymize_smcalflow_target

ast_parser = ASTParser(config={})

f1 = open(r"datasets/smcalflow/train.dataflow_dialogues.jsonl", encoding='utf8')
f2 = open(r"datasets/smcalflow/valid.dataflow_dialogues.jsonl", encoding='utf8')

f_out = open("datasets/smcalflow/smcalflow.all.jsonl", "wt")
cnt = 0
for i, line in enumerate(tqdm(itertools.chain(f1, f2))):
    ex = json.loads(line)
    for j, turn in enumerate(ex['turns']):
        source = turn['user_utterance']['original_text']
        target = turn['lispress']
        ast_target = target
        # ast_target = anonymize_smcalflow_target(target, anonymize_level=1)

        # make sure everything is fine by parsing
        try:
            # ast_parser.get_ast(tokenize_lf(ast_target))
            target_to_ast_calflow(ast_target)
        except Exception as e:
            print(e)
            print(ast_target)
            continue

        # if j > 0:
        #     continue

        if target.count("(") <= 2:
            continue

        f_out.write(json.dumps({
            'qid': f'smcalflow_{i}_{j}',
            'source': source.strip(),
            'target': target.strip(),
        }) + "\n")
        cnt += 1

print(f"Saved {cnt} examples")
