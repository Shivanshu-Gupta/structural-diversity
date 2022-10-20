import json
import logging

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from overrides import overrides

from .base_dataset_reader import BaseDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("general_dataset_reader")
class GeneralDatasetReader(BaseDatasetReader):

    @overrides
    def get_examples(self, file_path, is_train: bool):
        all_examples = [json.loads(s) for s in open(file_path, 'rt')]

        if self._split_path:
            split_info = json.load(open(self._split_path, 'rt'))
            train_ids = set(split_info['train'])
            test_ids = set(split_info['test'])

            if is_train:
                all_examples = [ex for ex in all_examples if ex['qid'] in train_ids]
                print('#train examples:', len(all_examples))
            else:
                all_examples = [ex for ex in all_examples if ex['qid'] in test_ids]
                print('#test examples:', len(all_examples))

        assert len(all_examples) > 0
        return all_examples
