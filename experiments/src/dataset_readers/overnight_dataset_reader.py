import logging

import json
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from .base_dataset_reader import BaseDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("overnight")
class OvernightDatasetReader(BaseDatasetReader):
    def __init__(
            self,
            sub_dataset: str,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sub_dataset = sub_dataset

    def get_examples(self, file_path, is_train: bool):
        if self._split_path:
            split_info = json.load(open(self._split_path, 'rt'))
            if 'test_examples' in split_info:
                test_ids = set(split_info['test_examples'])
            else:
                test_ids = set(split_info['test'])

            if 'train_examples' in split_info:
                train_ids = set(split_info['train_examples'])
            else:
                train_ids = set(split_info['train'])
        else:
            train_ids = set()
            test_ids = set()

        file_path = file_path.replace("{sub_dataset}", self._sub_dataset)
        all_examples = [json.loads(s) for s in open(file_path, 'rt')]

        if self._split_path:
            if is_train:
                all_examples = [ex for ex in all_examples if ex['qid'] in train_ids]
            else:
                all_examples = [ex for ex in all_examples if ex['qid'] in test_ids]

        return all_examples
