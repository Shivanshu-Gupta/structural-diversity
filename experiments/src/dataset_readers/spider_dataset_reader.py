import json
import logging

import collections
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from overrides import overrides

from .general_dataset_reader import GeneralDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("spider_dataset_reader")
class SpiderDatasetReader(GeneralDatasetReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        _tables_json = json.load(open("../datasets/spider/tables.json"))
        self.tables_json = {schema['db_id']: schema for schema in _tables_json}

    def _get_schema_string(self, table_json):
        """
        Returns the schema serialized as a string.
        Code taken from https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py
        """
        table_id_to_column_names = collections.defaultdict(list)
        for table_id, name in table_json["column_names_original"]:
            table_id_to_column_names[table_id].append(name.lower())
        tables = table_json["table_names_original"]

        table_strings = []
        for table_id, table_name in enumerate(tables):
            column_names = table_id_to_column_names[table_id]
            table_string = " | %s : %s" % (table_name.lower(), " , ".join(column_names))
            table_strings.append(table_string)

        return "".join(table_strings)

    @overrides
    def get_schema_string(self, ex) -> str:
        return self._get_schema_string(self.tables_json[ex['db_id']])
