{
    "train_data_path": "../datasets/overnight/{sub_dataset}.train.jsonl",
    "validation_data_path": "@../datasets/overnight/{sub_dataset}.test.jsonl",  // '@' is a hack to indicate this is validation
    "dataset_reader": {
        "type": "overnight"
    }
}