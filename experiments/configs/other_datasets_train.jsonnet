{
    "train_data_path": "../datasets/{dataset}/{dataset}.all.jsonl",
    "validation_data_path": "@../datasets/{dataset}/{dataset}.all.jsonl",  // '@' is a hack to indicate this is validation
    "dataset_reader": {
        "n_train_sample": null,
        "n_test_sample": 2000,
    }
}