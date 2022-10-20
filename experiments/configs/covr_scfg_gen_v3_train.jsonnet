{
    // "train_data_path": "../datasets/covr/train_9.jsonl",
    // "validation_data_path": "@../datasets/covr/train_9.jsonl",  // '@' is a hack to indicate this is validation
    // "train_data_path": "../datasets/covr/splits/template/split_0/grammar_gen.jsonl",
    // "validation_data_path": "@../datasets/covr/splits/template/split_0/grammar_gen.jsonl",  // '@' is a hack to indicate this is validation
    "train_data_path": "../datasets/covr_scfg_gen_v3/covr_scfg_gen_v3.all.jsonl",
    "validation_data_path": "@../datasets/covr_scfg_gen_v3/covr_scfg_gen_v3.all.jsonl",  // '@' is a hack to indicate this is validation
    "dataset_reader": {
        "type": "general_dataset_reader",
        // "n_train_sample": 2500
    }
}