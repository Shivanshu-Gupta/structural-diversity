local target_max_tokens = 230;

{
    "trainer": {
        "num_epochs": 18
    },
    "dataset_reader": {
        "target_max_tokens": target_max_tokens,
    },
    "model": {
        "max_decoding_steps": target_max_tokens,
    },
}