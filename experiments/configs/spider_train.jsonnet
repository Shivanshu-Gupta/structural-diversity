local target_max_tokens = 230;

{
    "dataset_reader": {
        "type": "spider_dataset_reader",
        "read_loops": 3,
        "truncate_source_too_long": true,
        "skip_target_too_long": true,
        "target_max_tokens": target_max_tokens,
    },
    "model": {
        "max_decoding_steps": target_max_tokens,
    },
    "trainer": {
     "num_epochs": 20,
    },
    "data_loader": {
        "batch_sampler": {
         "batch_size": 8
        }
    },
    "validation_data_loader": {
        "batch_sampler": {
         "batch_size": 8
        }
    },
}