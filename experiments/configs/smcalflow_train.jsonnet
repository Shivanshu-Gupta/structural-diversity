// local source_max_tokens = 25; # supports over 95% of the data
local source_max_tokens = 512; # supports over 95% of the data
local target_max_tokens = 200; # supports ~95% of the data

{
    "dataset_reader": {
        "read_loops": 2,
        "target_max_tokens": target_max_tokens,
        "source_max_tokens": source_max_tokens,
        "skip_target_too_long": true,
    },
    "model": {
        "max_decoding_steps": target_max_tokens,
    },
    "data_loader": {
        "batch_sampler": {
         "batch_size": 16
        }
    },
    "validation_data_loader": {
        "batch_sampler": {
         "batch_size": 16
        }
    },
    "trainer": {
        "num_epochs": 9,
        "patience": 8
    }
}