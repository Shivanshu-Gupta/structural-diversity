# python train.py subsampled-test --dataset covr --model-name bart --split-name template/split_0/4-400-max_wt --tags split,template

import os
import sys
import re
import time
import typer
import comet_ml
# from comet_ml import API  # needs to be loaded before allennlp

from dotenv import load_dotenv
from rich import print
from pathlib import Path
from allennlp.commands.train import train_model
from allennlp.common import Params
# from allennlp.common.params import with_fallback
from allennlp.common.params import with_overrides

from allennlp.common.util import import_module_and_submodules

root_dir = Path('experiments')
sys.path.append(str(root_dir / 'src'))

app = typer.Typer()

def get_full_model_name(model_name):
    if model_name == "bart":
        return "facebook/bart-base"
    elif model_name == "t5":
        return "t5-base"
    elif model_name == "t5-large":
        return "t5-large"
    elif model_name == "bart-large":
        return "facebook/bart-large"
    raise ValueError(model_name)


def get_experiment_name(name, dataset, sub_dataset, model_name):
    name = f"{name} - " if name else ""
    name += f"{dataset}"
    if dataset == "overnight":
        name += f" ({sub_dataset})"
    name += f" - {model_name}"
    return name


def merge_settings(base, to_merge, override):
    loaded_settings = Params.from_file(to_merge, override)
    return with_overrides(base, loaded_settings.as_flat_dict())

def experiment_already_exists(experiment_name, tags):
    comet_api = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
    try:
        experiments_with_same_name = comet_api.get_experiments(
            workspace=os.getenv("COMET_WORKSPACE"),
            project_name=os.getenv("COMET_PROJECT_NAME"),
            pattern=re.escape(experiment_name))
        for exp in experiments_with_same_name:
            if set(exp.get_tags()) == set(tags):
                return True
    except comet_ml.exceptions.NotFound:
        print('Project not found')
    finally:
        return False

@app.command()
def main_train(name: str ='',  tags: str =None, model_name: str ='bart',
               dataset: str = 'covr', instance_source: str = 'source',
               sub_dataset: str = None, data_path: str = '', split_path: str = '', split_name: str = '',
               force: bool = False, recover: bool = False, overrides: str = '',
               gpu: int = 0, mini_test: bool = False, experiment_name: str = None,
               serialization_dir: str = '', train_data_path: str = '', validation_data_path: str = '',
               n_epochs: int = None, ben: bool = False,):
    # args = parse_args(argv)
    import_module_and_submodules('experiments.src')

    settings = Params.from_file(str(root_dir / "configs/base.jsonnet"), overrides).params

    dataset_specific_config_path = str(root_dir / f"configs/{dataset}_train.jsonnet")
    if os.path.exists(dataset_specific_config_path):
        print("Loading config file " + dataset_specific_config_path)
        settings = merge_settings(settings, dataset_specific_config_path, overrides)
    else:
        print(f"No specific config file found for {dataset}")

    if dataset == "overnight":
        settings['dataset_reader']['sub_dataset'] = sub_dataset
    # elif not dataset.startswith('covr'):
    #     settings = merge_settings(settings, "configs/other_datasets_train.jsonnet", overrides)
    #     settings['train_data_path'] = settings['train_data_path'].replace('{dataset}', dataset)
    #     settings['validation_data_path'] = settings['validation_data_path'].replace('{dataset}', dataset)

    if data_path:
        settings['train_data_path'] = data_path
        settings['validation_data_path'] = '@' + data_path
    if train_data_path:
        settings['train_data_path'] = train_data_path
    if validation_data_path:
        settings['validation_data_path'] = '@' + validation_data_path
    if n_epochs:
        settings['trainer']['num_epochs'] = n_epochs

    settings['trainer']['cuda_device'] = gpu
    settings['dataset_reader']['instance_source'] = instance_source

    experiment_name = experiment_name or get_experiment_name(name, dataset, sub_dataset, model_name)

    if split_name:
        settings['dataset_reader']['split_path'] = str(Path('../datasets') / ('ben' if ben else '') / dataset / (sub_dataset or "") / f'splits/{split_name}.json')
        experiment_name += f' - split({split_name})'
    elif split_path:
        settings['dataset_reader']['split_path'] = split_path
        split_path = Path()
        split_name = split_path.name.split('.')[0]
        experiment_name += f' - split({split_name})'
    print(settings)
    print(experiment_name)
    print(serialization_dir)

    if experiment_already_exists(experiment_name, tags):
        print(f"***** skipping {experiment_name} since already found in cometml *****")
        return

    model_name = get_full_model_name(model_name)
    settings['dataset_reader']['source_tokenizer']['model_name'] = model_name
    settings['dataset_reader']['source_token_indexers']['tokens']['model_name'] = model_name
    settings['model']['model_name'] = model_name

    if model_name == "t5-large":
        if dataset == 'smcalflow':
            settings['data_loader']['batch_sampler']['batch_size'] = 7
            settings['validation_data_loader']['batch_sampler']['batch_size'] = 7
        elif dataset == 'spider':
            settings['data_loader']['batch_sampler']['batch_size'] = 4
            settings['validation_data_loader']['batch_sampler']['batch_size'] = 4
        else:
            settings['data_loader']['batch_sampler']['batch_size'] = 8
            settings['validation_data_loader']['batch_sampler']['batch_size'] = 8

    tags = tags.split(',') if tags else []
    tags.append(model_name)
    settings['trainer']['callbacks'][0]['tags'] = ','.join(tags)
    settings['trainer']['callbacks'][0]['name'] = experiment_name

    parent_experiments_path = root_dir / "runs"
    serialization_dir_path = parent_experiments_path / serialization_dir.replace(" ", "_")
    print(serialization_dir_path)

    # check if in pycharm debug mode
    debug_mode = sys.gettrace() is not None
    if debug_mode:
        serialization_dir = 'debug'
        force = True

    if not debug_mode:
        if os.path.exists(serialization_dir_path) and not force:
            serialization_dir += '_' + str(time.time()).replace('.', '')
            serialization_dir_path = parent_experiments_path / serialization_dir.replace(" ", "_")

    if mini_test:
        settings['dataset_reader']['max_instances'] = 10

    print(settings)

    train_model(
        params=Params(settings),
        serialization_dir=serialization_dir_path,
        recover=recover,
        force=force
    )


if __name__ == "__main__":
    app()
