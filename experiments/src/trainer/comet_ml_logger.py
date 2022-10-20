from typing import Dict, Any

import os

import comet_ml

from allennlp.common import Params
from allennlp.training.callbacks.callback import TrainerCallback

from dotenv import load_dotenv

load_dotenv()


@TrainerCallback.register("comet_ml_logger")
class CometLoggerCallback(TrainerCallback):
    def __init__(
            self,
            serialization_dir: str,
            enabled: bool,
            name: str = 'run',
            tags: str = None
    ):
        super().__init__(serialization_dir)
        self._enabled = enabled

        if enabled and os.getenv("COMET_API_KEY"):
            self.experiment = comet_ml.Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=os.getenv("COMET_PROJECT_NAME"),
                workspace=os.getenv("COMET_WORKSPACE")
            )
        else:
            self.experiment = comet_ml.OfflineExperiment(
                offline_directory='comet_ml_experiments',
                disabled=True
            )
        self.experiment.log_system_info('serialization_dir', serialization_dir)
        self.experiment.log_other('serialization_dir', serialization_dir)
        self.experiment.set_name(name)
        if tags:
            for tag in tags.split(','):
                self.experiment.add_tag(tag)

    def on_start(
            self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        # We update comet.ml experiment with the config parameters by loading the config file
        config_file_path = os.path.join(trainer._serialization_dir, 'config.json')
        params = Params.from_file(config_file_path)
        self.experiment.log_parameters(params.as_flat_dict())

    def on_epoch(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_primary: bool = True,
            **kwargs,
    ) -> None:
        self.experiment.log_metrics(metrics, step=epoch+1)
