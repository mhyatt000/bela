from dataclasses import asdict, dataclass, field
import logging

from lerobot.common import envs
from lerobot.common.optim.optimizers import AdamConfig, AdamWConfig
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.utils.utils import init_logging
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
import tyro

from bela.trainer import Trainer


@PreTrainedConfig.register_subclass("bela")
@dataclass
class BELAConfig(ACTConfig):
    """BELA policy configuration."""

    def observation_delta_indices(self):
        return self.action_delta_indices

    def image_delta_indices(self):
        return [0]


@dataclass
class MyTrainConfig(TrainPipelineConfig):
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="none"))
    human_repos: list[str] = field(default_factory=list)
    human_revisions: list[str] = field(default_factory=list)
    robot_repos: list[str] = field(default_factory=list)
    robot_revisions: list[str] = field(default_factory=list)
    env: envs.EnvConfig | None = None

    policy: BELAConfig = field(default_factory=BELAConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    optimizer: AdamWConfig | AdamConfig = field(default_factory=AdamWConfig)

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict:
        return asdict(self)


def main(cfg: MyTrainConfig) -> None:
    logging.info(cfg)
    Trainer(cfg).run()


if __name__ == "__main__":
    init_logging()
    main(tyro.cli(MyTrainConfig))
