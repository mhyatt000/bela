from dataclasses import dataclass, field

from flax.traverse_util import flatten_dict, unflatten_dict
import jax
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.types import FeatureType, PolicyFeature
import numpy as np
from rich.pretty import pprint
import torch
import tyro

from bela.common.datasets.util import load_dataspec
from bela.typ import Head, HeadSpec, Morph


def typespec(d):
    return jax.tree.map(lambda x: type(x), d)


def spec(d):
    return jax.tree.map(lambda x: x.shape, d)


@dataclass
class HybridConfig:
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(repo_id="none"))
    human_repos: list[str] = field(default_factory=list)
    robot_repos: list[str] = field(default_factory=list)

    policy: ACTConfig = field(default_factory=ACTConfig)


from bela.common.policies.bela import BELAPolicy


def make_policy():
    batchspec = load_dataspec()

    def flatten_state_feat(x):
        if x.type == FeatureType.STATE:
            arr = np.zeros(x.shape).reshape(-1)
            return PolicyFeature(FeatureType.STATE, arr.shape)
        return x

    batchspec = jax.tree.map(flatten_state_feat, batchspec)

    # pprint(batchspec)
    input_features = flatten_dict(batchspec, sep=".")
    output_features = dict(input_features)
    output_features = jax.tree.map(
        lambda x: (PolicyFeature(FeatureType.ACTION, x.shape) if x.type == FeatureType.STATE else None),
        output_features,
    )
    output_features = {k: v for k, v in output_features.items() if v is not None}
    output_features = {k.replace("observation.", "action."): v for k, v in output_features.items()}

    batchspec = batchspec | unflatten_dict(output_features, sep=".")

    state_features = {k: v for k, v in input_features.items() if v.type == FeatureType.STATE}
    pprint(batchspec)

    def compute_head(feat, head):
        headfeat = {k: v for k, v in feat if head in k}
        headfeat = {k: Head(None, v.shape) for k, v in headfeat.items()}
        return sum(list(headfeat.values()), Head(None, (0,)))

    headspec = HeadSpec(
        robot=Head(Morph.ROBOT, compute_head(state_features.items(), "robot").shape),
        human=Head(Morph.HUMAN, compute_head(state_features.items(), "human").shape),
        share=Head(Morph.HR, compute_head(state_features.items(), "shared").shape),
    )
    pprint(headspec)

    bs = 4
    chunk = 50

    def generate_example(x):
        shape = list(x.shape)
        if x.type == FeatureType.STATE:
            return np.zeros(([bs] + shape))
        if x.type == FeatureType.VISUAL:
            return np.zeros(([bs] + shape))
        if x.type == FeatureType.ACTION:
            return np.zeros(([bs, chunk] + shape))

    def generate_stats(x):
        return {
            "count": np.array(55),
            "max": np.random.random(x.shape),
            "min": np.random.random(x.shape),
            "mean": np.random.random(x.shape),
            "std": np.random.random(x.shape),
        }

    example_batch = jax.tree.map(generate_example, batchspec)
    example_batch = jax.tree.map(torch.Tensor, example_batch)
    example_batch = flatten_dict(example_batch, sep=".")

    example_batch["action_is_pad"] = torch.zeros((bs, chunk)).bool()

    example_stats = jax.tree.map(generate_stats, batchspec)
    example_stats = jax.tree.map(torch.Tensor, example_stats)
    is_leaf = lambda d, k: "count" in k
    example_stats = flatten_dict(example_stats, sep=".", is_leaf=is_leaf)
    pprint(spec(example_stats))

    policycfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk,
        n_action_steps=chunk,
    )
    pprint(policycfg)
    policy = BELAPolicy(
        config=policycfg,
        headspec=headspec,
        dataset_stats=example_stats,
    )

    policy(example_batch, heads=["human", "shared"])
    policy(example_batch, heads=["robot", "shared"])
    return policy


def main(cfg: HybridConfig):
    hdatasets, rdatasets = [], []
    for h in cfg.human_repos:
        cfg.dataset.repo_id = h
        hdatasets.append(make_dataset(cfg))
    for r in cfg.robot_repos:
        cfg.dataset.repo_id = r
        rdatasets.append(make_dataset(cfg))

    print(hdatasets)
    print(rdatasets)


if __name__ == "__main__":
    main(tyro.cli(HybridConfig))
