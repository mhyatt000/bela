from dataclasses import dataclass
from enum import Enum

from lerobot.configs.types import FeatureType
from lerobot.configs.types import PolicyFeature as _PolicyFeature
import numpy as np


class Morph(str, Enum):
    ROBOT = "robot"
    HUMAN = "human"
    HR = "hr"


@dataclass
class Head:
    """Specifies the morph and flattened shape of a data head."""

    morph: Morph | None
    shape: tuple

    def __post_init__(self):
        self.shape = np.zeros(self.shape).reshape(-1).shape  # flatten shape

    def __add__(self, other):
        if isinstance(other, Head):
            arr = np.concatenate(
                [np.zeros(self.shape).reshape(-1), np.zeros(other.shape).reshape(-1)],
                axis=0,
            ).reshape(-1)
            return Head(None, arr.shape)
        raise TypeError(f"Unsupported type for addition: {type(other)}")


@dataclass
class PolicyFeature(_PolicyFeature):
    """Description of a feature used by a policy."""

    type: FeatureType
    shape: tuple

    def flatten(self):
        arr = np.zeros(self.shape).reshape(-1)
        return PolicyFeature(self.type, arr.shape)


@dataclass
class HeadSpec:
    """Shapes for robot, human and shared features."""

    robot: Head | None = None
    human: Head | None = None
    share: Head | None = None

    # encode to shared input space
    # decode to h, hr, r output space

    @property
    def h(self):
        self.validate(["human"])
        return self.human.shape

    @property
    def r(self):
        self.validate(["robot"])
        return self.robot.shape

    @property
    def s(self):
        self.validate(["share"])
        return self.share.shape

    @property
    def hs(self):
        self.validate(["human", "share"])
        return (self.human + self.share).shape

    @property
    def rs(self):
        self.validate(["robot", "share"])
        return (self.robot + self.share).shape

    def validate(self, attrs):
        for attr in attrs:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} must be defined")
