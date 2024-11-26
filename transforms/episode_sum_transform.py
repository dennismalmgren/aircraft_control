from __future__ import annotations

import functools
import importlib.util
import multiprocessing as mp
import warnings
from copy import copy
from enum import IntEnum
from functools import wraps
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

import torch

from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorData,
    set_lazy_legacy,
    TensorDict,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)
from tensordict.nn import dispatch, TensorDictModuleBase
from tensordict.utils import (
    _unravel_key_to_tuple,
    _zip_strict,
    expand_as_right,
    expand_right,
    NestedKey,
)
from torch import nn, Tensor
from torch.utils._pytree import tree_map

from torchrl._utils import _append_last, _ends_with, _make_ordinal_device, _replace_last

from torchrl.data.tensor_specs import (
    Binary,
    Bounded,
    Categorical,
    Composite,
    ContinuousBox,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    TensorSpec,
    Unbounded,
)
from torchrl.envs.common import _do_nothing, _EnvPostInit, EnvBase, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
from torchrl.envs.utils import _sort_keys, _update_during_reset, step_mdp
from torchrl.objectives.value.functional import reward2go

_has_tv = importlib.util.find_spec("torchvision", None) is not None

IMAGE_KEYS = ["pixels"]
_MAX_NOOPS_TRIALS = 10

FORWARD_NOT_IMPLEMENTED = "class {} cannot be executed without a parent environment."

T = TypeVar("T", bound="Transform")
from torchrl.envs.transforms import Transform

class EpisodeSum(Transform):
    """Tracks episode cumulative values.

    This transform accepts a list of tensordict reward keys (i.e. ´in_keys´) and tracks their cumulative
    value along the time dimension for each episode.

    When called, the transform writes a new tensordict entry for each ``in_key`` named
    ``episode_{in_key}`` where the cumulative values are written.

    Args:
        in_keys (list of NestedKeys, optional): Input observation keys.
            All ´in_keys´ should be part of the environment observation_spec.
        out_keys (list of NestedKeys, optional): The output sum keys, should be one per each input key.
        reset_keys (list of NestedKeys, optional): the list of reset_keys to be
            used, if the parent environment cannot be found. If provided, this
            value will prevail over the environment ``reset_keys``.

    Examples:
        >>> from torchrl.envs.transforms import RewardSum, TransformedEnv
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), RewardSum())
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> td = env.reset()
        >>> print(td["episode_reward"])
        tensor([0.])
        >>> td = env.rollout(3)
        >>> print(td["next", "episode_reward"])
        tensor([[1.],
                [2.],
                [3.]])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        reset_keys: Sequence[NestedKey] | None = None,

    ):
        """Initialises the transform. Filters out non-reward input keys and defines output keys."""
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys
        self._keys_checked = False
        self.reward_spec = False

    @property
    def in_keys(self):
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys in (None, []):
                raise Exception("Need to specify in keys")
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self):
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys in (None, []):
            out_keys = [
                _replace_last(in_key, f"episode_{_unravel_key_to_tuple(in_key)[-1]}")
                for in_key in self.in_keys
            ]
            self._out_keys = out_keys
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        # we must access the private attribute because this check occurs before
        # the parent env is defined
        if value is not None and len(self._in_keys) != len(value):
            raise ValueError(
                "Sum expects the same number of input and output keys"
            )
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def reset_keys(self):
        reset_keys = self.__dict__.get("_reset_keys", None)
        if reset_keys is None:
            parent = self.parent
            if parent is None:
                raise TypeError(
                    "reset_keys not provided but parent env not found. "
                    "Make sure that the reset_keys are provided during "
                    "construction if the transform does not have a container env."
                )
            # let's try to match the reset keys with the in_keys.
            # We take the filtered reset keys, which are the only keys that really
            # matter when calling reset, and check that they match the in_keys root.
            reset_keys = parent._filtered_reset_keys
            if len(reset_keys) == 1:
                reset_keys = list(reset_keys) * len(self.in_keys)

            def _check_match(reset_keys, in_keys):
                # if this is called, the length of reset_keys and in_keys must match
                for reset_key, in_key in _zip_strict(reset_keys, in_keys):
                    # having _reset at the root and the reward_key ("agent", "reward") is allowed
                    # but having ("agent", "_reset") and "reward" isn't
                    if isinstance(reset_key, tuple) and isinstance(in_key, str):
                        return False
                    if (
                        isinstance(reset_key, tuple)
                        and isinstance(in_key, tuple)
                        and in_key[: (len(reset_key) - 1)] != reset_key[:-1]
                    ):
                        return False
                return True

            if not _check_match(reset_keys, self.in_keys):
                raise ValueError(
                    f"Could not match the env reset_keys {reset_keys} with the {type(self)} in_keys {self.in_keys}. "
                    f"Please provide the reset_keys manually. Reset entries can be "
                    f"non-unique and must be right-expandable to the shape of "
                    f"the input entries."
                )
            reset_keys = copy(reset_keys)
            self._reset_keys = reset_keys

        if not self._keys_checked and len(reset_keys) != len(self.in_keys):
            raise ValueError(
                f"Could not match the env reset_keys {reset_keys} with the in_keys {self.in_keys}. "
                "Please make sure that these have the same length."
            )
        self._keys_checked = True

        return reset_keys

    @reset_keys.setter
    def reset_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._reset_keys = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets episode values."""
        for in_key, reset_key, out_key in _zip_strict(
            self.in_keys, self.reset_keys, self.out_keys
        ):
            _reset = _get_reset(reset_key, tensordict)
            value = tensordict.get(out_key, default=None)
            if value is None:
                value = self.parent.full_observation_spec[in_key].zero()
            else:
                value = torch.where(expand_as_right(~_reset, value), value, 0.0)
            tensordict_reset.set(out_key, value)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Updates the episode values with the step values."""
        # Update episode values
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if in_key in next_tensordict.keys(include_nested=True):
                value = next_tensordict.get(in_key)
                prev_value = tensordict.get(out_key, 0.0)
                next_tensordict.set(out_key, prev_value + value)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return next_tensordict

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        state_spec = input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = Composite(shape=input_spec.shape, device=input_spec.device)
        state_spec.update(self._generate_episode_value_spec())
        input_spec["full_state_spec"] = state_spec
        return input_spec

    def _generate_episode_value_spec(self) -> Composite:
        episode_value_spec = Composite()
        observation_spec = self.parent.full_observation_spec
        #observation_spec_keys = self.parent.reward_keys
        # Define episode specs for all out_keys
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Just assume the key is there.
            out_key = _unravel_key_to_tuple(out_key)
            temp_episode_value_spec = episode_value_spec
            temp_obs_spec = observation_spec
            for sub_key in out_key[:-1]:
                if (
                    not isinstance(temp_obs_spec, Composite)
                    or sub_key not in temp_obs_spec.keys()
                ):
                    break
                if sub_key not in temp_episode_value_spec.keys():
                    temp_episode_value_spec[sub_key] = temp_obs_spec[
                        sub_key
                    ].empty()
                temp_obs_spec = temp_obs_spec[sub_key]
                temp_episode_value_spec = temp_episode_value_spec[sub_key]
            episode_value_spec[out_key] = observation_spec[in_key].clone()
        return episode_value_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec, adding the new keys generated by RewardSum."""
        if self.reward_spec:
            return observation_spec
        if not isinstance(observation_spec, Composite):
            observation_spec = Composite(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(self._generate_episode_value_spec())
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return reward_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        time_dim = [i for i, name in enumerate(tensordict.names) if name == "time"]
        if not time_dim:
            raise ValueError(
                "At least one dimension of the tensordict must be named 'time' in offline mode"
            )
        time_dim = time_dim[0] - 1
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            reward = tensordict[in_key]
            cumsum = reward.cumsum(time_dim)
            tensordict.set(out_key, cumsum)
        return tensordict