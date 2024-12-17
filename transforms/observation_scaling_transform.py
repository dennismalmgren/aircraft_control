
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

from torchrl._utils import (
    _append_last,
    _ends_with,
    _make_ordinal_device,
    _replace_last,
    implement_for,
)

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
from torchrl.envs.transforms import Transform
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


def _apply_to_composite(function):
    @wraps(function)
    def new_fun(self, observation_spec):
        if isinstance(observation_spec, Composite):
            _specs = observation_spec._specs
            in_keys = self.in_keys
            out_keys = self.out_keys
            for in_key, out_key in _zip_strict(in_keys, out_keys):
                if in_key in observation_spec.keys(True, True):
                    _specs[out_key] = function(self, observation_spec[in_key].clone())
            return Composite(
                _specs, shape=observation_spec.shape, device=observation_spec.device
            )
        else:
            return function(self, observation_spec)

    return new_fun


class ObservationScaling(Transform):
    """Affine transform of the observation.

     The observation is transformed according to:

    .. math::
        observation = observation * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
        standard_normal (bool, optional): if ``True``, the transform will be

            .. math::
                observation = (observation-loc)/scale

            as it is done for standardization. Default is `False`.
    """

    def __init__(
        self,
        loc: Union[float, torch.Tensor],
        scale: Union[float, torch.Tensor],
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            raise Exception("Must define in_keys")
        if out_keys is None:
            out_keys = copy(in_keys)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not isinstance(standard_normal, torch.Tensor):
            standard_normal = torch.tensor(standard_normal)
        self.register_buffer("standard_normal", standard_normal)

        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale.clamp_min(1e-6))

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            observation = tensordict[in_key]
            if self.standard_normal:
                loc = self.loc
                scale = self.scale
                observation = (observation - loc) / scale
            else:
                scale = self.scale
                loc = self.loc
                observation = observation * scale + loc
            tensordict[out_key] = observation

        return tensordict
    
    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec such that the resulting spec matches transform mapping.

        Args:
            observation_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            observation_spec[out_key] = observation_spec[in_key].clone()
        return observation_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
