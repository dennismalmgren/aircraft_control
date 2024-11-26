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
from copy import copy

import torch
from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.transforms import _apply_to_composite

from tensordict.utils import (
    _zip_strict,
    expand_as_right,
    NestedKey,
)

from tensordict import (
    TensorDictBase,
)

from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
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

class TimeMinPool(Transform):
    """Take the minimum value in each position over the last T observations.

    This transform take the minimum value in each position for all in_keys tensors over the last T time steps.

    Args:
        in_keys (sequence of NestedKey, optional): input keys on which the max pool will be applied. Defaults to "observation" if left empty.
        out_keys (sequence of NestedKey, optional): output keys where the output will be written. Defaults to `in_keys` if left empty.
        T (int, optional): Number of time steps over which to apply max pooling.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, TimeMinPool(in_keys=["observation"], T=10))
        >>> torch.manual_seed(0)
        >>> env.set_seed(0)
        >>> rollout = env.rollout(10)
        >>> print(rollout["observation"])  # values should be increasing up until the 10th step
        tensor([[ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0216,  0.0000],
                [ 0.0000,  0.1149,  0.0000],
                [ 0.0000,  0.1990,  0.0000],
                [ 0.0000,  0.2749,  0.0000],
                [ 0.0000,  0.3281,  0.0000],
                [-0.9290,  0.3702, -0.8978]])

    .. note:: :class:`~TimeMinPool` currently only supports ``done`` signal at the root.
        Nested ``done``, such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        T: Optional[int] = 1,
        reset_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if T and T < 1:
            raise ValueError(
                "TimeMinPoolTransform T parameter should have a value greater or equal to one."
            )
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                "TimeMinPoolTransform in_keys and out_keys don't have the same number of elements"
            )
        self.buffer_size = T
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            setattr(
                self,
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        self.reset_key = reset_key

    @staticmethod
    def _buffer_name(in_key):
        in_key_str = "_".join(in_key) if isinstance(in_key, tuple) else in_key
        buffer_name = f"_minpool_buffer_{in_key_str}"
        return buffer_name

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key", None)
        if reset_key is None:
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:

        _reset = _get_reset(self.reset_key, tensordict)
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                continue
            if not _reset.all():
                _reset_exp = _reset.expand(buffer.shape[0], *_reset.shape)
                buffer[_reset_exp] = 0.0
            else:
                buffer.fill_(0.0)
        with _set_missing_tolerance(self, True):
            for in_key in self.in_keys:
                val_reset = tensordict_reset.get(in_key, None)
                val_prev = tensordict.get(in_key, None)
                # if an in_key is missing, we try to copy it from the previous step
                if val_reset is None and val_prev is not None:
                    tensordict_reset.set(in_key, val_prev)
                elif val_prev is None and val_reset is None:
                    raise KeyError(f"Could not find {in_key} in the reset data.")
            return self._call(tensordict_reset, _reset=_reset)

    def _make_missing_buffer(self, tensordict, in_key, buffer_name):
        buffer = getattr(self, buffer_name)
        data = tensordict.get(in_key)
        size = list(data.shape)
        size.insert(0, self.buffer_size)
        buffer.materialize(size)
        buffer = buffer.to(dtype=data.dtype, device=data.device).zero_()
        setattr(self, buffer_name, buffer)
        return buffer

    def _call(self, tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with min pooled keys."""
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(tensordict, in_key, buffer_name)
            if _reset is not None:
                # we must use only the reset data
                buffer[:, _reset] = torch.roll(buffer[:, _reset], shifts=1, dims=0)
                # add new obs
                data = tensordict.get(in_key)
                buffer[0, _reset] = data[_reset]
                # apply max pooling
                pooled_tensor, _ = buffer[:, _reset].min(dim=0)
                pooled_tensor = torch.zeros_like(data).masked_scatter_(
                    expand_as_right(_reset, data), pooled_tensor
                )
                # add to tensordict
                tensordict.set(out_key, pooled_tensor)
                continue
            # shift obs 1 position to the right
            buffer.copy_(torch.roll(buffer, shifts=1, dims=0))
            # add new obs
            buffer[0].copy_(tensordict.get(in_key))
            # apply max pooling
            pooled_tensor, _ = buffer.min(dim=0)
            # add to tensordict
            tensordict.set(out_key, pooled_tensor)

        return tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "TimeMinPool cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "TimeMinPool with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need TimeMaxPool to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )

class TimeMaxPool(Transform):
    """Take the maximum value in each position over the last T observations.

    This transform take the maximum value in each position for all in_keys tensors over the last T time steps.

    Args:
        in_keys (sequence of NestedKey, optional): input keys on which the max pool will be applied. Defaults to "observation" if left empty.
        out_keys (sequence of NestedKey, optional): output keys where the output will be written. Defaults to `in_keys` if left empty.
        T (int, optional): Number of time steps over which to apply max pooling.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, TimeMinPool(in_keys=["observation"], T=10))
        >>> torch.manual_seed(0)
        >>> env.set_seed(0)
        >>> rollout = env.rollout(10)
        >>> print(rollout["observation"])  # values should be increasing up until the 10th step
        tensor([[ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0216,  0.0000],
                [ 0.0000,  0.1149,  0.0000],
                [ 0.0000,  0.1990,  0.0000],
                [ 0.0000,  0.2749,  0.0000],
                [ 0.0000,  0.3281,  0.0000],
                [-0.9290,  0.3702, -0.8978]])

    .. note:: :class:`~TimeMaxPool` currently only supports ``done`` signal at the root.
        Nested ``done``, such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        T: Optional[int] = 1,
        reset_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if T and T < 1:
            raise ValueError(
                "TimeMaxPoolTransform T parameter should have a value greater or equal to one."
            )
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                "TimeMaxPoolTransform in_keys and out_keys don't have the same number of elements"
            )
        self.buffer_size = T
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            setattr(
                self,
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        self.reset_key = reset_key

    @staticmethod
    def _buffer_name(in_key):
        in_key_str = "_".join(in_key) if isinstance(in_key, tuple) else in_key
        buffer_name = f"_maxpool_buffer_{in_key_str}"
        return buffer_name

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key", None)
        if reset_key is None:
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:

        _reset = _get_reset(self.reset_key, tensordict)
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                continue
            if not _reset.all():
                _reset_exp = _reset.expand(buffer.shape[0], *_reset.shape)
                buffer[_reset_exp] = 0.0
            else:
                buffer.fill_(0.0)
        with _set_missing_tolerance(self, True):
            for in_key in self.in_keys:
                val_reset = tensordict_reset.get(in_key, None)
                val_prev = tensordict.get(in_key, None)
                # if an in_key is missing, we try to copy it from the previous step
                if val_reset is None and val_prev is not None:
                    tensordict_reset.set(in_key, val_prev)
                elif val_prev is None and val_reset is None:
                    raise KeyError(f"Could not find {in_key} in the reset data.")
            return self._call(tensordict_reset, _reset=_reset)

    def _make_missing_buffer(self, tensordict, in_key, buffer_name):
        buffer = getattr(self, buffer_name)
        data = tensordict.get(in_key)
        size = list(data.shape)
        size.insert(0, self.buffer_size)
        buffer.materialize(size)
        buffer = buffer.to(dtype=data.dtype, device=data.device).zero_()
        setattr(self, buffer_name, buffer)
        return buffer

    def _call(self, tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with min pooled keys."""
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(tensordict, in_key, buffer_name)
            if _reset is not None:
                # we must use only the reset data
                buffer[:, _reset] = torch.roll(buffer[:, _reset], shifts=1, dims=0)
                # add new obs
                data = tensordict.get(in_key)
                buffer[0, _reset] = data[_reset]
                # apply max pooling
                pooled_tensor, _ = buffer[:, _reset].max(dim=0)
                pooled_tensor = torch.zeros_like(data).masked_scatter_(
                    expand_as_right(_reset, data), pooled_tensor
                )
                # add to tensordict
                tensordict.set(out_key, pooled_tensor)
                continue
            # shift obs 1 position to the right
            buffer.copy_(torch.roll(buffer, shifts=1, dims=0))
            # add new obs
            buffer[0].copy_(tensordict.get(in_key))
            # apply max pooling
            pooled_tensor, _ = buffer.max(dim=0)
            # add to tensordict
            tensordict.set(out_key, pooled_tensor)

        return tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "TimeMinPool cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "TimeMinPool with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need TimeMaxPool to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )
