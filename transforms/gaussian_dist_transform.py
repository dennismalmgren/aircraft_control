
import copy
from typing import Union, List

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
import torch
from scipy.spatial.transform import Rotation as R

class GaussianDistance(Transform):
    """A transform to convert altitudes to a scale code.

    Args:
        in_keys (sequence of NestedKey): the entries for the altitude.
        out_keys (sequence of NestedKey): the name of the scale code.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     AltitudeToScaleCode(["alt"], ["alt_code"]),
        ... )
    """

    def __init__(
        self, 
        in_keys, 
        out_keys,
        scales: List[List],
        constant = None
    ):
        in_keys_inv = []
        out_keys_inv = copy.copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.constant = constant
        if self.constant is not None:
            if len(self.in_keys) != len(self.out_keys) or len(self.in_keys) != 1:
                raise ValueError(
                    f"The number of in_keys ({len(self.in_keys)}) should be the the same as the number of out_keys ({len(self.out_keys)})."
                )
        else:
            if len(self.in_keys) != 2 * len(self.out_keys):
                raise ValueError(
                    f"The number of in_keys ({len(self.in_keys)}) should be the twice number of out_keys ({len(self.out_keys)})."
                )

        self.scales = scales
        if len(self.scales) != len(self.out_keys):
            raise ValueError(f"The number of scale specifications need to match the number of out_keys")


    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for scales, i in zip(self.scales, range(len(self.out_keys))):
            if self.constant is None:
                in_key1 = self.in_keys[2 * i]
                in_key2 = self.in_keys[2 * i + 1]
                out_key = self.out_keys[i]
                
                value1 = tensordict[in_key1]
                value2 = tensordict[in_key2]
                difference = value1 - value2
                tensor_out = self.encode_gaussian_distance(difference, scales)
            else:
                in_key = self.in_keys[i]
                out_key = self.out_keys[i]
                value1 = tensordict[in_key]
                constant_tensor = torch.tensor(self.constant, dtype=torch.float32, device=value1.device)
                difference = value1 - constant_tensor
                tensor_out = self.encode_gaussian_distance(difference, scales)
            tensordict[out_key] = tensor_out
        return tensordict

    forward = _call

    def encode_gaussian_distance(self, difference: torch.Tensor, scales: List):
        scales_tensor = torch.tensor(scales, dtype=torch.float32, device=difference.device)
        activations = torch.exp(-((difference / scales_tensor)**2))
        return activations

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        for scales, i in zip(self.scales, range(len(self.out_keys))):
            out_key = self.out_keys[i]
            in_key1 = self.in_keys[2 * i]
            output_spec["full_observation_spec"][out_key] = Unbounded(shape=(*output_spec.shape, len(scales)), device=output_spec.device, dtype=torch.float32) 
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
