
import copy

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
import torch
from scipy.spatial.transform import Rotation as R

def angular_difference(target_angle: torch.Tensor, current_angle: torch.Tensor):
    """
    Calculate the shortest distance between two angles (in radians).
    Handles batch computations with PyTorch tensors.

    Parameters:
    - theta1: Current heading (tensor)
    - theta2: Desired heading (tensor)

    Returns:
    - error: cosine and sine of error
    """
    cos_error = torch.cos(target_angle - current_angle)
    sin_error = torch.sin(target_angle - current_angle)
    error = torch.cat([cos_error, sin_error], dim=-1)

    return error

class AngularDifference(Transform):
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
        self, in_keys, out_keys,
    ):
        in_keys_inv = []
        out_keys_inv = copy.copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if len(self.in_keys) != 2 * len(self.out_keys):
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should be the twice number of out_keys ({len(self.in_keys)})."
            )


    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for i in range(len(self.out_keys)):
            in_key1 = self.in_keys[2 * i]
            in_key2 = self.in_keys[2 * i + 1]
            out_key = self.out_keys[i]
            
            value1 = tensordict[in_key1]
            value2 = tensordict[in_key2]
            difference = angular_difference(value1, value2)
            
            tensordict[out_key] = difference
        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        output_dim = 2
        for out_key in self.out_keys:
            output_spec["full_observation_spec"][out_key] = Unbounded(shape=(*output_spec.shape, output_dim), device=output_spec.device, dtype=torch.float32) 
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
