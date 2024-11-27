
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
    - error: Signed heading error in radians (tensor)
    """
    cos_current = torch.cos(current_angle)
    sin_current = torch.sin(current_angle)
    cos_target = torch.cos(target_angle)
    sin_target = torch.sin(target_angle)
    cos_error = cos_current * cos_target + sin_current * sin_target
    sin_error = cos_current * sin_target - sin_current * cos_target
    error = torch.cat([cos_error, sin_error], dim=-1)

    return error

class PlanarAngleCosSin(Transform):
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
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should be the same as the number of out_keys ({len(self.in_keys)})."
            )


    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            value = tensordict[in_key]
            out_key_cos = torch.cos(value)
            out_key_sin = torch.sin(value)
            result = torch.cat([out_key_cos, out_key_sin], dim=-1)
            
            tensordict[out_key] = result
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
