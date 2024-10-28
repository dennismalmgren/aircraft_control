
import copy

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
import torch
from scipy.spatial.transform import Rotation as R

def _digits_encoding(position, device, N=5):
    """
    Generate a 1D vector of digits scaled to <1.0
    """
    encoding = torch.zeros((*position.shape[:-1], N), device=device, dtype=torch.float32)
    for i in range(N):
        factor = torch.pow(10, torch.tensor(i, device=device))
        tail = (position % (factor * 10) / factor).squeeze(-1)

        if i == 0:
            encoding[..., N - 1 - i] = tail
        else:
            encoding[..., N - 1 - i] = torch.floor(tail)
    encoding /= 10.0
        # % 10
    return encoding


class AltitudeToDigits(Transform):
    """A transform to convert altitude to a digits vector

    Args:
        in_keys (sequence of NestedKey): the entries for the altitude.
        out_keys (sequence of NestedKey): the name of the scale code.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     AltitudeToDigits(["alt"], ["alt_code"]),
        ... )
    """

    def __init__(
        self, in_keys, out_keys
    ):
        in_keys_inv = []
        out_keys_inv = copy.copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if len(self.in_keys) != 1:
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should be 3."
            )
        if len(self.out_keys) != 1:
            raise ValueError(
                f"The number of out_keys ({len(self.out_keys)}) should be 1."
            )

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        altitude = tensordict[self.in_keys[0]]
        scale_code = _digits_encoding(altitude, altitude.device)
        tensordict[self.out_keys[0]] = scale_code
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
        output_spec["full_observation_spec"][self.out_keys[0]] = Unbounded(shape=(*output_spec.shape, 5), device=output_spec.device, dtype=torch.float32)        
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
