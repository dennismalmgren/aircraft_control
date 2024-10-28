
import copy

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
import torch
from scipy.spatial.transform import Rotation as R

def _multi_scale_sinusoidal_encoding(position, device, N=5, base_scale=1, scale_factor=torch.e):
    """
    Generate a 1D multi-scale sinusoidal positional encoding.
    
    Parameters:
    - position: The input position to encode (scalar).
    - N: The number of scale levels (int).
    - base_scale: The initial scale level (float, typically 1).
    - scale_factor: The factor by which each subsequent scale increases (float, typically e or the golden ratio).
    
    Returns:
    - encoding: A numpy array of length N with the sinusoidal encoding values at each scale level.
    """
    encoding = torch.zeros((*position.shape[:-1], N), device=device, dtype=torch.float32)
    for i in range(N):
        wavelength = base_scale * (scale_factor ** i)  # Wavelength increases by scale_factor for each scale
        encoding[..., i] = torch.sin(2 * torch.pi * position / wavelength).squeeze(-1)  # Sinusoidal function with the given wavelength
    return encoding

class AltitudeToScaleCode(Transform):
    """A transform to convert euler angles to a rotation matrix.

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
        self, in_keys, out_keys
    ):
        in_keys_inv = []
        out_keys_inv = copy.copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should be the number of out_keys ({len(self.in_keys)})."
            )
        
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            value = tensordict[in_key]
            scale_code = _multi_scale_sinusoidal_encoding(value, value.device, N = 17)
            tensordict[out_key] = scale_code
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
        output_spec["full_observation_spec"][self.out_keys[0]] = Unbounded(shape=(*output_spec.shape, 17), device=output_spec.device, dtype=torch.float32)        
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
