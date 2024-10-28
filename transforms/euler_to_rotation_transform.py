
import copy

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
import torch
from scipy.spatial.transform import Rotation as R


class EulerToRotation(Transform):
    """A transform to convert euler angles to a rotation matrix.

    Args:
        in_keys (sequence of NestedKey): the entries for the angles.
        out_keys (sequence of NestedKey): the name of rotation tensor.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     EulerToRotation(["psi", "theta", "phi"], ["rotation"]),
        ... )
    """

    def __init__(
        self, in_keys, out_keys
    ):
        in_keys_inv = []
        out_keys_inv = copy.copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if len(self.in_keys) != 3:
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should be 3."
            )
        if len(self.out_keys) != 1:
            raise ValueError(
                f"The number of out_keys ({len(self.out_keys)}) should be 1."
            )

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        angles = torch.cat([tensordict[angle] for angle in self.in_keys], dim=-1)
        try:
            rotation_matrix = R.from_euler("ZYX", angles).as_matrix()
        except:
            rotation_matrix = torch.zeros(angles.shape[0], 3, 3)

        tensordict[self.out_keys[0]] = torch.tensor(rotation_matrix.reshape((rotation_matrix.shape[0], -1)), 
                                                    device=tensordict[self.in_keys[0]].device, dtype=torch.float32)
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
        output_spec["full_observation_spec"][self.out_keys[0]] = Unbounded(shape=(*output_spec.shape, 9), device=output_spec.device, dtype=torch.float32)        
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec
