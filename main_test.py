import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

expander_module = TensorDictModule(
    in_keys=["vals"],
    out_keys=["vals"],
        module = lambda vals: vals.unsqueeze(-1).expand(-1, -1, 100)
    )

td = TensorDict(
    {
        "vals": torch.randn((5, 47))
    },
    batch_size=torch.Size([5])
)


expander_module(td)

print(td)