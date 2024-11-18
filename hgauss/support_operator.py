from torch import nn

class SupportOperator(nn.Module):
    def __init__(self, support):
        super().__init__()
        self.register_buffer("support", support)

    def forward(self, x):
        return (x.softmax(-1) * self.support).sum(-1, keepdim=True)
    