import torch


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


val = torch.tensor([[1234.56], [793.1]], device="cpu")

encoding = _digits_encoding(val, device="cpu")
print('ok')