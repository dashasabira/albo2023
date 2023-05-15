import torch


def gramacy(x):
    x1 = x[:, 0].unsqueeze(1)
    x2 = x[:, 1].unsqueeze(1)

    f = -(x1 + x2)
    c1 = (3. / 2) - x1 - 2. * x2 - (1. / 2) * torch.sin(2. * torch.pi * (x1 ** 2 - 2. * x2))
    c2 = x1 ** 2 + x2 ** 2 - 3. / 2
    return torch.cat((f, c1, c2), dim=1)

