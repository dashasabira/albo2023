from albo.acquisition import OptimizationViaALBO
import torch
import numpy as np


def joint_function(x):
    x1 = x[:, 0].unsqueeze(1)
    x2 = x[:, 1].unsqueeze(1)

    f = -(x1 + x2)
    c1 = (3. / 2) - x1 - 2. * x2 - (1. / 2) * np.sin(2. * np.pi * (x1 ** 2 - 2. * x2))
    c2 = x1 ** 2 + x2 ** 2 - 3. / 2
    return torch.cat((f, c1, c2), dim=1)


def optimize_gramacy_toy(train_x=None,
                         bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
                         number_of_inner_loops=50,
                         number_of_outer_loops=1,
                         penalty_rate=1,
                         print_trace=False,
                         eta=None
                         ):
    opt = OptimizationViaALBO(joint_function=joint_function,
                              bounds=bounds,
                              train_x=train_x,
                              number_of_inner_loops=number_of_inner_loops,
                              number_of_outer_loops=number_of_outer_loops,
                              penalty_rate=penalty_rate,
                              eta=eta,
                              print_trace=print_trace)
    trace = opt.optimize()
    return trace


if __name__ == '__main__':
    train_x = torch.tensor(
        [[0.5718, 0.2027],
        [0.1001, 0.4709],
        [0.0046, 0.3743],
        [0.9466, 0.0061],
        [0.8808, 0.9170]], dtype=torch.float64)
    trace = optimize_gramacy_toy(train_x)
    for i in range(len(trace['x_inner'][0])):
        print(trace['x_inner'][0][i], '---', trace['lagrange_mults_inner'][0][i], '\n')