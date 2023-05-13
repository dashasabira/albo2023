from albo.acquisition import OptimizationViaALBO
import torch
import numpy as np


def f(x):
    return - (x - 1) * (x - 1) / 2


def c(x):
    return x


def joint_function(x):
    return torch.cat((f(x), c(x)), dim=1)


def optimize_gramacy_toy(train_x=None,
                         bounds=torch.tensor([[0.], [1.]], dtype=torch.double),
                         number_of_inner_loops=30,
                         number_of_outer_loops=10,
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
    train_x = torch.tensor([[-0.1063],
                            [-1.0511],
                            [-0.3702],
                            [0.2950]], dtype=torch.float64)
    bounds = torch.tensor([[-2.], [2]], dtype=torch.double)
    trace = optimize_gramacy_toy(train_x=train_x, bounds=bounds, print_trace=True)
