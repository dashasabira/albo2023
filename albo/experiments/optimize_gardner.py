from albo.acquisition import OptimizationViaALBO
from albo.acquisition import OptimizationViaCMCO
import torch
import numpy as np
import json
from tqdm import tqdm
from print_results_exp import make_feasibility_plot_2d
from matplotlib import pyplot as plt


def gardner_function(x):
    x_1 = x[:, 0].unsqueeze(1)
    x_2 = x[:, 1].unsqueeze(1)

    f =-(np.cos(2.0 * x_1) * np.cos(x_2) + np.sin(x_1))
    c = np.cos(x_1) * np.cos(x_2) - np.sin(x_1) * np.sin(x_2) + 0.5

    return torch.cat((f, c), dim=1)


def plot_gardner(train_x=None):
    fh = plt.figure(figsize=[6, 6])
    axes = fh.subplots()
    contours = make_feasibility_plot_2d(axes, gardner_function,
                                        bounds=torch.tensor([[0.0, 0.0], [6.0, 6.0]], dtype=torch.double))
    if train_x is not None:
        train_x1 = train_x[:, 0]
        train_x2 = train_x[:, 1]
        axes.scatter(train_x1, train_x2, marker='*', zorder=10, color='b', s=50)
    plt.show()


def optimize_gardner(train_x=None,
                     values=None,
                     bounds=torch.tensor([[0.0, 0.0], [6.0, 6.0]], dtype=torch.double),
                     penalty_rate=5.,
                     eta=1.,
                     number_of_outer_loops=30,
                     number_of_inner_loops=60,
                     print_trace=False,
                     mix=True,
                     exploitation=False,
                     mc_samples=1024,
                     raw_samples=100,
                     number_restarts=6,
                     seed=100,
                     seed_points=5,
                     noise_level=1e-4,
                     name_objective="classic",
                     case="albo"
                         ):
    if case == "albo":
        opt = OptimizationViaALBO(joint_function=gardner_function,
                                  bounds=bounds,
                                  train_x=train_x,
                                  values=values,
                                  number_of_inner_loops=number_of_inner_loops,
                                  number_of_outer_loops=number_of_outer_loops,
                                  penalty_rate=penalty_rate,
                                  eta=eta,
                                  print_trace=print_trace,
                                  mix=mix,
                                  exploitation=exploitation,
                                  mc_samples=mc_samples,
                                  raw_samples=raw_samples,
                                  number_restarts=number_restarts,
                                  seed=seed,
                                  seed_points=seed_points,
                                  noise_level=noise_level,
                                  name_objective=name_objective)
    if case == "cmco":
        opt = OptimizationViaCMCO(joint_function=gardner_function,
                                  bounds=bounds,
                                  train_x=train_x,
                                  values=values,
                                  number_of_inner_loops=number_of_inner_loops,
                                  number_of_outer_loops=number_of_outer_loops,
                                  print_trace=print_trace,
                                  mix=mix,
                                  exploitation=exploitation,
                                  mc_samples=mc_samples,
                                  raw_samples=raw_samples,
                                  number_restarts=number_restarts,
                                  seed=seed,
                                  seed_points=seed_points,
                                  noise_level=noise_level)
    trace = opt.optimize()
    return trace


if __name__ == '__main__':
    results = []
    for i in tqdm(range(1, 2)):
        results.append(
            optimize_gardner(
                train_x=None,
                case="albo",
                mix=True,
                exploitation=False,
                name_objective="classic",
                penalty_rate=5,
                eta=1,
                number_of_outer_loops=60,
                number_of_inner_loops=60
            )
        )
    with open("my_results_gardner_5_1_classic_.json", "w") as json_file:
        json.dump(results, json_file)

    with open("my_results_gardner_5_1_classic_.json", "r") as file:
        data = json.load(file)
    number_of_experiment_to_plot = 0
    number_of_outer_iteration_to_plot = 30
    trace = data[number_of_experiment_to_plot]
    train_x = torch.tensor(trace['seed_points'], dtype=torch.double)
    train_x = torch.cat((train_x, torch.tensor(trace['x'][0:number_of_outer_iteration_to_plot])[:, 0, :]))
    plot_gardner(train_x)