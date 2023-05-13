from albo.acquisition import OptimizationViaALBO
from albo.acquisition import OptimizationViaCMCO
import torch
import numpy as np
import json
from tqdm import tqdm


def joint_function(x):
    x1 = x[:, 0].unsqueeze(1)
    x2 = x[:, 1].unsqueeze(1)

    f = -(x1 + x2)
    c1 = (3. / 2) - x1 - 2. * x2 - (1. / 2) * np.sin(2. * np.pi * (x1 ** 2 - 2. * x2))
    c2 = x1 ** 2 + x2 ** 2 - 3. / 2
    return torch.cat((f, c1, c2), dim=1)


def optimize_gramacy_toy(train_x=None,
                         values=None,
                         bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
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
        opt = OptimizationViaALBO(joint_function=joint_function,
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
        opt = OptimizationViaCMCO(joint_function=joint_function,
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
    with open("my_results_albo_classic_rho1.json", "r") as file:
        data = json.load(file)
    for i in tqdm(range(0,1)):
        results.append(
            optimize_gramacy_toy(
                train_x=torch.tensor(data[i]['seed_points'], dtype=torch.double),
                case="albo",
                mix=True,
                exploitation=False,
                name_objective="exp",
                penalty_rate=2,
                eta=0.5,
                number_of_outer_loops=30,
                number_of_inner_loops=60
            )
        )
    with open("my_results_exp_rho2_eta0.5_new_cool.json", "w") as json_file:
        json.dump(results, json_file)


    # results = []
    # with open("my_results_albo_classic_rho1.json", "r") as file:
    #     data = json.load(file)
    # for i in tqdm(range(20)):
    #     results.append(
    #         optimize_gramacy_toy(
    #             train_x=torch.tensor(data[i]['seed_points'], dtype=torch.double),
    #             case="albo",
    #             mix=False,
    #             exploitation=True,
    #             name_objective="exp",
    #             penalty_rate=1,
    #             eta=0.2,
    #             number_of_outer_loops=30,
    #             number_of_inner_loops=60
    #         )
    #     )
    # with open("my_results_debag_expl.json", "w") as json_file:
    #     json.dump(results, json_file)
