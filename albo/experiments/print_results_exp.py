from albo.experiments.optimize_via_albo_Gramacy_toy import optimize_gramacy_toy
from albo.experiments.optimize_via_albo_Gramacy_toy import joint_function

import torch
import botorch
import numpy as np
from matplotlib import pyplot as plt
from albo.objective_albo import AlboMCObjective

from botorch.models import FixedNoiseGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils import get_objective_weights_transform
from tqdm import tqdm
from gpytorch.mlls import ExactMarginalLogLikelihood


def gramacy(x):
    """Gramacy problem function
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    f = -(x1 + x2)
    c1 = (3. / 2) - x1 - 2. * x2 - (1. / 2) * np.sin(2. * np.pi * (x1 ** 2 - 2. * x2))
    c2 = x1 ** 2 + x2 ** 2 - 3. / 2
    return np.stack([f, c1, c2], axis=1)


def make_feasibility_plot_2d(
        ax,
        fcn,
        bounds,
        nx: int = 100,
        ny: int = 100,
        levels=None,
        levels_fmt='%2.1f'
):
    x_bounds = bounds[:, 0]
    x = np.linspace(x_bounds[0], x_bounds[1], num=nx)

    y_bounds = bounds[:, 1]
    y = np.linspace(y_bounds[0], y_bounds[1], num=ny)

    x_grid, y_grid = np.meshgrid(x, y)
    x_ = x_grid.flatten()
    y_ = y_grid.flatten()

    X = np.stack((x_, y_), axis=1)
    Z = fcn(X)

    contours = list()
    for i in range(Z.shape[1]):
        c = Z[:, i]
        c_grid = c.reshape((len(x), len(y)))

        if i > 0:
            cfill = ax.contourf(x_grid, y_grid, c_grid, levels=[0.0, np.inf], colors='lightgray')
            clines = ax.contour(x_grid, y_grid, c_grid, levels=[0.0])
            contours.append((cfill, clines))
        else:
            clines = ax.contour(x_grid, y_grid, c_grid, levels=levels)
            ax.clabel(clines, fmt=levels_fmt, colors='k')
            contours.append((clines))

    return contours


def plot_firstly(train_x):
    fh = plt.figure(figsize=[6, 6])
    axes = fh.subplots()
    contours = make_feasibility_plot_2d(axes, gramacy,
                                        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double))
    local_optimizers = {'x': [0.1954, 0.7197, 0.0], 'y': [0.4044, 0.1411, 0.75]}
    train_x1 = train_x[:, 0]
    train_x2 = train_x[:, 1]
    f_opt = 0.599788
    axes.scatter(local_optimizers['x'], local_optimizers['y'], marker='*', zorder=10, color='k', s=50)
    axes.scatter(local_optimizers['x'][:1], local_optimizers['y'][:1], marker='*', zorder=10, color='r', s=100)
    axes.scatter(train_x1, train_x2, marker='*', zorder=10, color='b', s=50)
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_title('Gramacy toy problem')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    plt.show()


def do_meshgrid(bounds, n):
    x_1 = np.linspace(bounds[0][0], bounds[1][0], num=n)
    x_2 = np.linspace(bounds[0][1], bounds[1][1], num=n)
    x_1_grid, x_2_grid = np.meshgrid(x_1, x_2)
    x1_ = x_1_grid.flatten()
    x2_ = x_2_grid.flatten()
    x = np.stack((x1_, x2_), axis=1)
    x = torch.from_numpy(x)
    return x, x_1_grid, x_2_grid


def return_model(train_x):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    train_y = joint_function(train_x)
    m = train_y.shape[1]
    d = train_x.shape[1]

    model = FixedNoiseGP(train_X=train_x,
                         train_Y=train_y,
                         train_Yvar=torch.full_like(train_y, 1e-6),
                         outcome_transform=Standardize(m=m),
                         input_transform=Normalize(d=d, bounds=bounds))
    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    mll = mll.to(train_x)
    botorch.fit.fit_gpytorch_mll(mll)
    model.eval()

    return model


def callable_functions(m):
    zeros = torch.zeros(m, dtype=torch.double)
    zeros[0] = 1
    weights_objective = zeros
    objective_callable = get_objective_weights_transform(weights_objective)

    constraint_callable_list = []
    for i in range(m - 1):
        zeros = torch.zeros(m, dtype=torch.double)
        zeros[i + 1] = 1
        weights_constraint = zeros
        constraint_callable_list.append(get_objective_weights_transform(weights_constraint))
    return objective_callable, constraint_callable_list


def dual_values(train_x, bounds_x, bounds_lmbds, n):
    # n - количество точек в разбиении лямбд и иксов
    train_y = joint_function(train_x)
    m = train_y.shape[1]

    model = return_model(train_x)

    sampler = IIDNormalSampler(sample_shape=1024, seed=100)
    objective_callable, constraint_callable_list = callable_functions(m)

    x, x_1_grid, x_2_grid = do_meshgrid(bounds_x, n)
    lmbds, lmbd_1_grid, lmbd_2_grid = do_meshgrid(bounds_lmbds, n)

    for i in tqdm(range(len(lmbds))):
        lmbd = lmbds[i].unsqueeze(1)
        albo_risk_objective = AlboMCObjective(
            objective=objective_callable,
            constraints=constraint_callable_list,
            penalty_rate=1,
            lagrange_mults=lmbd)
        z = torch.mean(albo_risk_objective.forward(sampler(model.posterior(x))), dim=0).unsqueeze(1)
        if i == 0:
            ans = z
        else:
            ans = torch.cat((ans, z), dim=1)
    ans = torch.max(ans, dim=0).values.detach().numpy()
    ans_grid = ans.reshape((n, n))
    return ans_grid, (x, x_1_grid, x_2_grid), (lmbds, lmbd_1_grid, lmbd_2_grid)


def print_dual_values(train_x, bounds_x, bounds_lmbds, n, lambdas_trace):
    fh = plt.figure(figsize=[6, 6])
    axes = fh.subplots()

    ans_grid, (x, x_1_grid, x_2_grid), (lmbds, lmbd_1_grid, lmbd_2_grid) = dual_values(train_x, bounds_x, bounds_lmbds, n)
    clines = axes.contour(lmbd_1_grid, lmbd_2_grid, ans_grid, levels=None)
    axes.clabel(clines, fmt='%2.1f', colors='k')
    axes.plot(lambdas_trace[:, 0, 0], lambdas_trace[:, 1, 0])
    plt.show()


def albo_objective_values_for_lambda(train_x, bounds, lmbds, n, x_candidate):
    train_y = joint_function(train_x)
    m = train_y.shape[1]
    x, x_1_grid, x_2_grid = do_meshgrid(bounds, n)
    model = return_model(train_x)
    objective_callable, constraint_callable_list = callable_functions(m)
    albo_objective = AlboMCObjective(
            objective=objective_callable,
            constraints=constraint_callable_list,
            penalty_rate=1,
            lagrange_mults=lmbds)
    sampler = IIDNormalSampler(sample_shape=1024, seed=100)
    pst = model.posterior(x)
    smpl = sampler(pst)
    z = torch.mean(albo_objective.forward(smpl), dim=0)
    z = z.reshape((n, n))
    value_at_candidate = torch.mean(albo_objective.forward(sampler(model.posterior(x_candidate))), dim=0)
    return z, (x, x_1_grid, x_2_grid), value_at_candidate


def print_albo_objective_values_for_lambda(train_x, bounds, lmbds, n, x_candidate, show=True, save=False, path=None):
    # train_x - обучающая выборка для модели
    # bounds - в каких пределах рисуем иксы
    # lmbds - для каких фиксированных лямбда строим рисунки
    # n - точек в разбиении икса

    fh = plt.figure(figsize=[6, 6])
    axes = fh.subplots()
    values, (x, x_1_grid, x_2_grid), value_at_candidate = albo_objective_values_for_lambda(train_x, bounds, lmbds, n, x_candidate)
    clines = axes.contour(x_1_grid, x_2_grid, values.detach().numpy(), levels=None)
    axes.clabel(clines, fmt='%2.3f', colors='k')
    axes.scatter([x_candidate[0][0]], [x_candidate[0][1]], marker='*', zorder=10, color='r', s=10)
    axes.set_xlim([bounds[0][0].item(), bounds[1][0].item()])
    axes.set_ylim([bounds[0][1].item(), bounds[1][1].item()])
    axes.set_title('lmbds={0}, f(x^*)={1}'.format(lmbds, value_at_candidate.item()))
    axes.set_xlabel('x1')
    axes.set_ylabel('x2')
    plt.grid()
    if save:
        path = path
        plt.savefig(path)
    if show:
        plt.show()


if __name__ == '__main__':
    train_x = torch.tensor(
        [[0.5718, 0.2027],
         [0.1001, 0.4709],
         [0.0046, 0.3743],
         [0.9466, 0.0061],
         [0.8808, 0.9170]], dtype=torch.float64)
    number_inner_loops = 200
    trace = optimize_gramacy_toy(train_x, eta=0.1, number_of_inner_loops=number_inner_loops)
    # plot_firstly(train_x)
    print_dual_values(train_x=trace['seed_points'],
                      bounds_x=torch.tensor([[0.0, 0.0], [1., 1.]]),
                      bounds_lmbds=torch.tensor([[0.0, 0.0], [2., 2]]),
                      n=15,
                      lambdas_trace=torch.tensor(trace['lagrange_mults_inner'][0]))
    # for i in tqdm(range(number_inner_loops)):
    #     if i % 1 == 0:
    #         print_albo_objective_values_for_lambda(train_x,
    #                                                torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    #                                                torch.tensor(trace['lagrange_mults_inner'][0][i]),
    #                                                n=30,
    #                                                x_candidate=trace['x_inner'][0][i],
    #                                                show=False,
    #                                                save=True,
    #                                                path='/Users/Dasha/desktop/experiment_plots_eta0.1/iter{0}'.format(i))

    # print_albo_objective_values_for_lambda(train_x, torch.tensor([[1.283627560956143], [0.0]]), 50)