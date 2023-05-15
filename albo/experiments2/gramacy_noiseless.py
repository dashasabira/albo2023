import torch

from botorch.acquisition import qSimpleRegret, get_acquisition_function
from botorch.optim.optimize import optimize_acqf
from botorch.utils.objective import get_objective_weights_transform
from botorch.sampling import SobolQMCNormalSampler

from albo.inverted import train_model_fixed_noise
from albo.objective_albo import ExpAlboMCObjective
from albo.synthetic.gramacy_toy import gramacy

import warnings
warnings.filterwarnings("ignore")

bounds = torch.Tensor([[0.0, 0.0], [1.0, 1.0]])
d = bounds.shape[1]
I = torch.eye(d)

sampler = SobolQMCNormalSampler(
    num_samples=512,
    seed=2243,
    resample=False
)

objective_fcn = get_objective_weights_transform(torch.Tensor([1.0, 0.0, 0.0]))
constraints_fcn = [
    get_objective_weights_transform(torch.Tensor([0.0, 1.0, 0.0])),
    get_objective_weights_transform(torch.Tensor([0.0, 0.0, 1.0]))
]


def run_gramacy_noiseless():
    x = torch.Tensor(
        [[0.4128, 0.5733],
        [0.8064, 0.1532],
        [0.4540, 0.2569],
        [0.9602, 0.2538],
        [0.6867, 0.1463]]
    )

    f = gramacy(x)

    exp_albo_objective = ExpAlboMCObjective(
        objective=objective_fcn,
        constraints=constraints_fcn,
        penalty_rate=5.0,
        eta=0.1,
    )

    y = torch.Tensor([0.001, 0.001])
    exp_albo_objective.lagrange_mults = y

    n_iter = 25
    for i in range(n_iter):
        print(x)
        print(f)

        model = train_model_fixed_noise(x, f, bounds, noise_level=1.e-4)

        lagrangian = qSimpleRegret(
            model=model,
            sampler=sampler,
            objective=exp_albo_objective
        )

        def dual(y, gradient_check=False):
            exp_albo_objective.lagrange_mults = y

            x_, l_ = optimize_acqf(
                acq_function=lagrangian,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100
            )

            samples = sampler.forward(model.posterior(x_))
            g_ = torch.zeros(2,)
            for i in range(2):
                t = exp_albo_objective.constraints[i](samples)
                g_[i] = exp_albo_objective.grad_penalty(t, y[i])

            g__ = None
            if gradient_check:
                g__ = torch.zeros(2)
                for i in range(2):
                    eps = 1.e-3
                    y_p = y + eps * I[i, :]
                    x_p, l_p, _, _ = dual(y_p)
                    y_m = y - eps * I[i, :]
                    x_m, l_m, _, _ = dual(y_m)
                    g__[i] = (l_p - l_m) / (2 * eps)

            return x_, l_, g_, g__

        x_, l_, g_, g__ = dual(y, gradient_check=True)
        print(x_, l_, g_, g__)
        l_prev = 10000.0
        step_size = 0.1
        y_best = None

        for j in range(20):
            x_, l_, g_, g__ = dual(y, gradient_check=False)
            l_diff = l_prev - l_
            y_ = y * (1 + 0.1 * g_)
            print(l_, y, y_, step_size, l_diff)
            if l_diff > 0:
                step_size = step_size * 1.1
                y_best = y
            else:
                step_size = step_size * 0.5
            y = y_
            l_prev = l_

        x_, l_, g_, g__ = dual(y_best, gradient_check=False)
        print("INNER POINT:", x_, l_, g_, g__)

        ei = get_acquisition_function(
            acquisition_function_name="qEI",
            model=model,
            objective=exp_albo_objective,
            X_observed=x
        )

        xx, e = optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100
        )

        print("NEXT POINT:", xx)
        print("GREEDY POINT:", x_)
        x = torch.cat([x, xx])
        f = torch.cat([f, gramacy(xx)])


def run_results1():
    x = torch.Tensor([
        [0.4128, 0.5733],
        [0.8064, 0.1532],
        [0.4540, 0.2569],
        [0.9602, 0.2538],
        [0.6867, 0.1463],
        [0.4667, 0.0000],
        [0.0000, 0.6299],
        [0.0000, 0.0000],
        [0.6918, 0.0000],
        [0.0000, 0.3972],
        [0.1757, 0.4744],
        [0.3671, 0.5056],
        [0.0000, 0.3046],
        [0.0000, 0.4501],
        [0.3359, 0.4573],
        [0.1248, 0.3947],
        [0.2368, 0.4176],
        [0.0000, 0.9489],
        [0.0000, 0.8668],
        [0.0000, 0.3739],
        [0.0000, 0.3816],
        [0.0000, 0.3816],
        [0.0000, 0.3814],
        [0.0000, 0.3821],
        [0.0000, 0.3823],
        [0.3010, 0.0763],
        [0.0289, 0.3815],
        [0.0000, 0.3819],
        [0.0000, 0.3819]
    ])

    f = torch.Tensor([
        [-0.9861, -0.1339, -1.0009],
        [-0.9596, -0.0283, -0.8262],
        [-0.7109,  0.9997, -1.2279],
        [-1.2140, -0.2240, -0.5136],
        [-0.8330,  0.0697, -1.0070],
        [-0.4667,  0.5436, -1.2822],
        [-0.6299,  0.7391, -1.1032],
        [-0.0000,  1.5000, -1.5000],
        [-0.6918,  0.7411, -1.0214],
        [-0.3972,  0.2249, -1.3422],
        [-0.6501,  0.1289, -1.2441],
        [-0.8727, -0.2285, -1.1096],
        [-0.3046,  0.5740, -1.4072],
        [-0.4501,  0.3064, -1.2974],
        [-0.7931, -0.2242, -1.1781],
        [-0.5195,  0.0914, -1.3286],
        [-0.6544, -0.0637, -1.2695],
        [-0.9489, -0.6973, -0.5996],
        [-0.8668, -0.7309, -0.7487],
        [-0.3739,  0.2522, -1.3602],
        [-0.3816,  0.2386, -1.3544],
        [-0.3816,  0.2385, -1.3543],
        [-0.3814,  0.2389, -1.3546],
        [-0.3821,  0.2378, -1.3540],
        [-0.3823,  0.2376, -1.3539],
        [-0.3772,  1.2362, -1.4036],
        [-0.4104,  0.2096, -1.3536],
        [-0.3819,  0.2381, -1.3541],
        [-0.3819,  0.2381, -1.3542]
    ])

    # lagrange_mults = torch.Tensor([5.3489e-01, 1.4013e-45])
    lagrange_mults = torch.Tensor([6.8e-01, 1.4013e-45])
    #lagrange_mults = torch.Tensor([4.8963e-01, 1.4013e-45])

    exp_albo_objective = ExpAlboMCObjective(
        objective=objective_fcn,
        constraints=constraints_fcn,
        penalty_rate=1.0,
        eta=0.1,
        lagrange_mults=lagrange_mults
    )

    model = train_model_fixed_noise(x, f, bounds, noise_level=1.e-4)

    lagrangian = qSimpleRegret(
        model=model,
        sampler=sampler,
        objective=exp_albo_objective
    )

    def primal(x):
        """ x: nb x nq x nd
        """
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        nb, nq, nd = x.shape
        nm = 3

        z = torch.zeros(nb, nq, nm + 2)
        z[:, :, -1] = lagrangian.forward(x).unsqueeze(1)

        samples = sampler.forward(model.posterior(x))
        z[:, :, 0] = exp_albo_objective.objective(samples).mean(0)

        for i in range(2):
            t = exp_albo_objective.constraints[i](samples)
            z[:, :, i+1] = - exp_albo_objective.penalty(t, lagrange_mults[i]).mean(0)

        z[:, :, 3] = z[:, :, :2].sum(-1)

        print(z)

    # primal(x)

    x_opt = torch.Tensor([
        [0.0, 0.3819],
        [0.1954, 0.4044],
        [0.7197, 0.1411],
        [0, 0.75]
    ])

    primal(x_opt)

if __name__ == '__main__':
    run_gramacy_noiseless()
    #run_results1()
