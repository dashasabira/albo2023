import torch
import botorch

from botorch.models import FixedNoiseGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qSimpleRegret
from botorch.optim import optimize_acqf

from botorch.utils.transforms import unnormalize
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.sampling import SobolQMCNormalSampler
from botorch.utils import get_objective_weights_transform

from typing import Optional

from torch import Tensor
from albo.objective_albo import AlboMCObjective


class OptimizationViaALBO:
    def __init__(self,
                 joint_function,
                 bounds: Tensor,
                 penalty_rate: float = 1.,
                 mc_samples: int = 1024,
                 raw_samples: int = 100,
                 number_restarts: int = 6,
                 number_of_outer_loops: int = 20,
                 number_of_inner_loops: int = 100,
                 seed: int = 100,
                 seed_points: int = 5,
                 noise_level: float = 1e-3,
                 train_x: Optional[Tensor] = None,
                 print_trace: bool = False
                 ):
        self.joint_function = joint_function
        self.bounds = bounds
        self.penalty_rate = penalty_rate
        self.mc_samples = torch.Size([mc_samples])
        self.raw_samples = raw_samples
        self.number_restarts = number_restarts
        self.number_of_outer_loops = number_of_outer_loops
        self.number_of_inner_loops = number_of_inner_loops
        self.seed_points = seed_points
        self.seed = seed
        self.noise_level = noise_level
        self.print_trace = print_trace

        if train_x is None:
            self.train_x = torch.rand(size=(self.seed_points, bounds.shape[1]), dtype=torch.double)
            self.train_x = unnormalize(self.train_x, bounds)  # number_points x d
        else:
            self.train_x = train_x.clone()

    def noise_(self, x):
        return torch.full_like(x, self.noise_level)

    def print_trace_(self, i, trace):
        print('index:', i)
        print('new x:', trace['x'][-1])
        print('inner lagrange mults:')
        for i in range(10, 0, -1):
            print(trace['lagrange_mults_inner'][-1][-i])
        print('outer lagrange mults:', trace['lagrange_mults_outer'][-1])
        print('---')

    def optimize(self):
        trace = {
            'x': [],
            'lagrange_mults_inner': [],
            'lagrange_mults_outer': []
        }

        d = self.bounds.shape[1]
        bounds_normalized = torch.cat((torch.zeros(1, d), torch.ones(1, d)), dim=0)
        m = self.joint_function(self.train_x).shape[1]

        sampler = SobolQMCNormalSampler(sample_shape=self.mc_samples, seed=self.seed)

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

        for i in range(self.number_of_outer_loops):
            trace_inner_lagranges = []
            values_functions = self.joint_function(self.train_x)

            albo_risk_objective = AlboMCObjective(
                objective=objective_callable,
                constraints=constraint_callable_list,
                penalty_rate=self.penalty_rate)

            model = FixedNoiseGP(train_X=self.train_x,
                                 train_Y=values_functions,
                                 train_Yvar=self.noise_(values_functions),
                                 outcome_transform=Standardize(m=m),
                                 input_transform=Normalize(d=d, bounds=self.bounds))
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            mll = mll.to(self.train_x)
            botorch.fit.fit_gpytorch_mll(mll)
            model.eval()

            for j in range(self.number_of_inner_loops):
                qSR = qSimpleRegret(model=model, objective=albo_risk_objective, sampler=sampler)
                candidate, acq_value_list = optimize_acqf(
                    acq_function=qSR,
                    bounds=bounds_normalized,
                    q=1,
                    num_restarts=self.number_restarts,
                    raw_samples=self.raw_samples
                )
                albo_risk_objective.update_mults(sampler(model.posterior(candidate)))
                trace_inner_lagranges.append(albo_risk_objective.lagrange_mults.clone().tolist())

            lagrangian_values = torch.mean(albo_risk_objective.forward(sampler(model.posterior(self.train_x))), dim=0)
            best_f = max(lagrangian_values)
            qEI = qExpectedImprovement(model=model, objective=albo_risk_objective, sampler=sampler,
                                       best_f=best_f.item())
            candidate, acq_value_list = optimize_acqf(
                acq_function=qEI,
                bounds=bounds_normalized,
                q=1,
                num_restarts=self.number_restarts,
                raw_samples=self.raw_samples
            )

            self.train_x = torch.cat((self.train_x, candidate), 0)
            trace['x'].append(candidate)
            trace['lagrange_mults_outer'].append(albo_risk_objective.lagrange_mults.clone().tolist())
            trace['lagrange_mults_inner'].append(trace_inner_lagranges)
            if self.print_trace:
                self.print_trace_(i, trace)

        return trace
