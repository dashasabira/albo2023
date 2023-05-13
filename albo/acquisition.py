from abc import abstractmethod

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
from albo.objective_albo import ClassicAlboMCObjective, ExpAlboMCObjective
from botorch.acquisition.objective import ConstrainedMCObjective
from tqdm import tqdm


class MyOptimization:
    def __init__(self,
                 joint_function,
                 bounds: Tensor,
                 mc_samples: int = 1024,
                 raw_samples: int = 100,
                 number_restarts: int = 6,
                 number_of_outer_loops: int = 30,
                 number_of_inner_loops: int = 60,
                 seed: int = 100,
                 seed_points: int = 5,
                 noise_level: float = 1e-4,
                 train_x: Optional[Tensor] = None,
                 values: Optional[Tensor] = None,
                 print_trace: bool = False,
                 mix: bool = False,
                 # по умолчанию делается exploration, если mix, то последние 1/6 операций будут exploitation
                 exploitation: bool = False  # если выполнен, то все операции будут exploitation
                 ):
        self.joint_function = joint_function
        self.bounds = bounds
        self.mc_samples = torch.Size([mc_samples])
        self.raw_samples = raw_samples
        self.number_restarts = number_restarts
        self.number_of_outer_loops = number_of_outer_loops
        self.number_of_inner_loops = number_of_inner_loops
        self.seed_points = seed_points
        self.seed = seed
        self.noise_level = noise_level
        self.print_trace = print_trace
        self.mix = mix
        self.exploitation = exploitation
        self.sampler = SobolQMCNormalSampler(sample_shape=self.mc_samples, seed=self.seed)
        if train_x is None:
            self.train_x = torch.rand(size=(self.seed_points, bounds.shape[1]), dtype=torch.double)
            self.train_x = unnormalize(self.train_x, bounds)  # number_points x d
        else:
            self.train_x = train_x.clone()
        if values is None:
            self.values = self.joint_function(self.train_x)
        else:
            self.values = values.clone()

    def trained_model(self):
        d = self.bounds.shape[1]
        m = self.values.shape[1]
        model = FixedNoiseGP(train_X=self.train_x,
                             train_Y=self.values,
                             train_Yvar=torch.full_like(self.values, self.noise_level),
                             outcome_transform=Standardize(m=m),
                             input_transform=Normalize(d=d, bounds=self.bounds))
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        mll = mll.to(self.train_x)
        botorch.fit.fit_gpytorch_mll(mll)
        model.eval()
        return model

    @staticmethod
    def callable_functions_(m):
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

    @abstractmethod
    def print_trace_(self, i, trace):
        pass

    @abstractmethod
    def inner_optimization_loop_(self, trace, model, albo_objective):
        pass

    @abstractmethod
    def build_trace_(self):
        pass

    @abstractmethod
    def build_objective_(self, objective_callable, constraint_callable_list):
        pass

    def optimize(self):
        trace = self.build_trace_()

        m = self.values.shape[1]
        objective_callable, constraint_callable_list = self.callable_functions_(m)

        for i in tqdm(range(self.number_of_outer_loops)):
            albo_objective = self.build_objective_(objective_callable, constraint_callable_list)
            model = self.trained_model()
            albo_objective = self.inner_optimization_loop_(trace, model, albo_objective)

            if (i >= self.number_of_outer_loops - self.number_of_outer_loops / 6 and self.mix) or (
                    self.exploitation):
                qSR = qSimpleRegret(model=model, objective=albo_objective, sampler=self.sampler)
                candidate, acq_value_list = optimize_acqf(
                    acq_function=qSR,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=self.number_restarts,
                    raw_samples=self.raw_samples
                )
            else:
                lagrangian_values = torch.mean(albo_objective.forward(self.sampler(model.posterior(self.train_x))),
                                               dim=0)
                # print(lagrangian_values)
                best_f = max(lagrangian_values)
                qEI = qExpectedImprovement(model=model, objective=albo_objective, sampler=self.sampler,
                                           best_f=best_f.item())
                candidate, acq_value_list = optimize_acqf(
                    acq_function=qEI,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=self.number_restarts,
                    raw_samples=self.raw_samples
                )

            self.train_x = torch.cat((self.train_x, candidate), 0)
            self.values = torch.cat((self.values, self.joint_function(candidate)), 0)

            trace['x'].append(candidate.clone().tolist())
            if self.print_trace:
                self.print_trace_(i, trace)

        return trace


class OptimizationViaALBO(MyOptimization):
    def __init__(self,
                 joint_function,
                 bounds: Tensor,
                 mc_samples: int = 1024,
                 raw_samples: int = 100,
                 number_restarts: int = 6,
                 number_of_outer_loops: int = 20,
                 number_of_inner_loops: int = 100,
                 seed: int = 100,
                 seed_points: int = 5,
                 noise_level: float = 1e-4,
                 train_x: Optional[Tensor] = None,
                 values: Optional[Tensor] = None,
                 print_trace: bool = False,
                 mix: bool = False,
                 exploitation: bool = False,
                 penalty_rate: float = 1.,
                 eta: Optional[float] = None,
                 name_objective="classic"):
        self.penalty_rate = penalty_rate
        self.eta = eta
        self.name_objective = name_objective
        super().__init__(joint_function,
                         bounds,
                         mc_samples,
                         raw_samples,
                         number_restarts,
                         number_of_outer_loops,
                         number_of_inner_loops,
                         seed,
                         seed_points,
                         noise_level,
                         train_x,
                         values,
                         print_trace,
                         mix,
                         exploitation
                         )

    def inner_optimization_loop_(self, trace, model, albo_objective):
        trace_inner_lagranges = []
        trace_inner_x = []
        best_v = torch.inf
        best_l = albo_objective.lagrange_mults
        for j in range(self.number_of_inner_loops):
            qSR = qSimpleRegret(model=model, objective=albo_objective, sampler=self.sampler)
            candidate, acq_value_list = optimize_acqf(
                acq_function=qSR,
                bounds=self.bounds,
                q=1,
                num_restarts=self.number_restarts,
                raw_samples=self.raw_samples
            )
            trace_inner_lagranges.append(albo_objective.lagrange_mults.clone().tolist())
            trace_inner_x.append(candidate.clone().tolist())
            if acq_value_list < best_v:
                best_l = albo_objective.lagrange_mults
                best_v = acq_value_list
            albo_objective.update_mults(self.sampler(model.posterior(candidate)))
        albo_objective.lagrange_mults = best_l
        trace['lagrange_mults_inner'].append(trace_inner_lagranges)
        trace['x_inner'].append(trace_inner_x)
        trace['lagrange_mults_outer'].append(albo_objective.lagrange_mults.clone().tolist())
        return albo_objective

    def build_trace_(self):
        trace = {'seed_points': self.train_x.clone().tolist(), 'x': [], 'x_inner': [], 'lagrange_mults_inner': [],
                 'lagrange_mults_outer': []}
        return trace

    def print_trace_(self, i, trace):
        print('index:', i)
        print('new x:', trace['x'][-1])
        print('inner lagrange mults:')
        for i in range(10, 0, -1):
            print(trace['lagrange_mults_inner'][-1][-i])
        print('outer lagrange mults:', trace['lagrange_mults_outer'][-1])
        print('---')

    def build_objective_(self, objective_callable, constraint_callable_list):
        if self.name_objective == "classic":
            return ClassicAlboMCObjective(
                objective=objective_callable,
                constraints=constraint_callable_list,
                penalty_rate=self.penalty_rate,
                eta=self.eta)
        if self.name_objective == "exp":
            return ExpAlboMCObjective(
                objective=objective_callable,
                constraints=constraint_callable_list,
                penalty_rate=self.penalty_rate,
                eta=self.eta)


class OptimizationViaCMCO(MyOptimization):
    def inner_optimization_loop_(self, trace, model, objective):
        return objective

    def build_trace_(self):
        trace = {'seed_points': self.train_x.clone().tolist(), 'x': []}
        return trace

    def print_trace_(self, i, trace):
        print('index:', i)
        print('new x:', trace['x'][-1])
        print('---')

    def build_objective_(self, objective_callable, constraint_callable_list):
        return ConstrainedMCObjective(objective=objective_callable,
                                      constraints=constraint_callable_list,
                                      infeasible_cost=100)
