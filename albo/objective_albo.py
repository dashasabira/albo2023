from abc import abstractmethod
from typing import List, Callable, Optional

import torch
from torch import Tensor
from botorch.acquisition.objective import MCAcquisitionObjective


class AlboMCObjective(MCAcquisitionObjective):
    _default_mult = 0.001

    def __init__(
            self,
            objective: Callable[[Tensor], Tensor],
            constraints: List[Callable[[Tensor], Tensor]],
            penalty_rate: float = 1.,
            lagrange_mults: Optional[Tensor] = None,
            eta: Optional[float] = None
    ) -> None:
        """
            objective: 'sample_shape x batch_shape x q x m' -> 'sample_shape x batch_shape x q'
            constraints: list['sample_shape x batch_shape x q x m' -> 'sample_shape x batch_shape x q']

        """
        super(AlboMCObjective, self).__init__()
        self.objective = objective
        self.constraints = constraints
        self.penalty_rate = penalty_rate

        if eta is not None:
            self.eta = eta
        else:
            self.eta = self.penalty_rate

        if lagrange_mults is not None:
            self.register_buffer("lagrange_mults", lagrange_mults.clone())
        else:
            default_lagrange_mults = torch.full((len(constraints), 1), fill_value=self._default_mult,
                                                dtype=torch.double)
            self.register_buffer("lagrange_mults", default_lagrange_mults)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        obj = self.objective(samples)  # sample_size x batch_size x q
        penalty = torch.zeros_like(obj)
        for i, constraint in enumerate(self.constraints):
            penalty += self.penalty(constraint(samples), self.lagrange_mults[i])
        return obj - penalty

    def update_mults(self, samples: Tensor):
        """
            samples: 'sample_shape x batch_shape=1 x q=1 x m'
        """
        for i, constraint in enumerate(self.constraints):
            y = self.lagrange_mults[i]
            t = constraint(samples)
            y_new = y + self.eta * self.grad_penalty(t, y)
            self.lagrange_mults[i] = max(0, y_new.item())

    @abstractmethod
    def penalty(self, t: Tensor, y: float) -> Tensor:
        pass

    @abstractmethod
    def grad_penalty(self, t: Tensor, m: float):
        pass


class ClassicAlboMCObjective(AlboMCObjective):
    def penalty(self, t: Tensor, y: float):
        """
            t: 'sample_size x batch_size x q'
            ouput: 'sample_size x batch_size x q'
        """
        r = self.penalty_rate

        return torch.where(
            y + r * t < 0,
            - y ** 2 / (2.0 * r) * torch.ones_like(t),
            y * t + (r / 2.0) * t ** 2
        )

    def grad_penalty(self, t, y):
        r = self.penalty_rate
        z = y + r * t

        return torch.where(z < 0, -y / r, t).mean()


class ExpAlboMCObjective(AlboMCObjective):
    def penalty(self, t: Tensor, y: float):
        """
            t: 'sample_size x batch_size x q'
            ouput: 'sample_size x batch_size x q'
        """
        r = self.penalty_rate

        return (y / r) * (torch.exp(r * t) - 1.0)

    def update_mults(self, samples: Tensor):
        """
            samples: 'sample_shape x batch_shape=1 x q=1 x m'
        """
        for i, constraint in enumerate(self.constraints):
            y = self.lagrange_mults[i]
            t = constraint(samples)
            y_new = y * (1 + self.eta * self.grad_penalty(t, y))
            self.lagrange_mults[i] = max(y / 10, min(y * 10, y_new))

    def grad_penalty(self, t, y):
        r = self.penalty_rate

        return ((1 / r) * (torch.exp(r * t) - 1.0)).mean()
