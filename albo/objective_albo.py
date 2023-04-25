from typing import List, Callable, Optional

import torch
from torch import Tensor
from botorch.acquisition.objective import MCAcquisitionObjective


class AlboMCObjective(MCAcquisitionObjective):
    _default_mult = 0.

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

    def penalty(self, t: Tensor, lagrange_mult: float):
        """
            t: 'sample_size x batch_size x q'
            ouput: 'sample_size x batch_size x q'
        """
        y = lagrange_mult
        r = self.penalty_rate

        return torch.where(
            y + r * t < 0,
            - y ** 2 / (2.0 * r) * torch.ones_like(t),
            y * t + (r / 2.0) * t ** 2
        )

    def update_mults(self, samples: Tensor):
        """
            samples: 'sample_shape x batch_shape=1 x q=1 x m'
        """
        for i, constraint in enumerate(self.constraints):
            y = self.lagrange_mults[i]
            r = self.penalty_rate
            c = constraint(samples)
            z = y + r * c
            y_new = y + self.eta * torch.where(z < 0, -y / r, c).mean()
            self.lagrange_mults[i] = y_new.item()
