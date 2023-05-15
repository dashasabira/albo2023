import torch
import botorch

from botorch.models import FixedNoiseGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition import qExpectedImprovement, qSimpleRegret, get_acquisition_function

from botorch.optim.optimize import optimize_acqf

from gpytorch import ExactMarginalLogLikelihood


def train_model_fixed_noise(x, y, bounds, noise_level=1.e-6):
    d = bounds.shape[1]
    m = y.shape[1]

    model = FixedNoiseGP(
        train_X=x,
        train_Y=y,
        train_Yvar=torch.full_like(y, noise_level),
        outcome_transform=Standardize(m=m),
        input_transform=Normalize(d=d, bounds=bounds)
    )

    mll = ExactMarginalLogLikelihood(
        likelihood=model.likelihood,
        model=model
    )

    # XXX: what does this do?
    mll = mll.to(x)

    # This fits kernel hyperparameters?
    botorch.fit.fit_gpytorch_mll(mll)

    # switch model to eval mode
    model.eval()

    return model


def iteration(
    x,
    y,
    model,
    bounds,
    objective: MCAcquisitionObjective,
    acquisition_function_name: str = "qEI",
    num_restart: int = 5
):
    acq_function = get_acquisition_function(
        acquisition_function_name=acquisition_function_name,
        model=model,
        objective=objective,
        X_observed=x,
    )

    candidate, value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=1,
        num_restarts=num_restart,
        raw_samples=1024,
        return_best_only=True
    )

    return candidate, value

