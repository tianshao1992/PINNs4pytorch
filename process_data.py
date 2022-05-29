import numpy as np
import torch
from distutils.version import LooseVersion
import skopt



class data_norm():

    def __init__(self, data, method="min-max"):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

    def norm(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device)) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std, device=x.device))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std)

        return x

    def back(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device)) + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std, device=x.device)) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min) + self.min
            elif self.method == "mean-std":
                x = x * (self.std) + self.mean
        return x



def sample(n_samples, dimension, sampler="pseudo"):
    """Generate random or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudo(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampler is not available.")


def pseudo(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config.real(np))
    return np.random.random(size=(n_samples, dimension)).astype(dtype=np.float32)


def quasirandom(n_samples, dimension, sampler):
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            space = [(0.0, 1.0)] * dimension
            return np.array(
                sampler.generate(space, n_samples + 2)[2:], dtype=np.float32
            )
    space = [(0.0, 1.0)] * dimension
    return np.array(sampler.generate(space, n_samples), dtype=np.float32)