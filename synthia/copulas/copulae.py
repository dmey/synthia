import numpy as np
try:
    import copulae
except ImportError:
    copulae = None

from .copula import Copula


class CopulaeCopula(Copula):
    """A Copulae copula.
    """

    def __init__(self, copulae_class: type, **kw) -> None:
        """
        Args:
            copulae_class (type): Copulae class to use, e.g. copulae.GaussianCopula.
            **kw: Extra keyword arguments to pass to the Copulae class.
        """
        assert copulae, "copulae not installed but required for CopulaeCopula()"
        self.copulae_class = copulae_class
        self.copulae_kw = kw

    def fit(self, rank_standardized: np.ndarray) -> None:
        """Fit a Copulae copula to data.

        Args:
            rank_standardized (ndarray): 2D array of shape (feature, feature)
                with values in range [-1,1]

        Returns:
            None

        """
        self.model = self.copulae_class(dim=rank_standardized.shape[1], **self.copulae_kw)
        self.model.fit(rank_standardized)

    def generate(self, n_samples: int, qrng=False) -> np.ndarray:
        """Generate n_samples gaussian copula entries.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            2D array of shape (n_samples, feature) with gaussian copula entries.
        """
        assert not qrng, "qrng not supported"
        return self.model.random(n_samples)
