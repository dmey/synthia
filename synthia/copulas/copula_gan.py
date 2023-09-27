from typing import Literal, Union, Annotated
from .copula import Copula
from ..generators.gan import GAN
from derivative import dxdt
import numpy as np
import torch
import zfit

def derivative(
        x: Annotated[torch.Tensor, 'CDF'],
        t: Annotated[torch.Tensor, 'Support'] = None,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
    """
    Derivative of the CDF with respect to support t.

    Args:
        x (Annotated[torch.Tensor, 'CDF']): CDF of the data.
        t (Annotated[torch.Tensor, 'Support']): Support of the data. Will default to the interval [0,1].
        device (torch.device): Device to use for computation. (Default: torch.device('cpu'))
    """
    if t is None:
        t = torch.linspace(0,1,x.shape[0], device=device)
    return dxdt(x.T, t).T

def compute_cdf(kde_model, n_samples=1000):
    """
    Computes the CDF of a KDE model using the cumulative trapezoidal rule.

    Args:
        kde_model: KDE model to compute CDF of.
        n_samples: Number of samples to use for computing the CDF. (Default: 1000)
    """
    lower_limit = float(kde_model.space.limits[0])
    upper_limit = float(kde_model.space.limits[1])

    support = torch.linspace(lower_limit, upper_limit, n_samples)
    pdf_values = torch.Tensor([float(kde_model.pdf(x)) for x in support])

    # Compute the CDF using cumulative trapezoidal rule
    cdf_values = torch.cumulative_trapezoid(pdf_values, dx=(support[1] - support[0]))

    # Normalize the CDF to [0, 1] by dividing by the final value
    cdf_values /= cdf_values[-1].clone()
    #Add zero on beginning
    cdf_values = torch.cat((torch.zeros(1), cdf_values))

    return cdf_values, support

def map_cdf(cdf_values, support)->function:
    """
    Maps the data in the support to its corresponding CDF value.

    Args:
        cdf_values: CDF values to map to.
        support: Support of the data.
    """
    return lambda x: np.interp(x, support, cdf_values)

def inverse_map_cdf(cdf_values, support):
    """
    Computes the inverse CDF of a KDE model using linear interpolation.

    Args:
        cdf_values: CDF values to compute inverse of.
        support: Support of the data.
    """
    return lambda x: np.interp(x, cdf_values, support)

class CopulaGAN(Copula, GAN):
    """
    Learns Copula from data using Generative Adversarial Networks.
    """
    def __init__(
            self,
            generator_deep_layers: list[int] = [32, 32],
            discriminator_deep_layers: list[int] = [32, 32],
            device: Literal['cpu', 'cuda', 'auto'] = 'auto',
            optimizer: Literal['adam', 'sgd'] = 'adam',
            generator_fake_size: int = 1000,
        ) -> None:
        super().__init__(
            generator_deep_layers,
            discriminator_deep_layers,
            device,
            optimizer,
            generator_fake_size
        )

    def initialize(self, X: torch.Tensor, **kwargs):
        # Generate marginals and sample space
        n_features = X.shape[1]

        # Create target KDEs and CDFs
        target_kdes = []
        target_pdfs = []
        cdfs = []
        real_spaces = []
        real_U = []
        for i in range(n_features):
            obs_space = zfit.Space('X' + str(i), limits=(X[:, i].min(), X[:, i].max()))
            kde = zfit.pdf.KDE1DimGrid(
                obs=obs_space,
                data=zfit.Data.from_tensor(tensor=X[:, i], obs=obs_space)
            )
            cdf, support = compute_cdf(kde, **kwargs)
            real_U.append(map_cdf(cdf, support)(X[:,i]))
            pdf = kde.pdf(obs_space, torch.linspace(X[:,i].min(),X[:,i].max(),n_samples))
            target_kdes.append(kde)
            cdfs.append(cdf)
            real_spaces.append(obs_space)
            target_pdfs.append(pdf)

        self.real_space = zfit.dimension.combine_spaces(*real_spaces)
        self.target_kdes = target_kdes
        self.target_cdf = torch.stack(cdfs, dim=1)
        self.target_pdf = torch.stack(target_pdfs, dim=1)
        self.real_U = torch.stack(real_U, dim=1)

    def construct_discriminator(
            self,
            n_features: int,
            hidden_size: int = 32,
        ) -> torch.nn.Sequential:
        """
        Constructs the discriminator for the GAN. It is a GRU that receives samples from CDFs as input
        and outputs a single value.

        Input (batch_size, sequence_length, n_features)
        Output (batch_size, 1)

        Args:
            n_features (int): Number of features for the discriminator.
            hidden_size (int): Hidden size for the GRU. (Default: 32)
        """
        self.discriminator = torch.nn.Sequential(
            torch.nn.GRU(
                input_size = n_features,
                hidden_size = hidden_size,
                batch_first=True),
            torch.nn.Linear(hidden_size, 1)
        ).to(self.device)
        self.discriminator.apply(self.init_weights)
        return self.discriminator

    def fit(
            self,
            X: Union[torch.Tensor, np.ndarray],
            global_iterations: int,
            discriminator_iterations: int = 1,
            lr: float = 0.001,
            batch_size: int = 32,
            generator_kwargs: dict = {},
            discriminator_kwargs: dict = {},
            CDFN: int = 1000
        ):
        """
        Uses GAN to learn a Copula flow from vector U sampled from the marginals of an arbitrary dataset.

        If each variable $X_i$ in a dataset has a CDF denoted by $F_i$, then the marginals are defined as $U_i = F_i(X_i)$.

        Args:
            X (Union[torch.Tensor, np.ndarray]): Dataset in format (n_samples, n_features).
            global_iterations (int): Number of global iterations to train for.
            discriminator_iterations (int): Number of discriminator iterations per global iteration. (Default: 1)
            lr (float): Learning rate for the optimizer. (Default: 0.001)
            batch_size (int): Batch size for training. (Default: 32)
            generator_kwargs (dict): Keyword arguments to pass to the generator. (Default: {})
            discriminator_kwargs (dict): Keyword arguments to pass to the discriminator. (Default: {})
            CDFN (int): Number of samples to use for computing the CDF. (Default: 1000)
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X = X.float()
        else:
            raise TypeError(f"Invalid type {type(X)} for X. Use torch.Tensor or np.array.")

        self.initialize(X, n_samples = CDFN)

        self.construct_generator(X.shape[1], activation='sigmoid', **generator_kwargs)
        self.construct_discriminator(X.shape[1], **discriminator_kwargs)

        with self.device:

            self.generator_optimizer = self.get_optimizer(self.opt, lr)
            self.discriminator_optimizer = self.get_optimizer(self.opt, lr)

            #Train
            for _ in range(global_iterations):
                for __ in range(discriminator_iterations):
                    #Train Discriminator
                    rand = torch.rand(batch_size, self.latent_dimension, device=self.device)
                    fake_U = self.generator(rand)
                    real_U = self.real_U[torch.randperm(self.real_U.shape[0])[:batch_size]]
                    real_discriminant, _ = self.discriminator(real_U)
                    fake_discriminant, _ = self.discriminator(fake_U)
                    self.discriminator.zero_grad()
                    (
                        self.loss(real_discriminant, torch.ones_like(real_discriminant)) +\
                        self.loss(fake_discriminant, torch.zeros_like(fake_discriminant))
                    ).backward()
                    self.discriminator_optimizer.step()
                self.generator.zero_grad()
                rand = torch.rand(batch_size, self.latent_dimension, device=self.device)
                fake_U = self.generator(rand)
                fake_discriminant, _ = self.discriminator(fake_U)
                loss = self.loss(fake_discriminant, torch.ones_like(fake_discriminant))
                loss.backward()
                self.generator_optimizer.step()
    
    def generate(self, n_samples: int, **kws) -> np.ndarray:
        return super().generate(n_samples, **kws)