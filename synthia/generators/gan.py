from typing import Literal, Union
import torch
from torch.nn import Module
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GAN(Module):
    """
    PyTorch implementation of a Generative Adversarial Network.
    """
    def __init__(
            self,
            generator_deep_layers: list[int] = [32, 32],
            discriminator_deep_layers: list[int] = [32, 32],
            device: Literal['cpu', 'cuda', 'auto'] = 'auto',
            optimizer: Literal['adam', 'sgd'] = 'adam',
            generator_fake_size: int = 1000,
        )->None:
        """
        Args:
            generator_deep_layers (list[int]): Number of deep layers for the generator. (Default: [32, 32])
            discriminator_deep_layers (list[int]): Number of deep layers for the discriminator. (Default: [32, 32])
            device (Literal['cpu', 'cuda', 'auto']): Device to use for training. Use 'auto' to automatically select the device.
            optimizer (Literal['adam', 'sgd']): Optimizer to use for training.
            generator_fake_size (int): Number of fake samples to generate for the generator. (Default: 1000).
        
        Returns:
            None
        """
        super().__init__()

        self.device = None
        match device:
            case 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            case 'cpu':
                self.device = torch.device('cpu')
            case 'cuda':
                self.device = torch.device('cuda')
            case _:
                raise ValueError(f"Invalid device {device}. Use 'cpu', 'cuda' or 'auto'.")

        self.n_gen = generator_deep_layers
        self.n_dis = discriminator_deep_layers
        self.opt = optimizer
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.fake_batch_size = generator_fake_size
    
    def init_weights(self, m: torch.nn.Module) -> None:
        """
        Initializes weights for the generator and discriminator.
        
        Args:
            m (torch.nn.Module): Module to initialize weights for.
        
        Returns:
            None
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)
    
    def get_optimizer(self, optimizer_type: Literal['adam', 'sgd'], lr: float) -> torch.optim.Optimizer:
        """
        Get the optimizer for the generator or discriminator.

        Args:
            optimizer (Literal['adam', 'sgd']): The optimizer to get.
            lr (float): The learning rate for the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        match optimizer_type:
            case 'adam':
                return torch.optim.Adam(self.generator.parameters(), lr=lr)
            case 'sgd':
                return torch.optim.SGD(self.generator.parameters(), lr=lr)
            case _:
                raise ValueError(f"Invalid optimizer {optimizer_type}. Use 'adam' or 'sgd'.")

    def construct_generator(
            self,
            n_features: int,
            dropout: float = 0.1,
            latent_dimension: int = 10,
            activation: Literal['tanh', 'sigmoid'] = 'tanh'
        ) -> torch.nn.Sequential:
        """
        Constructs the generator for the GAN.

        Args:
            n_features (int): Number of features for the generator.
            dropout (float): Dropout probability for the generator.
            latent_dimension (int): Dimension of the latent space.
            activation (Literal['tanh', 'sigmoid']): Activation function for the generator.
        """
        self.latent_dimension = latent_dimension
        match activation:
            case 'tanh':
                activation = torch.nn.Tanh()
            case 'sigmoid':
                activation = torch.nn.Sigmoid()
            case _:
                raise ValueError(f"Invalid activation {activation}. Use 'tanh' or 'sigmoid'.")
        #Create Generator
        self.deep_gen_layers = [latent_dimension] + self.n_gen + [n_features]
        self.deep_gen_layers = list(zip(self.deep_gen_layers[:-1], self.deep_gen_layers[1:]))
        self.generator = torch.nn.Sequential(
            *[x for i, j in self.deep_gen_layers for x in [torch.nn.Linear(i, j), torch.nn.LeakyReLU()]],
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.deep_gen_layers[-1][1], self.deep_gen_layers[-1][1]),
            activation
        )
        #Send to device
        self.generator.to(self.device)
        #Initialize weights
        self.generator.apply(self.init_weights)
    
    def construct_discriminator(
            self,
            n_features: int
        )->torch.nn.Sequential:
        """
        Constructs the discriminator for the GAN.

        Args:
            n_features (int): Number of features for the discriminator.
        """
        #Create Discriminator
        self.deep_dis_layers = [n_features] + self.n_dis + [2]
        self.deep_dis_layers = list(zip(self.deep_dis_layers[:-1], self.deep_dis_layers[1:]))
        self.discriminator = torch.nn.Sequential(
            *[x for i,j in self.deep_dis_layers for x in [torch.nn.Linear(i, j), torch.nn.LeakyReLU()]],
            torch.nn.Linear(2,1)
        )
        #Send to device
        self.discriminator.to(self.device)
        #Initialize weights
        self.discriminator.apply(self.init_weights)

    def fit(
            self,
            X: Union[torch.Tensor, np.array],
            global_iterations: int,
            discriminator_iterations: int = 1,
            lr: float = 0.001,
            batch_size: int = 32,
            generator_kwargs: dict = {},
            discriminator_kwargs: dict = {}
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fits the model to data.

        Args:
            X (Union[torch.Tensor, np.array]): Data to fit the model to.
            global_iterations (int): Number of global iterations for training.
            discriminator_iterations (int): Number of discriminator iterations for training.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            generator_kwargs (dict): Keyword arguments for the generator. See :func:`construct_generator`.
            discriminator_kwargs (dict): Keyword arguments for the discriminator. See :func:`construct_discriminator`.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Generator and discriminator loss.
        """

        #Check X values
        assert X.max() <= 1 and X.min() >= -1, "Values must be within [-1,1]. Try using np.tanh first."

        self.construct_discriminator(X.shape[1], **discriminator_kwargs)
        self.construct_generator(X.shape[1], **generator_kwargs)

        with self.device:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif isinstance(X, torch.Tensor):
                X = X.float()
            else:
                raise TypeError(f"Invalid type {type(X)} for X. Use torch.Tensor or np.array.")
            X = X.to(self.device)
            self.n_features = X.shape[1]

            self.generator_optimizer = self.get_optimizer(self.opt, lr)
            self.discriminator_optimizer = self.get_optimizer(self.opt, lr)

            #Train GAN
            for _ in range(global_iterations):
                for i in range(discriminator_iterations):

                    #Sample from X
                    real = X[torch.randperm(X.shape[0])[:batch_size]]
                    #At the end the batch size might be smaller than the specified batch size
                    actual_batch_size = real.shape[0]
                    fake = self.generator(torch.randn(actual_batch_size, self.latent_dimension))

                    self.discriminator.zero_grad()
                    disc_real = self.discriminator(real)
                    disc_fake = self.discriminator(fake)

                    loss_real = self.loss(disc_real, torch.ones(actual_batch_size, 1))
                    #Inserting 1 for y effectively calculates -log(D(x)), as the other term on
                    #the loss vanishes as it is proportional to y

                    loss_fake = self.loss(disc_fake, torch.zeros(actual_batch_size, 1))
                    #In a similar fashion, inserting 0 for y yields -log(1-D(G(z)))
                    loss_discriminator = loss_real + loss_fake
                    loss_discriminator.backward()
                    self.discriminator_optimizer.step()

                self.generator.zero_grad()
                new_fake = self.generator(torch.randn(self.fake_batch_size, self.latent_dimension))
                new_disc_fake = self.discriminator(new_fake)
                loss_generator = self.loss(new_disc_fake, torch.ones(self.fake_batch_size, 1))
                #Inserting 0 for y calculates -log(D(G(z)))
                loss_generator.backward()
                self.generator_optimizer.step()
        
        return loss_discriminator, loss_generator
        
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generates samples from the copula.
        
        Args:
            n_samples (int): Number of samples to generate.
        
        Returns:
            torch.Tensor: Samples from the copula.
        """
        with torch.no_grad():
            with self.device:
                return self.generator(torch.randn(n_samples, self.latent_dimension))