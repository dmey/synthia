from typing import Optional, Union
import numpy as np
import xarray as xr

from ..parameterizers.quantile import QuantileParameterizer
from ..util import to_feature_array, from_feature_array, per_feature

class FPCADataGenerator:
    def fit(self, data: Union[np.ndarray, xr.DataArray, xr.Dataset],
            n_fpca_components: int, n_samples_reduced: Optional[int]=None) -> None:
        """tbd

        Args:
            data (ndarray or DataArray or Dataset): The input data, either a
                2D array of shape (sample, feature) or a dataset where all
                variables have the shape (sample[, ...]).

            n_fpca_components (int): Reduces the number of features
            
            n_samples_reduced (int, optional): Reduces the number of samples

        Returns:
            None
        """

        data, self.data_info = to_feature_array(data)

        # TODO instead of number of fpca components, can we do it by retained variance?
        assert n_fpca_components > 0
        self.n_fpca_components = n_fpca_components
        self.n_samples_reduced = n_samples_reduced
        
        n_samples, n_features = data.shape
        assert self.n_fpca_components <= n_features
        assert self.n_samples_reduced is None or self.n_samples_reduced <= n_samples

        self.means = data.mean(axis=0)
        assert self.means.shape == (n_features,)

        cov = np.cov(data.T)
        assert cov.shape == (n_features, n_features)
        
        eig_vals, eig_funs = np.linalg.eig(cov)
        assert eig_vals.shape == (n_features,)
        assert eig_funs.shape == (n_features, n_features)
        
        # We assume real eigenvalues and due to finite precision
        # it might happen that we get some spurious complex parts, so we remove the imaginary part again.
        eig_vals = np.real(eig_vals)
        self.eig_funs = np.real(eig_funs)

        data_centered = (data.T - self.means).T
        assert data_centered.shape == data.shape

        eig_scores = np.dot(data_centered, self.eig_funs[:, :self.n_fpca_components])
        assert eig_scores.shape == (n_samples, self.n_fpca_components)

        if self.n_samples_reduced is None:
            self.eig_scores = eig_scores
        else:
            self.eig_parameterizer = []
            for i in range(self.n_fpca_components):
                p = QuantileParameterizer(self.n_samples_reduced)
                # TODO remove type conversion?
                p.fit(np.array(eig_scores[:,i], dtype='d'))
                self.eig_parameterizer.append(p)

    def generate(self, n_samples: int,
                 scaling_factor: Optional[float]=None) \
                 -> Union[np.ndarray, xr.DataArray, xr.Dataset]:
        """tbd

        Args:
            n_samples (int): Number of samples to generate.

            scaling_factor (float, optional): tbd

        Returns:
            Synthetic samples in the form of the input data
        """
        eig_scores = np.empty((n_samples, self.n_fpca_components))
        for i in range(self.n_fpca_components):
            if self.n_samples_reduced is None:
                eig_score = self.eig_scores[:,i]
            else:
                eig_score = self.eig_parameterizer[i].generate(n_samples)

            # TODO check if factor should be applied after np.quantile
            # TODO check if factor should be done with median like in data_generator.py
            if scaling_factor:
                eig_score = eig_score * scaling_factor

            eig_scores[:, i] = np.quantile(eig_score, q=np.random.uniform(0, 1, n_samples), interpolation='linear')
        
        samples = self.reconstruct(eig_scores)

        samples = from_feature_array(samples, self.data_info)

        return samples

    def reconstruct(self, eig_scores: np.ndarray) -> np.ndarray:
        samples = np.array(self.means) + np.dot(eig_scores, self.eig_funs[:, :self.n_fpca_components].T)
        return samples
