from typing import Optional, Union, Dict
import xarray as xr

from ..copulas.copula import Copula
from ..parameterizers.parameterizer import Parameterizer
from ..generators.copula import CopulaDataGenerator
from ..generators.fpca import FPCADataGenerator

class MixedFPCADataGenerator:
    def fit(self, data: xr.Dataset, 
            n_fpca_components: int,
            copula: Copula,
            parameterize_by: Optional[Union[Parameterizer, Dict[str, Parameterizer]]] = None,
            ) -> None:
        """tbd

        Args:
            data (Dataset): The input data, a dataset where all
                variables have the shape (sample,) or (sample, feature).
                FPCAs are used only for the latter shape.

            copula: The underlying copula to use, for example a GaussianCopula object.

            parameterize_by (Parameterizer or mapping, optional): The
                following forms are valid:
                
                - Parameterizer
                - per-variable mapping {var name: Parameterizer}

        Returns:
            None
        """
        
        copula_input = xr.Dataset()
        self.fpca_generators = {}

        for name in data:
            da = data[name]
            if da.ndim == 1:
                copula_input[name] = xr.DataArray(da, dims=da.dims)
            elif da.ndim == 2:
                fpca = FPCADataGenerator()
                fpca.fit(da, n_fpca_components=n_fpca_components)
                copula_input[name] = xr.DataArray(fpca.eig_scores, dims=(da.dims[0], '_' + da.dims[1]))
                self.fpca_generators[name] = fpca
            else:
                raise RuntimeError('only 1D/2D variables supported')

        self.copula_generator = CopulaDataGenerator()
        self.copula_generator.fit(copula_input, copula, parameterize_by=parameterize_by)

    def generate(self, n_samples: int) -> xr.Dataset:
        """tbd

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Synthetic samples in the form of the input data
        """
        synthetic = self.copula_generator.generate(n_samples)

        for name in synthetic:
            da = synthetic[name]
            if da.ndim != 2:
                continue
            reconstructed = self.fpca_generators[name].reconstruct(da)
            synthetic = synthetic.assign({name: xr.DataArray(reconstructed, dims=(da.dims[0], da.dims[1][1:]))})
        
        return synthetic
