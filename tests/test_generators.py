from pathlib import Path
import pickle

import pytest
import numpy as np
import xarray as xr

try:
    import pyvinecopulib as pv
except ImportError:
    pv = None
import synthia as syn

THIS_DIR = Path(__file__).absolute().parent

def test_independent_dataset_generation():
    n_samples = 200
    n_features = [10, 20]
    input_data = xr.Dataset({
        'a': (('sample', 'foo'), np.random.normal(size=(n_samples, n_features[0]))),
        'b': (('sample', 'bar'), np.random.normal(size=(n_samples, n_features[1])))
    })

    generator = syn.IndependentDataGenerator()

    # TODO test all variants of `parameterize_by`
    generator.fit(input_data, parameterize_by=syn.QuantileParameterizer(n_quantiles=100))

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data['a'].shape == (n_synthetic_samples, n_features[0])
    assert synthetic_data['b'].shape == (n_synthetic_samples, n_features[1])

def test_independent_1d_feature_generation():
    n_samples = 200
    input_data = np.random.normal(size=n_samples)

    generator = syn.IndependentDataGenerator()

    generator.fit(input_data)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data.shape == (n_synthetic_samples,)

def test_independent_feature_generation_with_distribution():
    n_samples = 20
    n_features = 2
    input_data = np.random.normal(size=(n_samples, n_features))

    generator = syn.IndependentDataGenerator()

    dist_names = set(syn.DistributionParameterizer.get_dist_names())
    # Remove all very slow distributions
    dist_names -= set(['genexpon', 'levy_stable', 'recipinvgauss', 'vonmises', 'kstwo'])

    generator.fit(input_data, parameterize_by=syn.DistributionParameterizer(dist_names, verbose=True))

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data.shape == (n_synthetic_samples, n_features)
    assert isinstance(synthetic_data, np.ndarray)

def test_copula_dataset_generation():
    n_samples = 200
    n_features = [10, 20]
    input_data = xr.Dataset({
        'a': (('sample', 'foo'), np.random.normal(size=(n_samples, n_features[0]))),
        'b': (('sample', 'bar'), np.random.normal(size=(n_samples, n_features[1])))
    })

    generator = syn.CopulaDataGenerator()

    # TODO test all variants of `parameterize_by`
    generator.fit(input_data, copula=syn.GaussianCopula(), 
                  parameterize_by=syn.QuantileParameterizer(n_quantiles=100))

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data['a'].shape == (n_synthetic_samples, n_features[0])
    assert synthetic_data['b'].shape == (n_synthetic_samples, n_features[1])

def test_fpca_dataset_generation():
    n_samples = 200
    n_features = [10, 20]
    input_data = xr.Dataset({
        'a': (('sample', 'foo'), np.random.normal(size=(n_samples, n_features[0]))),
        'b': (('sample', 'bar'), np.random.normal(size=(n_samples, n_features[1])))
    })

    generator = syn.FPCADataGenerator()

    generator.fit(input_data, n_fpca_components=5)

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data['a'].shape == (n_synthetic_samples, n_features[0])
    assert synthetic_data['b'].shape == (n_synthetic_samples, n_features[1])


def test_mixed_fpca_dataset_generation():
    n_samples = 200
    n_features = [10]
    input_data = xr.Dataset({
        'a': (('sample', 'foo'), np.random.normal(size=(n_samples, n_features[0]))),
        'b': (('sample',), np.random.normal(size=(n_samples,)))
    })

    generator = syn.MixedFPCADataGenerator()

    generator.fit(input_data, n_fpca_components=5, copula=syn.GaussianCopula())

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data['a'].shape == (n_synthetic_samples, n_features[0])
    assert synthetic_data['b'].shape == (n_synthetic_samples,)

def test_gaussian_copula_feature_generation():
    n_samples = 200
    n_features = 100
    input_data = xr.DataArray(np.random.normal(size=(n_samples, n_features)))

    generator = syn.CopulaDataGenerator(verbose=True)

    # TODO test all variants of `parameterize_by`
    generator.fit(input_data, copula=syn.GaussianCopula(),
                  parameterize_by=syn.QuantileParameterizer(n_quantiles=100))

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data.shape == (n_synthetic_samples, n_features)
    assert isinstance(synthetic_data, xr.DataArray)

@pytest.mark.skipif(pv == None, reason="Skip test if pyvinecopulib not installed")
def test_vine_copula_feature_generation():
    n_samples = 200
    n_features = 100
    input_data = xr.DataArray(np.random.normal(size=(n_samples, n_features)))

    generator = syn.CopulaDataGenerator(verbose=True)

    # TODO test all variants of `parameterize_by`
    ctrl = pv.FitControlsVinecop(family_set=[pv.gaussian], trunc_lvl=1, select_trunc_lvl=False, show_trace=True)
    generator.fit(input_data, copula=syn.VineCopula(controls=ctrl),
                  parameterize_by=syn.QuantileParameterizer(n_quantiles=100))

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data.shape == (n_synthetic_samples, n_features)
    assert isinstance(synthetic_data, xr.DataArray)

def test_copula_ndarray_feature_generation():
    n_samples = 200
    n_features = 100
    input_data = np.random.normal(size=(n_samples, n_features))

    generator = syn.CopulaDataGenerator(verbose=True)

    generator.fit(input_data, copula=syn.GaussianCopula(),
                  parameterize_by=syn.QuantileParameterizer(n_quantiles=100))

    pickled = pickle.dumps(generator)
    generator = pickle.loads(pickled)

    n_synthetic_samples = 50
    synthetic_data = generator.generate(n_samples=n_synthetic_samples)

    assert synthetic_data.shape == (n_synthetic_samples, n_features)
    assert isinstance(synthetic_data, np.ndarray)
