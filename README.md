<div align="center">
  <img src="assets/img/logo.png" alt="synthia" height="120">

  ![PyPI](https://img.shields.io/pypi/v/synthia) ![CI](https://github.com/dmey/synthia/workflows/CI/badge.svg)

  [Overview](#overview) | [Documentation](#documentation) | [How to cite](#how-to-cite) | [Contributing](#contributing) | [Development notes](#development-notes) | [Copyright and license](#copyright-and-license) | [Acknowledgements](#acknowledgements)
</div>

## Overview

Synthia is a tool for generating multi-dimensional data in Python. It has a simple and succinct API to natively handle multidimensional data using [xarray](https://xarray.pydata.org)'s labeled arrays and datasets. It supports different methods of data generation such as[ functional Principle Component Analysis (fPCA)](https://dmey.github.io/synthia/fpca.html), and Parametric (Gaussian) and Vine [copula models](https://dmey.github.io/synthia/copula.html). Depending on the type of model, it can be used to fit and generate different types of multivariate data with (or a mix of) discrete, categorical, and continuous variables. For example, given some vertical profiles of atmospheric temperature, we can use Synthia to generate new but statistically similar profiles using a copula and fPCA model in three lines of code (Table 1). For more information about the features included in the latest version of Synthia, please see the [Synthia website](https://dmey.github.io/synthia).

**Table 1. Example application of Gaussian and fPCA classes in Synthia used to generate random profiles of atmospheric temperature similar to those included in the SAF dataset (REF). The xarray dataset structure is maintained and returned by Synthia.**

| Source                                   | Synthetic with Gaussian Copula                     | Synthetic with fPCA                            |
| ---------------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| `ds = syn.util.load_dataset()`           | `g = syn.CopulaDataGenerator()`                    | `g = syn.fPCADataGenerator()`                  |
|                                          | `g.fit(ds, syn.GaussianCopula())`                  | `g.fit(ds)`                                    |
|                                          | `g.generate(n_samples=500)`                        | `g.generate(n_samples=500)`                    |
|                                          |                                                    |                                                |
| ![dd](./assets/img/temperature_true.png) | ![dd](./assets/img/temperature_synth_gaussian.png) | ![dd](./assets/img/temperature_synth_fPCA.png) |


## Documentation

For installation instructions, getting started guides and tutorials, background information, and API reference summaries, please see the [Synthia website](https://dmey.github.io/synthia).


## How to cite

Please cite the software summary paper and software version using the following Digital Object Identifiers (DOIs) to [generate citations in your preferred style](https://citation.crosscite.org/):

| Software summary paper | Software version* |
| ---------------------- | ----------------- |
| [TODO](TODO)           | [TODO](TODO)      |

*please make sure to cite the same version you are using with the correct DOI. For a list of all available versions see the [list of available versions on TODO](TODO).


## Contributing

If you are looking to contribute, please read our [Contributors' guide](CONTRIBUTING.md) for details.


## Development notes

If you would like to know more about specific development guidelines, testing and deployment, please refer to our [development notes](DEVELOP.md).


## Copyright and license

Copyright 2020 D. Meyer and T. Nagler. Licensed under [MIT](LICENSE.txt).


## Acknowledgements

Special thanks to [@letmaik](https://github.com/letmaik) for his suggestions and contributions to the project.
