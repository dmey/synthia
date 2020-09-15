# Development notes

This are temporary development notes until we have a stable version on PyPI.


## Conda environment (optional)

Only required if you wish to install all the required prerequisites in [Miniconda](https://docs.conda.io/en/latest/miniconda.html)/[Anaconda](https://www.anaconda.com/) for local development and testing

```
conda env create -f environment.yml
```

Then activate with `conda activate synthia`.


## Install synthia

During development:

```
pip install -e .
```


## Documentation

```
sphinx-build -v -b html docs/ docs/_build/
```

Use `SKIP_NB=1` to skip building the example notebooks (they take a while to build):
```
SKIP_NB=1 sphinx-build -v -b html docs/ docs/_build/
```

## Docstrings

Use [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#google-vs-numpy).

Note that type hints are intentionally not rendered as some of them become too complex and are better described in prose, following xarray's style.
