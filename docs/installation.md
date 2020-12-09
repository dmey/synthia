# Installation

## Required dependencies

- Python (3.8 or later)
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/scipylib/index.html)
- [xarray](http://xarray.pydata.org/)

## Optional dependencies

- [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib): For vine copula and quasirandom number support.
- [pytest](https://pytest.org): For running the test suite

## Instructions

```
pip install synthia
```

or with optional dependencies

```
pip install synthia[full]
```

After installation, you can run the test suite using

```
git clone https://github.com/dmey/synthia.git
cd synthia
python -m pytest -s tests/
```
