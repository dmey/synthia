import os
from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='synthia',
    version='1.1.0',
    description='Multidimensional synthetic data generation in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dmey/synthia',
    license="MIT",
    author='D. Meyer, T. Nagler',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    python_requires=">=3.8", # when changing this, also change ci.yml
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "bottleneck", # required by xarray.DataArray.rank
    ],
    extras_require = {
        "full":  ["pyvinecopulib==0.5.5"]
    }
)
