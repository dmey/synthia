import os
from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='synthia',
    version='0.1.1',
    description='Multi-dimensional synthetic data generation in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://dmey.github.io/synthia',
    license="MIT",
    author='D. Meyer, T. Nagler',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    python_requires=">=3.8", # when changing this, also change environment-test.yml
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "bottleneck", # required by xarray.DataArray.rank
    ],
    extras_require = {
        "full":  ["pyvinecopulib"]
    }
)
