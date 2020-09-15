import numpy
import pytest

@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy

@pytest.fixture(autouse=True)
def set_np_seed():
    numpy.random.seed(42)
