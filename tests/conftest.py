import numpy as np
import pytest


@pytest.fixture(scope="session")
def photo():
    """A real photo shipped with matplotlib (600 x 512)."""
    from matplotlib import cbook
    from matplotlib.image import imread
    img = imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False))
    return np.ascontiguousarray(img[:, :, :3])


@pytest.fixture(scope="session")
def small(photo):
    return np.ascontiguousarray(photo[::3, ::3])            # 200 x 171


@pytest.fixture
def rng():
    return np.random.default_rng(0)
