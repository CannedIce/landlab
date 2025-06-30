import numpy as np
import pytest
from numpy.testing import assert_allclose
from landlab.components.genveg.dispersal import Clonal, Seed, Random


def test_clonal_disperse(clonal, example_plant):
    # Test distance from 0, 0 is equal to pup_cost / unit_cost
    x_loc = example_plant["x_loc"]
    y_loc = example_plant["y_loc"]
    plant_pup = clonal.disperse(example_plant)
    distance = np.sqrt((plant_pup["dispersal"]["pup_x_loc"] - x_loc)**2 + (plant_pup["dispersal"]["pup_y_loc"] - y_loc)**2)
    est_distance = plant_pup["dispersal"]["pup_cost"] / clonal.unit_cost
    assert_allclose(distance, est_distance, rtol=0.0001)
    # Test pup_cost is less than reproductive


def test_seed_disperse(seed, one_cell_grid):
    pass


def test_random_disperse(random, one_cell_grid):
    pass
