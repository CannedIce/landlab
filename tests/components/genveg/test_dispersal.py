import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
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
    assert (plant_pup["dispersal"]["pup_cost"] <= example_plant["reproductive"]) or np.isnan(plant_pup["dispersal"]["pup_cost"])

def test_seed_disperse(seed, one_cell_grid):
    pass


def test_random_disperse(random, one_cell_grid, example_plant_array):
    # Test we have 
    extent = one_cell_grid.extent
    reference = one_cell_grid.xy_of_reference
    random.daily_col_prob = 0.5
    update_plant_array = random.disperse(example_plant_array)
    filter = ~np.isnan(update_plant_array["dispersal"]["rand_x_loc"])
    x_upper_bound = reference[1] + extent[1]
    y_upper_bound = reference[0] + extent[0]
    assert_array_less(update_plant_array["dispersal"]["rand_x_loc"][filter], x_upper_bound)
    assert_array_less(update_plant_array["dispersal"]["rand_y_loc"][filter], y_upper_bound)

