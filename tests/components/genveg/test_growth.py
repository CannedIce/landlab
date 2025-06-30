import numpy as np
from numpy.testing import assert_allclose


def test__init__methods_run(growth_obj, example_plant):
    pass


def test_species_plants(growth_obj, example_plant):
    # Test to make sure this gets plant array
    pass


def test_species_get_variables(growth_obj, example_input_params):
    pass


def test_update_plants(growth_obj, example_plant_array):
    pass


def test_add_new_plants(growth_obj, example_plant_array, example_plant):
    pass


def test_grow(growth_obj, example_plant, one_cell_grid):
    pass


def test__init_plants_from_grid(growth_obj, one_cell_grid):
    pass


def test_set_event_flags(growth_obj):
    pass


def test_kill_small_plants(growth_obj, example_plant):
    pass


def test_remove_plants(growth_obj, example_plant_array):
    pass


def test_save_plant_output(growth_obj):
    pass
