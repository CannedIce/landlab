import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from unittest import mock


def test__init_methods_run_with_array(growth_obj, example_plant_array):
    # Test to make sure this gets init plant array or loads
    # species cover properly
    # Init: creates plants, initialize plants, calculates n_plants,
    # creates data record of plants
    init_array = growth_obj.plants
    col_names = example_plant_array.dtype.names
    special_handling = ["species", "dispersal", "grid_element", "vegetation__species"]
    for name in col_names:
        if name not in special_handling:
            assert_allclose(init_array[name][~init_array[name].mask], example_plant_array[name])
        elif name == "species":
            assert np.all(init_array[name]) == np.all(example_plant_array[name])
        else:
            dispersal_subs = [
                "pup_x_loc",
                "pup_y_loc",
                "pup_cost",
                "seedling_x_loc",
                "seedling_y_loc",
                "seedling_reserve",
                "rand_x_loc",
                "rand_y_loc"
            ]
            for sub_name in dispersal_subs:
                assert_allclose(init_array["dispersal"][sub_name][~init_array["dispersal"][sub_name].mask], np.ravel(example_plant_array["dispersal"][sub_name]))
    pd_out = growth_obj.record_plants.dataset.to_dataframe()
    col_map = {
        "species": "vegetation__species",
        "root": "vegetation__root_biomass",
        "leaf": "vegetation__leaf_biomass",
        "stem": "vegetation__stem_biomass",
        "reproductive": "vegetation__repro_biomass",
        "dead_root": "vegetation__dead_root_biomass",
        "dead_leaf": "vegetation__dead_leaf_biomass",
        "dead_stem": "vegetation__dead_stem_biomass",
        "dead_reproductive": "vegetation__dead_repro_biomass",
        "total_leaf_area": "vegetation__total_leaf_area",
        "live_leaf_area": "vegetation__live_leaf_area",
        "shoot_sys_width": "vegetation__shoot_sys_diameter",
        "basal_dia": "vegetation__basal_diameter",
        "shoot_sys_height": "vegetation__plant_height",
        "root_sys_width": "vegetation__root_sys_diameter",
        "root_sys_depth": "vegetation__root_depth",
        "plant_age": "vegetation__plant_age",
        "n_stems": "vegetation__stem_count"
    }
    for array_col, pd_col in col_map.items():
        if pd_col not in special_handling:
            assert_allclose(pd_out[pd_col].to_numpy(), example_plant_array[array_col])
    assert growth_obj.n_plants == 8


def test__init_plants_from_grid(cover_growth_obj):
    # Test to make sure this gets init plant array or loads
    # species cover properly
    # Init: creates plants, initialize plants, calculates n_plants,
    # creates data record of plants
    array = cover_growth_obj._init_plants_from_grid(True, np.ones_like(cover_growth_obj._grid.at_cell["vegetation__cover_fraction"]))
    basal_dias = array["basal_dia"]
    occupied_area = np.sum(basal_dias**2 * np.pi / 4)
    area_left = cover_growth_obj._grid.area_of_cell[0] * 0.907 - occupied_area
    assert area_left < (1.2 * cover_growth_obj.species_morph_params["basal_dia"]["min"])


def test_species_plants(growth_obj, example_plant_array):
    # Test to make sure this returns the plant array
    plant_out = growth_obj.species_plants()
    col_names = example_plant_array.dtype.names
    special_handling = ["species", "dispersal"]
    for name in col_names:
        if name not in special_handling:
            assert_allclose(plant_out[name], example_plant_array[name])
        elif name == "species":
            assert np.all(plant_out[name]) == np.all(example_plant_array[name])
        else:
            dispersal_subs = [
                "pup_x_loc",
                "pup_y_loc",
                "pup_cost",
                "seedling_x_loc",
                "seedling_y_loc",
                "seedling_reserve",
                "rand_x_loc",
                "rand_y_loc"
            ]
            for sub_name in dispersal_subs:
                assert_allclose(plant_out["dispersal"][sub_name], example_plant_array["dispersal"][sub_name])


def test_update_plants(growth_obj, example_plant_array):
    # Test to make sure this updates plants for array & subarray
    # variables for list of specific plant pids
    var_names = ["root", "leaf"]
    pids = [0, 1]
    var_vals = [[27.2, 27.2], [13.1, 2.3]]
    plants_out = growth_obj.update_plants(var_names, pids, var_vals)
    assert_allclose(plants_out["root"][0:2], var_vals[0])
    assert_allclose(plants_out["leaf"][0:2], var_vals[1])
    var_names = ["pup_x_loc", "pup_y_loc", "seedling_x_loc"]
    var_vals = [[np.nan, np.nan], [np.nan, np.nan], [[np.nan] * 10, [np.nan] * 10]]
    plants_out = growth_obj.update_plants(var_names, pids, var_vals, subarray="dispersal")
    assert_allclose(plants_out["dispersal"]["pup_x_loc"][0:2], var_vals[0])
    assert_allclose(plants_out["dispersal"]["pup_y_loc"][0:2], var_vals[1])
    assert_allclose(plants_out["dispersal"]["seedling_x_loc"][0:2], var_vals[2])


def test_add_new_plants(growth_obj, example_plant_array, example_plant):
    # Add a list of new plants starting at pid+1 and updating mask
    # Should update array and record
    rel_time = 28
    array_out = growth_obj.add_new_plants(example_plant, rel_time)
    n_plants = array_out["root"].count()
    assert n_plants == (example_plant_array.size + 1)
    record = growth_obj.record_plants.number_of_items
    assert record == (example_plant_array.size + 1)


def test_grow(growth_obj, example_plant, one_cell_grid):
    # Save this for last - consider mocking up results for singular days under
    # different logical circumstances
    pass


def test_set_event_flags(growth_obj):
    # Test jdays to make sure all events are flagged properly
    jdays_to_test = [97, 144, 150, 180, 227, 250, 273, 280, 305, 325]
    assert_dict = {
        "_in_growing_season": False,
        "_is_emergence_day": False,
        "_in_reproductive_period": False,
        "_in_senescence_period": False,
        "_is_dormant_day": False,
    }
    flags_to_change = [
        [],
        ["_is_emergence_day"],
        ["_in_growing_season"],
        ["_in_growing_season", "_in_reproductive_period"],
        ["_in_growing_season", "_in_reproductive_period"],
        ["_in_growing_season"],
        ["_in_growing_season", "_in_senescence_period"],
        ["_in_growing_season", "_in_senescence_period"],
        ["_is_dormant_day"],
        []
    ]
    for day_flag in zip(jdays_to_test, flags_to_change):
        today_dict = assert_dict.copy()
        jday = day_flag[0]
        for key in day_flag[1]:
            today_dict[key] = True
        flags = growth_obj.set_event_flags(jday)
        assert flags == today_dict
    # Test for nan values
    evergreen_vals = {
        "growing_season_start": np.nan,
        "growing_season_end": np.nan,
        "senescence_start": np.nan
    }
    evergreen_obj = growth_obj
    evergreen_obj.species_duration_params.update(evergreen_vals)
    jday = jdays_to_test[0]
    flags = evergreen_obj.set_event_flags(jday)
    assert flags["_in_growing_season"] is True


def test_kill_small_plants(growth_obj, example_plant_array):
    # Kill live plants with biomass < minimum and move
    # any remaining biomass to dead
    gro_obj = growth_obj
    plants = example_plant_array
    plants["root"][0] = 0.01
    plants["leaf"][0] = 0.01
    plants["stem"][0] = 0.01
    min_biomass = gro_obj.species_grow_params["plant_part_min"]
    gro_obj.species_grow_params["min_growth_biomass"] = min_biomass["root"] + min_biomass["stem"] + min_biomass["leaf"]
    plants_out = gro_obj.kill_small_plants(plants)
    parts = ["root", "leaf", "stem"]
    for part in parts:
        assert plants_out[part][0] == 0.0
        assert_allclose(plants_out[part][1:], plants[part][1:])


def test_remove_plants(growth_obj):
    # Remove plants from plant array that have dead biomass too small
    # to track and save final point in save record
    gro_obj = growth_obj
    gro_obj.plants["root"][0] = 0.01
    gro_obj.plants["leaf"][0] = 0.01
    gro_obj.plants["stem"][0] = 0.01
    gro_obj.plants["dead_root"][0] = 0.01
    gro_obj.plants["dead_leaf"][0] = 0.01
    gro_obj.plants["dead_stem"][0] = 0.01
    gro_obj.plants["dead_reproductive"][0] = 0.01
    min_biomass = gro_obj.species_grow_params["plant_part_min"]
    gro_obj.species_grow_params["min_growth_biomass"] = min_biomass["root"] + min_biomass["stem"] + min_biomass["leaf"]
    plants_out, n_plants = gro_obj.remove_plants()
    assert n_plants == 7
    assert plants_out.mask[0]["pid"]


def test_save_plant_output(growth_obj, example_plant_array):
    # Force save of plant output to record
    # Make sure this works if we add new plants
    rel_time = 200
    gro_obj = growth_obj
    gro_obj.save_plant_output(
        rel_time, [
            "dead_root",
            "dead_leaf",
            "dead_stem",
            "dead_reproductive",
            "total_leaf_area",
            "live_leaf_area",
            "shoot_sys_width",
            "basal_dia",
            "shoot_sys_height",
            "root_sys_width",
            "root_sys_depth",
            "plant_age",
            "n_stems",
        ]
    )
    pd_out = gro_obj.record_plants.dataset.to_dataframe()
    assert pd_out.index[-1][0] == rel_time
    columns = list(pd_out.columns)
    for col_name in columns:
        for plant in range(8):
            assert pd_out.loc[194, plant][col_name] == pd_out.loc[200, plant][col_name]
