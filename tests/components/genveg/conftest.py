import pytest
import numpy as np


# Test parameter dictionary
@pytest.fixture
def example_input_params():
    param_dict = {
        "BTS": {
            "col_params": {"prob_colonization": 0.01, "time_to_colonization": 365},
            "dispersal_params": {
                "max_dist_dispersal": 0.4,
                "min_size_dispersal": 0.5,
                "unit_cost_dispersal": 1.2,
            },
            "duration_params": {
                "growing_season_end": 305,
                "growing_season_start": 144,
                "max_age": 1000,
                "peak_biomass": 227,
                "reproduction_end": 250,
                "reproduction_start": 180,
                "senescence_start": 273,
            },
            "grow_params": {
                "glucose_requirement": {
                    "leaf": 1.463,
                    "reproductive": 1.414,
                    "root": 1.444,
                    "stem": 1.513,
                },
                "growth_max_biomass": 13.899999999999999,
                "growth_min_biomass": 0.06222222222222222,
                "incremental_nsc": {
                    "leaf": [1.25, 0, -1, 0.5],
                    "reproductive": [1.5625, -1.875, 0.0625, 2.5],
                    "root": [1.25, -2.5, 0, 2],
                    "stem": [0, -0.5, 0, 0.5],
                },
                "max_nsc_content": {
                    "leaf": 0.36629,
                    "reproductive": 0.36643,
                    "root": 0.36643,
                    "stem": 0.30964,
                },
                "min_nsc_content": {
                    "leaf": 0.01548,
                    "reproductive": 0.01071,
                    "root": 0.01071,
                    "stem": 0.0075,
                },
                "nsc_content": {
                    "leaf": 0.2096344,
                    "reproductive": 0.2369178,
                    "root": 0.2369178,
                    "stem": 0.10,
                },
                "plant_part_max": {
                    "leaf": 5.5,
                    "reproductive": 4,
                    "root": 4.3,
                    "stem": 4.1,
                },
                "plant_part_min": {
                    "leaf": 0.03,
                    "reproductive": 0,
                    "root": 0.01,
                    "stem": 0.022222222222222223,
                },
                "respiration_coefficient": {
                    "leaf": 0.03,
                    "reproductive": 0.01,
                    "root": 0.015,
                    "stem": 0.015,
                },
                "root_to_leaf": {"a": 0.031, "b1": 0.951, "b2": 0},
                "root_to_stem": {"a": -0.107, "b1": 1.098, "b2": 0.0216},
                "total_max_biomass": 17.9,
                "total_min_biomass": 0.06222222222222222,
            },
            "morph_params": {
                "biomass_decay_rate": 0.07,
                "lai_cr": 4,
                "max_height": 0.75,
                "max_n_stems": 10,
                "max_plant_density": 18,
                "max_root_sys_depth": 0.33,
                "max_root_sys_width": 0.35,
                "max_shoot_sys_width": 0.3,
                "min_height": 0.075,
                "min_root_sys_depth": 0.02,
                "min_root_sys_width": 0.01,
                "min_shoot_sys_width": 0.005423267786434878,
                "sp_leaf_area": 0.0074,
            },
            "mortality_params": {
                "coeffs": {
                    "1": [9837.624573797346, 0.20082011968400607],
                    "2": [1270116309.000265, 33.52512742454386],
                },
                "duration": {1: 207, 2: 365},
                "mort_variable_name": {
                    "1": "Distance to shore",
                    "2": "elevation__above_WL",
                },
                "period": {
                    "1": "during growing  season",
                    "2": "during growing season",
                },
                "predictor": {
                    "1": [30, 40, 50, 55, -9999],
                    "2": [0, 0.5, 0.6, 0.68, 1.12],
                },
                "response": {
                    "1": [0.01, 0.25, 0.67, 0.9, -9999],
                    "2": [0, 0, 0.3, 1, 1],
                },
            },
            "photo_params": {
                "ci": 65,
                "co": 209,
                "kc": 21,
                "ko": 650,
                "spec_factor_25": 120,
                "stomatal_conductance": 0.2,
                "vcmax": 20,
            },
            "plant_factors": {
                "angio_gymno": "angiosperm",
                "duration": "perennial deciduous",
                "growth_form": "rhizomatous",
                "growth_habit": "graminoid",
                "monocot_dicot": "monocot",
                "p_type": "C3",
                "shape": "erect",
                "species": "BTS",
            },
        }
    }
    return param_dict


# test plant array
@pytest.fixture
def example_plant():
    dtypes = [
        ("species", "U10"),
        ("pid", int),
        ("cell_index", int),
        ("x_loc", float),
        ("y_loc", float),
        (("root", "root_biomass"), float),
        (("leaf", "leaf_biomass"), float),
        (("stem", "stem_biomass"), float),
        (("reproductive", "repro_biomass"), float),
        ("dead_root", float),
        ("dead_stem", float),
        ("dead_leaf", float),
        ("dead_reproductive", float),
        ("dead_age", float),
        ("shoot_sys_width", float),
        ("root_sys_width", float),
        ("shoot_sys_height", float),
        ("root_sys_depth", float),
        ("total_leaf_area", float),
        ("live_leaf_area", float),
        ("plant_age", float),
        ("n_stems", int),
        ("pup_x_loc", float),
        ("pup_y_loc", float),
        ("pup_cost", float),
        ("item_id", int),
    ]
    plants = np.empty((0, 26), dtype=dtypes)
    plantlist = []
    plant = "Corn"
    pidval = 0
    cell_index = 0
    for i in range(1):
        pidval = i
        cell_index = i + 1
        plantlist.append(
            (
                plant,
                pidval,
                cell_index,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                np.nan,
                np.nan,
                np.nan,
                i,
            )
        )
    plants = np.array(plantlist, dtype=dtypes)
    plants["root"] = np.array([0.80000000000000004])
    plants["stem"] = np.array([0.29999999999999999])
    plants["leaf"] = np.array([0.50000000000000000])
    plants["live_leaf_area"] = 0.022 * plants["leaf"]
    plants["shoot_sys_width"] = np.array(
        [(4 * (plants["live_leaf_area"] / 0.012) / np.pi) ** 0.5]
    )
    plants["shoot_sys_height"] = np.array([0.010000000000000000])
    plants["total_leaf_area"] = plants["live_leaf_area"]
    plants["plant_age"] = np.array([90])
    plants["n_stems"] = np.array([1.0])
    return plants