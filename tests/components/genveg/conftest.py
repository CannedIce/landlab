import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components.genveg.integrator import GenVeg
from landlab.components.genveg.growth import PlantGrowth
from landlab.components.genveg.allometry import Biomass, Dimensional
from landlab.components.genveg.duration import Annual, Deciduous, Evergreen
from landlab.components.genveg.habit import Habit, Graminoid
from landlab.components.genveg.species import Species
from landlab.components.genveg.dispersal import Clonal, Seed, Random


@pytest.fixture(autouse=True)
def set_random_seed():
    return np.random.default_rng(seed=39)


# Test parameter dictionary
@pytest.fixture
def example_input_params():
    param_dict = {
        "BTS": {
            "col_params": {"prob_colonization": 0.01, "time_to_colonization": 365},
            "disperse_params": {
                "max_dist_runner": 0.4,
                "min_size_dispersal": 0.5,
                "unit_cost_dispersal": 1.2,
                "mass_to_seedling_rate": 0.02,
                "mean_seed_distance": 0.6,
                "max_seed_distance": 10,
                "seed_distribution_shape_param": 1.5,
                "seed_mass": 0.002,
                "seed_efficiency": (300 / 5000),
                "daily_colonization_probability": 0.01,
            },
            "duration_params": {
                "growing_season_end": 305,
                "growing_season_start": 144,
                "max_age": 1000,
                "peak_biomass": 227,
                "reproduction_end": 250,
                "reproduction_start": 180,
                "senescence_start": 273,
                "death_rate": {
                    "leaf": 0.03,
                    "reproductive": 0.02,
                    "root": 0.02,
                    "stem": 0.02,
                },
            },
            "grow_params": {
                "glucose_requirement": {
                    "leaf": 1.463,
                    "reproductive": 1.414,
                    "root": 1.444,
                    "stem": 1.513,
                },
                "translocation_rate": {
                    "leaf": 0.07,
                    "reproductive": 0.07,
                    "root": 0.07,
                    "stem": 0.07,
                },
                "growth_max_biomass": 13.899999999999999,
                "growth_min_biomass": 0.06222222222222222,
                "hypoxic_ratio": {
                    "leaf": 2.18,
                    "reproductive": 1.6,
                    "root": 1.6,
                    "stem": 2.18,
                },
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
                "plant_part_mean": {
                    "leaf": 2.8,
                    "reproductive": 1,
                    "root": 3.1,
                    "stem": 1.5,
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
                "allometry_method": "min-max",
                "biomass_decay_rate": {
                    "dead_leaf": 0.07,
                    "dead_reproductive": 0.07,
                    "dead_root": 0.07,
                    "dead_stem": 0.07,
                },
                "empirical_coeffs": {
                    "basal_dia_coeffs": {
                        "a": -9999,
                        "b": -9999,
                    },
                    "canopy_area_coeffs": {
                        "a": -9999,
                        "b": -9999,
                        "c": -9999,
                    },
                    "height_coeffs": {
                        "a": -9999,
                        "b": -9999,
                    },
                    "root_dia_coeffs": {
                        "a": -9999,
                        "b": -9999,
                    },
                },
                "lai_cr": 2,
                "height": {
                    "max": np.array([0.931]),
                    "mean": np.array([0.8]),
                    "min": np.array([0.16]),
                },
                "n_stems": {
                    "max": 10,
                    "mean": 3,
                    "min": 1,
                },
                "max_plant_density": 18,
                "shoot_sys_width": {
                    "max": np.array([0.3]),
                    "mean": np.array([0.12]),
                    "min": np.array([0.01]),
                },
                "basal_dia": {
                    "max": np.array([0.1]),
                    "mean": np.array([0.07]),
                    "min": np.array([0.005]),
                },
                "root_depth": {
                    "max": np.array([1.45]),
                    "mean": np.array([0.8]),
                    "min": np.array([0.01]),
                },
                "root_sys_width": {
                    "max": np.array([2]),
                    "mean": np.array([0.2]),
                    "min": np.array([0.01]),
                },
                "sp_leaf_area": 0.0074,
            },
            "mortality_params": {
                "coeffs": {
                    "1": [-3.91197300292798, 34.64562783023239],
                    "2": [0.3911973002927978, -6.456278302323929],
                },
                "duration": {"1": 207, "2": 365},
                "mort_variable_name": {
                    "1": "air__max_temperature_C",
                    "2": "air__min_temperature_C",
                },
                "period": {
                    "1": "during growing season",
                    "2": "year-round",
                },
                "predictor": {
                    "1": [-30, 0, 20, 35, 37],
                    "2": [-30, -20, 0, 35, 37],
                },
                "response": {
                    "1": [1, 1, 1, 0.2, 0],
                    "2": [0, 0.2, 1, 1, 1],
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
                "growth_habit": "graminoid",
                "monocot_dicot": "monocot",
                "p_type": "C3",
                "reproductive_modes": ["clonal", "seed", "random"],
                "species": "BTS",
                "storage": "belowground"
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
        ("dead_root_age", float),
        ("dead_leaf_age", float),
        ("dead_stem_age", float),
        ("dead_reproductive_age", float),
        ("shoot_sys_width", float),
        ("basal_dia", float),
        ("root_sys_width", float),
        ("shoot_sys_height", float),
        ("root_sys_depth", float),
        ("total_leaf_area", float),
        ("live_leaf_area", float),
        ("plant_age", float),
        ("n_stems", int),
        ("dispersal", [
            ("pup_x_loc", float),
            ("pup_y_loc", float),
            ("pup_cost", float),
            ("seedling_x_loc", float, (10,)),
            ("seedling_y_loc", float, (10,)),
            ("seedling_reserve", float),
            ("random_x_loc", float, (10,)),
            ("random_y_loc", float, (10,)),
        ]),
        ("item_id", int),
    ]
    plants = np.empty(1, dtype=dtypes)
    plantlist = []
    plant = "Corn"
    pidval = 0
    cell_index = 0
    for i in range(1):
        pidval = i
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
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                (
                    np.nan,
                    np.nan,
                    np.nan,
                    [np.nan],
                    [np.nan],
                    np.nan,
                    [np.nan],
                    [np.nan],
                ),
                i,
            )
        )
    plants = np.array(plantlist, dtype=dtypes)
    plants["root"] = np.array([0.80000000000000004])
    plants["stem"] = np.array([0.29999999999999999])
    plants["leaf"] = np.array([0.50000000000000000])
    plants["reproductive"] = np.array([0.30])
    plants["live_leaf_area"] = 0.022 * plants["leaf"]
    plants["shoot_sys_width"] = np.array(
        [(4 * (plants["live_leaf_area"] / 0.012) / np.pi) ** 0.5]
    )
    plants["shoot_sys_height"] = np.array([0.010000000000000000])
    plants["total_leaf_area"] = plants["live_leaf_area"]
    plants["plant_age"] = np.array([90])
    plants["n_stems"] = np.array([1.0])
    return plants


@pytest.fixture
def example_plant_array(set_random_seed):
    rng = set_random_seed
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
        ("dead_root_age", float),
        ("dead_leaf_age", float),
        ("dead_stem_age", float),
        ("dead_reproductive_age", float),
        ("shoot_sys_width", float),
        ("basal_dia", float),
        ("root_sys_width", float),
        ("shoot_sys_height", float),
        ("root_sys_depth", float),
        ("total_leaf_area", float),
        ("live_leaf_area", float),
        ("plant_age", float),
        ("n_stems", int),
        ("dispersal", [
            ("pup_x_loc", float),
            ("pup_y_loc", float),
            ("pup_cost", float),
            ("seedling_x_loc", float, (10,)),
            ("seedling_y_loc", float, (10,)),
            ("seedling_reserve", float),
            ("rand_x_loc", float, (10,)),
            ("rand_y_loc", float, (10,)),
        ]),
        ("item_id", int),
    ]
    plants = np.empty(100, dtype=dtypes)
    plantlist = []
    plant = "Corn"
    pidval = 0
    cell_index = 0
    for i in range(8):
        pidval = i
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
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                (
                    np.nan,
                    np.nan,
                    np.nan,
                    [np.nan],
                    [np.nan],
                    np.nan,
                    [np.nan],
                    [np.nan],
                ),
                i,
            )
        )
    plants = np.array(plantlist, dtype=dtypes)
    plants["root"] = np.array([0.05, 0.05, 0.05, 0.05, 7.00, 7.00, 7.00, 7.00])
    plants["stem"] = np.array([0.05, 7.00, 0.05, 7.00, 0.05, 7.00, 0.05, 7.00])
    plants["leaf"] = np.array([0.05, 0.05, 7.00, 7.00, 0.05, 0.05, 7.00, 7.00])
    plants["reproductive"] = np.array([3.5, 3.5, 3.5, 3.5, 3.5, 0.05, 0.05, 0.05])
    plants["dead_root"] = plants["root"] * 0.25
    plants["dead_stem"] = plants["stem"] * 0.25
    plants["dead_leaf"] = plants["leaf"] * 0.25
    plants["dead_reproductive"] = plants["reproductive"] * 0.25
    # ran rng.uniform(low=0, high=260, size=4) to gain the four below so i can unit test
    plants["dead_root_age"] = np.array([1094.035397, 267.2368885, 382.4262438, 347.6774074, 1017.381215, 911.6529431, 314.3525663, 982.1220087])
    plants["dead_leaf_age"] = np.array([1329.561664, 1352.338509, 1238.787392, 1433.019734, 1348.797247, 735.8478796, 1177.379645, 234.6844825])
    plants["dead_stem_age"] = np.array([1315.902291, 1453.913069, 688.1424854, 329.2440673, 1362.62031, 457.1701171, 1377.041709, 1364.471028])
    plants["dead_reproductive_age"] = np.array([507.8777074, 1121.713079, 204.4386012, 1073.25249, 660.3439315, 426.4383309, 1261.53064, 1085.549562])
    # plants["dead_root_age"] = rng.uniform(low=0, high=(365 * 2), size=plants.size)
    # plants["dead_leaf_age"] = rng.uniform(low=0, high=290, size=plants.size)
    # plants["dead_stem_age"] = rng.uniform(low=0, high=290, size=plants.size)
    # plants["dead_reproductive_age"] = rng.uniform(low=0, high=365, size=plants.size)
    plants["shoot_sys_width"] = rng.uniform(low=0.1, high=3, size=plants.size)
    plants["basal_dia"] = rng.uniform(low=0.05, high=1, size=plants.size)
    plants["root_sys_width"] = rng.uniform(low=0.1, high=1, size=plants.size)
    plants["shoot_sys_height"] = rng.uniform(low=0.2, high=4, size=plants.size)
    plants["root_sys_depth"] = rng.uniform(low=0.0, high=2, size=plants.size)
    plants["total_leaf_area"] = rng.uniform(low=0.001, high=3, size=plants.size)
    plants["live_leaf_area"] = plants["total_leaf_area"]
    plants["plant_age"] = rng.uniform(low=1 / 365, high=5, size=plants.size)
    plants["n_stems"] = rng.integers(1, 6, size=plants.size)
    plants["dispersal"]["pup_x_loc"][plants["reproductive"] > 0.1] = 0.1
    plants["dispersal"]["pup_y_loc"][plants["reproductive"] > 0.1] = 0.2
    plants["dispersal"]["pup_cost"][plants["reproductive"] > 0.1] = 0.75
    plants["item_id"] = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    return plants


@pytest.fixture
def one_cell_grid():
    # Create grid with one cell containing min max temp, par, location, species
    grid = RasterModelGrid((3, 3), 2, xy_of_reference=(-74.08, 39.79))
    grid.axis_units = ("m", "m")
    maxtemp = np.array([15.53])
    mintemp = np.array([8.62])
    water_content = np.array([0.88])
    NJ_avg_par = np.array([118.11])

    # Initialize with a dummy data sets
    _ = grid.add_field(
        "air__max_temperature_C",
        maxtemp * np.ones(grid.number_of_cells),
        at="cell",
        units="C",
    )
    _ = grid.add_field(
        "air__min_temperature_C",
        mintemp * np.ones(grid.number_of_cells),
        at="cell",
        units="C",
    )
    _ = grid.add_field(
        "radiation__total_par",
        NJ_avg_par * np.ones(grid.number_of_cells),
        at="cell",
        units="W/m^2",
    )
    _ = grid.add_field(
        "vegetation__plant_species",
        np.full(grid.number_of_cells, "Corn"),
        at="cell",
    )
    _ = grid.add_field(
        "soil_water__volume_fraction",
        water_content * np.ones(grid.number_of_cells),
        at="cell",
        units="m**3/m**3",
    )
    return grid


@pytest.fixture
def two_cell_grid():
    # Create grid with one cell containing min max temp, par, location, species
    grid = RasterModelGrid((4, 3), 2, xy_of_reference=(-74.08, 39.79))
    grid.axis_units = ("m", "m")
    maxtemp = np.array([15.53])
    mintemp = np.array([8.62])
    water_content = np.array([0.88])
    NJ_avg_par = np.array([118.11])

    # Initialize with a dummy data sets
    _ = grid.add_field(
        "air__max_temperature_C",
        maxtemp * np.ones(grid.number_of_cells),
        at="cell",
        units="C",
    )
    _ = grid.add_field(
        "air__min_temperature_C",
        mintemp * np.ones(grid.number_of_cells),
        at="cell",
        units="C",
    )
    _ = grid.add_field(
        "radiation__total_par",
        NJ_avg_par * np.ones(grid.number_of_cells),
        at="cell",
        units="W/m^2",
    )
    _ = grid.add_field(
        "vegetation__plant_species",
        np.full(grid.number_of_cells, "Corn"),
        at="cell",
    )
    _ = grid.add_field(
        "soil_water__volume_fraction",
        water_content * np.ones(grid.number_of_cells),
        at="cell",
        units="m**3/m**3",
    )
    return grid


@pytest.fixture
def biomass_minmax(example_input_params):
    params = example_input_params["BTS"]
    params["grow_params"]["abg_biomass"] = {}
    params["morph_params"]["canopy_area"] = {}
    params["grow_params"]["abg_biomass"]["min"] = params["grow_params"]["plant_part_min"]["leaf"] + params["grow_params"]["plant_part_min"]["stem"]
    params["grow_params"]["abg_biomass"]["max"] = params["grow_params"]["plant_part_max"]["leaf"] + params["grow_params"]["plant_part_max"]["stem"]
    params["morph_params"]["canopy_area"]["min"] = params["morph_params"]["shoot_sys_width"]["min"]**2 * 0.25 * np.pi
    params["morph_params"]["canopy_area"]["max"] = params["morph_params"]["shoot_sys_width"]["max"]**2 * 0.25 * np.pi
    return Biomass(params)


@pytest.fixture
def dimensional_default(example_input_params):
    params = example_input_params["BTS"]
    params["morph_params"]["allometry_method"] = "default"
    params["grow_params"]["abg_biomass"] = {}
    params["morph_params"]["canopy_area"] = {}
    params["grow_params"]["abg_biomass"]["min"] = params["grow_params"]["plant_part_min"]["leaf"] + params["grow_params"]["plant_part_min"]["stem"]
    params["grow_params"]["abg_biomass"]["max"] = params["grow_params"]["plant_part_max"]["leaf"] + params["grow_params"]["plant_part_max"]["stem"]
    params["morph_params"]["canopy_area"]["min"] = params["morph_params"]["shoot_sys_width"]["min"]**2 * 0.25 * np.pi
    params["morph_params"]["canopy_area"]["max"] = params["morph_params"]["shoot_sys_width"]["max"]**2 * 0.25 * np.pi
    empirical_coeffs = {
        "basal_dia_coeffs": {"a": 0.5093, "b": 0.47},
        "height_coeffs": {"a": np.log(0.232995), "b": 0.619077},
        "canopy_area_coeffs": {"a": np.log(0.23702483 * 0.2329925**0.9459644), "b": 0.72682 + (0.619077 * 0.9459644)},
    }
    yield Dimensional(params, empirical_coeffs)


@pytest.fixture
def dimensional_user(example_input_params):
    params = example_input_params["BTS"]
    params["morph_params"]["allometry_method"] = "user-defined"
    params["grow_params"]["abg_biomass"] = {}
    params["morph_params"]["canopy_area"] = {}
    params["grow_params"]["abg_biomass"]["min"] = params["grow_params"]["plant_part_min"]["leaf"] + params["grow_params"]["plant_part_min"]["stem"]
    params["grow_params"]["abg_biomass"]["max"] = params["grow_params"]["plant_part_max"]["leaf"] + params["grow_params"]["plant_part_max"]["stem"]
    params["morph_params"]["canopy_area"]["min"] = params["morph_params"]["shoot_sys_width"]["min"]**2 * 0.25 * np.pi
    params["morph_params"]["canopy_area"]["max"] = params["morph_params"]["shoot_sys_width"]["max"]**2 * 0.25 * np.pi
    empirical_coeffs = {
        "basal_dia_coeffs": {"a": 0.5093, "b": 0.47},
        "height_coeffs": {"a": np.log(0.232995), "b": 0.619077},
        "canopy_area_coeffs": {"a": np.log(0.23702483 * 0.2329925**0.9459644), "b": 0.72682 + (0.619077 * 0.9459644)},
    }
    yield Dimensional(params, empirical_coeffs)


@pytest.fixture
def habit_object(example_input_params):
    params = example_input_params["BTS"]
    params["grow_params"]["abg_biomass"] = {}
    params["morph_params"]["canopy_area"] = {}
    params["grow_params"]["abg_biomass"]["min"] = params["grow_params"]["plant_part_min"]["leaf"] + params["grow_params"]["plant_part_min"]["stem"]
    params["grow_params"]["abg_biomass"]["max"] = params["grow_params"]["plant_part_max"]["leaf"] + params["grow_params"]["plant_part_max"]["stem"]
    params["morph_params"]["canopy_area"]["min"] = params["morph_params"]["shoot_sys_width"]["min"]**2 * 0.25 * np.pi
    params["morph_params"]["canopy_area"]["max"] = params["morph_params"]["shoot_sys_width"]["max"]**2 * 0.25 * np.pi
    allometry = Biomass(params, empirical_coeffs={"root_dia_coeffs": {"a": 0.08, "b": 0.24}})
    yield Habit(params, allometry, dt=1, green_parts=("leaf", "stem"))


@pytest.fixture
def graminoid_object(example_input_params):
    params = example_input_params["BTS"]
    params["grow_params"]["abg_biomass"] = {}
    params["morph_params"]["canopy_area"] = {}
    params["grow_params"]["abg_biomass"]["min"] = params["grow_params"]["plant_part_min"]["leaf"] + params["grow_params"]["plant_part_min"]["stem"]
    params["grow_params"]["abg_biomass"]["max"] = params["grow_params"]["plant_part_max"]["leaf"] + params["grow_params"]["plant_part_max"]["stem"]
    params["morph_params"]["canopy_area"]["min"] = params["morph_params"]["shoot_sys_width"]["min"]**2 * 0.25 * np.pi
    params["morph_params"]["canopy_area"]["max"] = params["morph_params"]["shoot_sys_width"]["max"]**2 * 0.25 * np.pi
    yield Graminoid(params, dt=1)


@pytest.fixture
def species_object(example_input_params, one_cell_grid):
    dt = np.timedelta64(1, 'D')
    yield Species(example_input_params["BTS"], one_cell_grid, dt=dt)


@pytest.fixture
def clonal(example_input_params):
    example_input_params["BTS"]["grow_params"]["growth_biomass"] = {}
    example_input_params["BTS"]["grow_params"]["growth_biomass"]["min"] = (
        example_input_params["BTS"]["grow_params"]["plant_part_min"]["root"]
        + example_input_params["BTS"]["grow_params"]["plant_part_min"]["leaf"]
        + example_input_params["BTS"]["grow_params"]["plant_part_min"]["stem"]
    )
    yield Clonal(example_input_params["BTS"])


@pytest.fixture
def random(example_input_params, one_cell_grid):
    example_input_params["BTS"]["grow_params"]["growth_biomass"] = {}
    example_input_params["BTS"]["grow_params"]["growth_biomass"]["min"] = (
        example_input_params["BTS"]["grow_params"]["plant_part_min"]["root"]
        + example_input_params["BTS"]["grow_params"]["plant_part_min"]["leaf"]
        + example_input_params["BTS"]["grow_params"]["plant_part_min"]["stem"]
    )
    cell_area = one_cell_grid.area_of_cell
    extent = one_cell_grid.extent
    reference = one_cell_grid.xy_of_reference
    yield Random(example_input_params["BTS"], cell_area, extent, reference)


@pytest.fixture
def seed(example_input_params, one_cell_grid):
    example_input_params["BTS"]["grow_params"]["growth_biomass"] = {}
    example_input_params["BTS"]["grow_params"]["growth_biomass"]["min"] = (
        example_input_params["BTS"]["grow_params"]["plant_part_min"]["root"]
        + example_input_params["BTS"]["grow_params"]["plant_part_min"]["leaf"]
        + example_input_params["BTS"]["grow_params"]["plant_part_min"]["stem"]
    )
    dt = np.timedelta64(1, 'D')
    yield Seed(example_input_params["BTS"], dt)


@pytest.fixture
def growth_obj(one_cell_grid, example_input_params, example_plant_array):
    dt = np.timedelta64(1, 'D')
    jday = 195
    rel_time = 194
    yield PlantGrowth(
        one_cell_grid,
        dt,
        rel_time,
        jday,
        species_params=example_input_params["BTS"],
        plant_array=example_plant_array,
    )


@pytest.fixture
def genveg_obj(one_cell_grid, example_input_params, example_plant_array):
    dt = np.timedelta64(1, 'D')
    current_day = np.datetime64("2019-07-14", "D")
    yield GenVeg(
        one_cell_grid,
        dt,
        current_day,
        example_input_params,
        plant_array=example_plant_array
    )
