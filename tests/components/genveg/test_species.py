import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_less

from landlab.components.genveg.form import Bunch
from landlab.components.genveg.form import Colonizing
from landlab.components.genveg.form import Multiplestems
from landlab.components.genveg.form import Rhizomatous
from landlab.components.genveg.form import Singlecrown
from landlab.components.genveg.form import Singlestem
from landlab.components.genveg.form import Stoloniferous
from landlab.components.genveg.form import Thicketforming
from landlab.components.genveg.habit import Forbherb
from landlab.components.genveg.habit import Graminoid
from landlab.components.genveg.habit import Shrub
from landlab.components.genveg.habit import Tree
from landlab.components.genveg.habit import Vine
from landlab.components.genveg.photosynthesis import C3
from landlab.components.genveg.photosynthesis import C4
from landlab.components.genveg.photosynthesis import Cam
from landlab.components.genveg.species import Species

dt = np.timedelta64(1, 'D')


def create_species_object(example_input_params, dt=dt):
    return Species(example_input_params["BTS"], dt=dt, latitude=0.9074)


def test_get_daily_nsc_concentration(example_input_params):
    species_object = create_species_object(example_input_params)
    days = example_input_params["BTS"]["duration_params"]
    parts = ["root", "leaf", "reproductive", "stem"]
    nsc_content_gs_start_actual = {
        "root": 276.9606782 / 1000,
        "leaf": 247.3937921 / 1000,
        "stem": 100 / 1000,
        "reproductive": 287.459626 / 1000,
    }
    nsc_content_spring195_actual = {
        "root": 205.575914 / 1000,
        "leaf": 223.822083 / 1000,
        "stem": 93.9498113 / 1000,
        "reproductive": 220.297893 / 1000,
    }
    gs_start_day = days["growing_season_start"]
    nsc_content_gs_start = species_object.get_daily_nsc_concentration(gs_start_day)
    nsc_content_195 = species_object.get_daily_nsc_concentration(195)
    for part in parts:
        assert_allclose(
            nsc_content_gs_start_actual[part], nsc_content_gs_start[part], rtol=0.0001
        )
        assert_allclose(
            nsc_content_spring195_actual[part], nsc_content_195[part], rtol=0.0001
        )


def test_calc_area_of_circle(example_input_params):
    species_object = create_species_object(example_input_params)
    morph_params = example_input_params["BTS"]["morph_params"]
    m_params = [
        "shoot_sys_width",
        "root_sys_width",
    ]
    vals = [
        "max",
        "min"
    ]
    # values from excel sheet
    area_values = np.array([[0.070685835, 0.0000785398], [3.141592654, 0.0000785398]])
    idx = 0
    for m_param in m_params:
        for val, a_value in zip(vals, area_values[idx]):
            assert_almost_equal(
                species_object.calc_area_of_circle(morph_params[m_param][val]), a_value
            )
        idx += 1


def test_calculate_dead_age(example_input_params):
    age_t1 = np.array([60, 60, 60])
    mass_t1 = np.array([2, 2, 2])
    mass_t2 = np.array([0, 3, 4])
    age_t2 = np.array([60, 40, 30])
    calc_age_t2 = create_species_object(example_input_params).calculate_dead_age(age_t1, mass_t1, mass_t2)
    assert_almost_equal(age_t2, calc_age_t2, decimal=6)


def test_calculate_shaded_leaf_mortality(example_plant_array, example_plant, example_input_params):
    species_object = create_species_object(example_input_params)
    # Check to make sure no leaf mortality occurred for LAI 0.012 < LAI_crit 2
    init_leaf_weight = example_plant_array["leaf"].copy()
    print(example_plant_array["total_leaf_area"])
    print(example_plant_array["total_leaf_area"] / (np.pi / 4 * example_plant_array["shoot_sys_width"]**2))
    plant_out = species_object.calculate_shaded_leaf_mortality(example_plant_array)
    assert_almost_equal(example_plant_array["leaf"][0:2], init_leaf_weight[0:2])
    assert_array_less(plant_out["leaf"][2], init_leaf_weight[2])
    # Change leaf area to make LAI greater than LAI_crit and reduce leaf biomass
    wofost_leaf_weight_change = np.array([0.00259091])
    big_leaf_plant = example_plant
    big_leaf_plant["total_leaf_area"] = 2.15
    big_leaf_plant_leaf_init = big_leaf_plant["leaf"].copy()
    plant_out = species_object.calculate_shaded_leaf_mortality(big_leaf_plant)
    weight_change = np.subtract(big_leaf_plant_leaf_init, plant_out["leaf"])
    assert_almost_equal(weight_change, wofost_leaf_weight_change, 6)


def test_calculate_whole_plant_mortality(example_plant_array, one_cell_grid, example_input_params):
    max_temp_dt = np.timedelta64(example_input_params["BTS"]["mortality_params"]["duration"]["1"], 'D')
    min_temp_dt = np.timedelta64(example_input_params["BTS"]["mortality_params"]["duration"]["1"], 'D')
    species_object_max = create_species_object(example_input_params, dt=max_temp_dt)
    species_object_min = create_species_object(example_input_params, dt=min_temp_dt)
    max_temp = one_cell_grid["cell"]["air__max_temperature_C"][example_plant_array["cell_index"]]
    min_temp = one_cell_grid["cell"]["air__min_temperature_C"][example_plant_array["cell_index"]]
    # Make sure plant doesn't die at moderate ambient temp
    new_biomass = species_object_max.calculate_whole_plant_mortality(example_plant_array, max_temp, "1")
    assert_allclose(new_biomass["root"], example_plant_array["root"], rtol=0.0001)
    # Change max temp and make sure plant dies
    one_cell_grid["cell"]["air__max_temperature_C"] = 38 * np.ones(one_cell_grid.number_of_cells)
    max_temp = one_cell_grid["cell"]["air__max_temperature_C"]
    new_biomass = species_object_max.calculate_whole_plant_mortality(example_plant_array, max_temp, "1")
    assert_allclose(new_biomass["root"], np.zeros_like(new_biomass["root"]), rtol=0.0001)
    # Change min temp and make sure plant dies
    one_cell_grid["cell"]["air__min_temperature_C"] = -38 * np.ones(one_cell_grid.number_of_cells)
    min_temp = one_cell_grid["cell"]["air__min_temperature_C"]
    new_biomass = species_object_min.calculate_whole_plant_mortality(example_plant_array, min_temp, "2")
    assert_allclose(new_biomass["root"], np.zeros_like(new_biomass["root"]), rtol=0.0001)


def test_update_morphology(example_input_params, example_plant_array):
    # This method should update basal diameter, shoot width, shoot height, root
    # width, live leaf area, dead leaf area
    example_input_params["BTS"]["morph_params"]["allometry_method"] = "default"
    sp_obj = create_species_object(example_input_params)
    # change to default
    pred_example_plant_array = sp_obj.update_morphology(example_plant_array)
    known_plants = {}
    known_plants["basal_dia"] = np.array([0.00002522, 0.2157987, 0.2157987, 0.9288838, 0.00002522, 0.2157987, 0.2157987, 0.9288838])
    known_plants["shoot_sys_width"] = np.array([0, 2.0703735, 2.0703735, 5.3955576, 0, 2.0703735, 2.0703735, 5.3955576,])
    known_plants["shoot_sys_height"] = np.array([0, 1.560362599, 1.560362599, 3.851822257, 0, 1.560362599, 1.560362599, 3.851822257])
    known_plants["root_sys_width"] = np.array([0.08, 0.548613798, 0.548613798, 2, 0.08, 0.548613798, 0.548613798, 2])
    known_plants["live_leaf_area"] = np.array([0.00037, 0.00037, 0.05180, 0.05180, 0.00037, 0.00037, 0.05180, 0.05180])
    update_vars = [
        "basal_dia",
        "shoot_sys_width",
        "shoot_sys_height",
        "root_sys_width",
        "live_leaf_area",
    ]
    for var in update_vars:
        print(var)
        assert_allclose(known_plants[var], pred_example_plant_array[var], rtol=0.001)


def test_populate_biomass_allocation_array(example_input_params):
    s = create_species_object(example_input_params)
    biomass_array_from_excel = np.array([
        [0.01, 0.11,0.21,0.31,0.41,0.51,0.61,0.71,0.81,0.91,1.01,1.11,1.21,1.31,1.41,1.51,1.61,1.71,1.81,1.91,2.01,2.11,2.21,2.31,2.41,2.51,2.61,2.71,2.81,2.91,3.01,3.11,3.21,3.31,3.41,3.51,3.61,3.71,3.81,3.91,4.01,4.11,4.21,4.31],
        [0.029531558, 0.314126222,0.597580079,0.881430388,1.165852132,1.450873064,1.736484358,2.022666426,2.309396945,2.596653692,2.884415563,3.1726629,3.461377538,3.750542729,4.040143031,4.330164189,4.620593014,4.911417273,5.202625596,5.494207382,5.786152728,6.078452358,6.371097567,6.664080162,6.957392425,7.251027061,7.544977173,7.839236221,8.133797995,8.428656593,8.723806393,9.019242035,9.314958401,9.610950598,9.907213944,10.20374395,10.50053631,10.7975869,11.09489174,11.39244701,11.69024903,11.98829426,12.28657927,12.58510076],
        [1.283164616, 1.143451734,1.118287452,1.091405185,1.073974314,1.061083353,1.050870103,1.042423648,1.035229826,1.028969927,1.023432791,1.018471342,1.013979073,1.009876397,1.006102299,1.002608985,0.999358326,0.996319423,0.993466895,0.990779636,0.988239915,0.98583269,0.983545095,0.981366039,0.979285895,0.977296253,0.975389729,0.973559797,0.971800668,0.970107179,0.968474711,0.96689911,0.965376631,0.963903883,0.962477789,0.961095547,0.959754597,0.958452596,0.957187394,0.955957012,0.954759627,0.953593552,0.952457226,0.951349204],
        [0.613402864, 0.691408432,0.716251113,0.747097905,0.770243126,0.789125974,0.805242837,0.819397029,0.832075367,0.84359754,0.854185916,0.864002036,0.873167308,0.881775509,0.889900723,0.897602596,0.90492992,0.911923172,0.918616329,0.925038223,0.931213544,0.937163614,0.942906988,0.948459918,0.953836728,0.959050116,0.96411139,0.969030676,0.973817075,0.978478798,0.983023289,0.98745731,0.991787029,0.99601809,1.000155667,1.004204522,1.008169042,1.012053284,1.015861004,1.01959569,1.023260584,1.026858708,1.030392882,1.033865742],
        [0.45574, 0.419044604,0.407412583,0.400034043,0.394560837,0.390184326,0.386525186,0.383373795,0.380601649,0.378124076,0.375882231,0.373833505,0.37194602,0.370195286,0.368562068,0.367030971,0.365589481,0.364227278,0.362935752,0.361707642,0.360536765,0.359417814,0.358346194,0.357317906,0.356329442,0.355377709,0.354459967,0.353573774,0.352716948,0.351887527,0.351083742,0.350303994,0.349546829,0.348810924,0.34809507,0.347398161,0.346719181,0.346057192,0.345411333,0.344780805,0.344164868,0.343562835,0.342974069,0.342397974],
        [0.20564, 0.23077769,0.24117008,0.248264898,0.253765062,0.258303186,0.262190337,0.26560441,0.268657435,0.271424879,0.273960179,0.276302663,0.278482173,0.280521931,0.282440387,0.28425246,0.285970396,0.287604382,0.289162988,0.290653497,0.292082152,0.293454353,0.294774798,0.296047608,0.297276413,0.298464433,0.299614534,0.300729282,0.301810986,0.302861725,0.303883383,0.304877674,0.305846157,0.306790261,0.307711293,0.308610455,0.309488855,0.310347518,0.311187389,0.312009348,0.312814211,0.313602737,0.314375637,0.315133572],
        [0.019531558, 0.204126222,0.387580079,0.571430388,0.755852132,0.940873064,1.126484358,1.312666426,1.499396945,1.686653692,1.874415563,2.0626629,2.251377538,2.440542729,2.630143031,2.820164189,3.010593014,3.201417273,3.392625596,3.584207382,3.776152728,3.968452358,4.161097567,4.354080162,4.547392425,4.741027061,4.934977173,5.129236221,5.323797995,5.518656593,5.713806393,5.909242035,6.104958401,6.300950598,6.497213944,6.69374395,6.890536314,7.087586902,7.284891742,7.482447012,7.680249033,7.878294259,8.07657927,8.275100765],
    ])
    # These are off very very slightly
    allocation_array_cols = [
        "prior_root_biomass",
        "total_biomass",
        "delta_leaf_unit_root",
        "delta_stem_unit_root",
        "leaf_mass_frac",
        "stem_mass_frac",
        "abg_biomass",
    ]
    idx = 0
    for var in allocation_array_cols:
        # Using larger relative tolerance because knowns are approximations not direct solve
        assert_allclose(s.biomass_allocation_array[var], biomass_array_from_excel[idx], rtol=0.05)
        idx += 1


def test_set_initial_cover(example_input_params):
    s = create_species_object(example_input_params)
    species_name = "BTS"
    pidval = 0
    cell_index = 1
    plantlist = []
    # Cover area is too small for any plants
    cover_area = 0.00001
    plantlist = s.set_initial_cover(cover_area, species_name, pidval, cell_index, plantlist)
    assert len(plantlist) == 0
    assert pidval == 0
    # Correct dimension is populated
    cover_area = 0.2
    plantlist = s.set_initial_cover(cover_area, species_name, pidval, cell_index, plantlist)
    basal_dias = plantlist[0][18]
    assert basal_dias > 0
    # Area being used properly
    sum_area = 0
    for i in range(len(plantlist)):
        inc_area = 0.25 * np.pi * plantlist[i][18]**2
        sum_area += inc_area
    min_cover_area = 0.25 * np.pi * s.species_morph_params["basal_dia"]["min"]**2
    assert sum_area < cover_area
    assert (cover_area - sum_area) < min_cover_area


def test_set_initial_biomass(example_input_params, example_plant):
    # Testing for Deciduous set_initial_biomass, testing for other classes handled under test_emerge
    s = create_species_object(example_input_params)
    in_growing_season = True
    abg_biomass = example_plant["leaf"] + example_plant["stem"]
    root = 0.433864
    leaf = 0.485539
    stem = 0.314443
    min_repro = example_input_params["BTS"]["grow_params"]["plant_part_min"]["reproductive"]
    max_repro = example_input_params["BTS"]["grow_params"]["plant_part_max"]["reproductive"]
    example_plant["basal_dia"], shoot_sys_width, height = s.habit.calc_abg_dims_from_biomass(abg_biomass)
    example_plant["root"] = np.zeros_like(example_plant["root"])
    example_plant["leaf"] = np.zeros_like(example_plant["leaf"])
    example_plant["stem"] = np.zeros_like(example_plant["stem"])
    emerged_plant = s.set_initial_biomass(example_plant, in_growing_season)
    assert_allclose(emerged_plant["root"], root, rtol=0.001)
    assert_allclose(emerged_plant["leaf"], leaf, rtol=0.001)
    assert_allclose(emerged_plant["stem"], stem, rtol=0.001)
    assert_array_less(emerged_plant["reproductive"], max_repro)
    assert_array_less(min_repro, emerged_plant["reproductive"])
    assert_allclose(emerged_plant["shoot_sys_width"], shoot_sys_width, rtol=0.001)
    assert_allclose(emerged_plant["shoot_sys_height"], height, rtol=0.001)
    in_growing_season = False
    emerged_plant = s.set_initial_biomass(example_plant, in_growing_season)
    zeros = np.zeros_like(example_plant["root"])
    assert_allclose(emerged_plant["root"], root, rtol=0.001)
    assert_allclose(emerged_plant["leaf"], zeros, rtol=0.001)
    assert_allclose(emerged_plant["stem"], zeros, rtol=0.001)
    assert_allclose(emerged_plant["shoot_sys_width"], zeros, rtol=0.001)
    assert_allclose(emerged_plant["shoot_sys_height"], zeros, rtol=0.001)


def test_sum_plant_parts(example_plant, example_input_params):
    s = create_species_object(example_input_params)
    example_plant["reproductive"] = np.array([0.32])
    example_plant["dead_root"] = np.array([0.15])
    example_plant["dead_leaf"] = np.array([0.12])
    example_plant["dead_stem"] = np.array([0.09])
    example_plant["dead_reproductive"] = np.array([0.02])
    sums = {
        "total": 1.92,
        "growth": 1.6,
        "aboveground": 0.8,
        "persistent": 1.12,
        "green": 0.8,
        "dead": 0.38,
        "dead_aboveground": 0.21,
    }
    for sum_type in sums:
        sum_calc = s.sum_plant_parts(example_plant, sum_type)
        assert_allclose(sum_calc, sums[sum_type], rtol=0.001)


def test_emerge(example_plant, example_input_params):
    species_object = create_species_object(example_input_params)
    example_plant["leaf"] = np.zeros_like(example_plant["leaf"])
    example_plant["stem"] = np.zeros_like(example_plant["stem"])
    jday = 195
    nsc_content_spring195_actual = {
        "root": 205.575914 / 1000,
        "leaf": 223.822083 / 1000,
        "stem": 93.9498113 / 1000,
        "reproductive": 220.297893 / 1000,
    }
    avail_biomass = (
        (
            nsc_content_spring195_actual["root"]
            - species_object.species_grow_params["min_nsc_content"]["root"]
            * np.ones_like(example_plant["root"])
        )
        * example_plant["root"]
        + (
            nsc_content_spring195_actual["reproductive"]
            - species_object.species_grow_params["min_nsc_content"]["reproductive"]
            * np.ones_like(example_plant["reproductive"])
        )
        * example_plant["reproductive"]
    )
    total_pers_biomass = example_plant["root"] + example_plant["reproductive"]
    emerged_plant = species_object.habit.duration.emerge(example_plant, avail_biomass, total_pers_biomass)
    emerged_plant = species_object.update_morphology(emerged_plant)
    check_plant = species_object.emerge(example_plant, jday)
    check_vars = [
        "root",
        "leaf",
        "stem",
        "reproductive",
        "basal_dia",
        "shoot_sys_width",
        "shoot_sys_height",
        "root_sys_width",
        "live_leaf_area"
    ]
    for var in check_vars:
        assert_allclose(check_plant[var], emerged_plant[var], rtol=0.001)


def test__adjust_biomass_allocation_towards_ideal(example_input_params, example_plant):
    s = create_species_object(example_input_params)
    new_root = np.array([0.783358])
    new_leaf = np.array([0.508459])
    new_stem = np.array([0.308183])
    adjust_plant = s._adjust_biomass_allocation_towards_ideal(example_plant)
    assert_allclose(adjust_plant["root"], new_root, rtol=0.001)
    assert_allclose(adjust_plant["leaf"], new_leaf, rtol=0.001)
    assert_allclose(adjust_plant["stem"], new_stem, rtol=0.001)
    # Check to see if ratios adjusted when root is small
    small_root_plant = example_plant
    small_root_plant["root"] = np.array([0.005])
    small_root_plant["leaf"] = np.array([0.5])
    small_root_plant["stem"] = np.array([0.2999])
    s.species_grow_params["plant_part_min"]["root"] = 0.03
    new_root = np.array([0.030000])
    new_leaf = np.array([0.484195])
    new_stem = np.array([0.290805])
    adjust_plant = s._adjust_biomass_allocation_towards_ideal(small_root_plant)
    assert_allclose(adjust_plant["root"], new_root, rtol=0.001)
    assert_allclose(adjust_plant["leaf"], new_leaf, rtol=0.001)
    assert_allclose(adjust_plant["stem"], new_stem, rtol=0.001)


def test_allocate_biomass_dynamically(example_input_params, example_plant):
    s = create_species_object(example_input_params)
    new_root = np.array([0.789284])
    new_leaf = np.array([0.514743])
    new_stem = np.array([0.312879])
    new_repro = np.array([0.000104917])
    allocated_plant = s.allocate_biomass_dynamically(example_plant, np.array([0.025]))
    assert_allclose(allocated_plant["root"], new_root, rtol=0.001)
    assert_allclose(allocated_plant["leaf"], new_leaf, rtol=0.001)
    assert_allclose(allocated_plant["stem"], new_stem, rtol=0.001)
    assert_allclose(allocated_plant["reproductive"], new_repro, rtol=0.001)


def test_mortality(example_input_params, two_cell_grid, example_plant_array):
    temp_dt = np.timedelta64(example_input_params["BTS"]["mortality_params"]["duration"]["2"], 'D')
    species_object = create_species_object(example_input_params, dt=temp_dt)
    example_plant_array["cell_index"][4:] = np.array([1, 1, 1, 1])
    inital_leaf = example_plant_array["leaf"].copy()
    initial_plants = example_plant_array["root"].copy()
    # Check to make sure leaf self shade is working
    growing_season = True
    shaded_plants = species_object.mortality(example_plant_array, two_cell_grid, growing_season)
    assert_almost_equal(shaded_plants["leaf"][0:2], inital_leaf[0:2])
    assert_array_less(shaded_plants["leaf"][2], inital_leaf[2])
    # Check to be sure mortality doesn't happen outside period
    growing_season = False
    two_cell_grid["cell"]["air__min_temperature_C"] = np.array([-38, 8.62])
    two_cell_grid["cell"]["air__max_temperature_C"] = np.array([15.53, 45])
    plant_out = species_object.mortality(example_plant_array, two_cell_grid, growing_season)
    assert_almost_equal(plant_out["root"][4:], initial_plants[4:], decimal=6)
    assert_almost_equal(plant_out["root"][0:4], np.zeros_like(plant_out["root"][0:4]), decimal=6)
    # Check multiple mortality factors
    growing_season = True
    plant_out = species_object.mortality(example_plant_array, two_cell_grid, growing_season)
    assert_almost_equal(plant_out["root"], np.zeros_like(plant_out["root"]), decimal=6)
    # Check leaf weight if whole plant and shaded leaf mortality occurs
    assert_almost_equal(plant_out["leaf"], np.zeros_like(plant_out["leaf"]), decimal=6)


def test_respire(example_plant, one_cell_grid, example_input_params):
    species_object = create_species_object(example_input_params)
    # Respire with no water growth limitation from WOFOST in PCSE and crop development stage=1 divided by carb conversion
    wofost_resp_root = np.array([0.0033926])
    wofost_resp_stem = np.array([0.00121421])
    wofost_resp_leaf = np.array([0.00418568])
    # GV inputs
    max_temp = one_cell_grid["cell"]["air__max_temperature_C"][example_plant["cell_index"]]
    min_temp = one_cell_grid["cell"]["air__min_temperature_C"][example_plant["cell_index"]]
    relative_saturation = np.zeros_like(one_cell_grid["cell"]["soil_water__volume_fraction"][example_plant["cell_index"]])
    plant_out = species_object.respire(min_temp, max_temp, relative_saturation, example_plant)
    mass_change_resp_root = np.subtract(example_plant["root"], plant_out["root"])
    mass_change_resp_leaf = np.subtract(example_plant["leaf"], plant_out["leaf"])
    mass_change_resp_stem = np.subtract(example_plant["stem"], plant_out["stem"])
    assert_almost_equal(wofost_resp_root, mass_change_resp_root, 6)
    assert_almost_equal(wofost_resp_leaf, mass_change_resp_leaf, 6)
    assert_almost_equal(wofost_resp_stem, mass_change_resp_stem, 6)
    # Respire with from WOFOST in PCSEand crop development stage=1 with saturation limitation divided by carb conversion
    wofost_resp_root = np.array([0.00441038])
    wofost_resp_stem = np.array([0.00121421])
    wofost_resp_leaf = np.array([0.00418568])
    # GV inputs
    relative_saturation = np.ones_like(one_cell_grid["cell"]["soil_water__volume_fraction"][example_plant["cell_index"]]) * 0.5
    plant_out_half_sat = species_object.respire(min_temp, max_temp, relative_saturation, example_plant)
    mass_change_resp_root_half_sat = np.subtract(example_plant["root"], plant_out_half_sat["root"])
    mass_change_resp_leaf_half_sat = np.subtract(example_plant["leaf"], plant_out_half_sat["leaf"])
    mass_change_resp_stem_half_sat = np.subtract(example_plant["stem"], plant_out_half_sat["stem"])
    assert_almost_equal(wofost_resp_root, mass_change_resp_root_half_sat, 6)
    assert_almost_equal(wofost_resp_leaf, mass_change_resp_leaf_half_sat, 6)
    assert_almost_equal(wofost_resp_stem, mass_change_resp_stem_half_sat, 6)


def test_sum_vars_in_calculate_derived_params(example_input_params):
    species_object = create_species_object(example_input_params)
    species_param = species_object.calculate_derived_params(example_input_params["BTS"])
    # Checked via excel
    # Max total Biomass
    assert_almost_equal(species_param["grow_params"]["total_biomass"]["max"], 17.9)
    # max_growth_biomass
    assert_almost_equal(species_param["grow_params"]["growth_biomass"]["max"], 13.9)
    # max_abg_biomass
    assert_almost_equal(species_param["grow_params"]["abg_biomass"]["max"], 9.6)
    # min_total_biomass
    assert_almost_equal(species_param["grow_params"]["total_biomass"]["min"], 0.062222222)
    # min_growth_biomass
    assert_almost_equal(species_param["grow_params"]["growth_biomass"]["min"], 0.062222222)
    # min_abg_biomass
    assert_almost_equal(species_param["grow_params"]["abg_biomass"]["min"], 0.052222222)
    # min_nsc_biomass
    assert_almost_equal(species_param["grow_params"]["nsc_biomass"]["min"], 0.03369)


def test_senesce(example_input_params, example_plant):
    species_object = create_species_object(example_input_params)
    jday = 195
    # leaves and stems should move nonstructural carb content to roots at a fixed rate
    # calculated values from Excel at day 195 for one plant
    end_root = np.array([0.803921028])
    end_stem = np.array([0.29399902])
    end_leaf = np.array([0.485])
    end_repro = np.array([0.])
    plant_out = species_object.senesce(example_plant, jday)
    assert_almost_equal(plant_out["reproductive"], end_repro, decimal=6)
    assert_almost_equal(plant_out["root"], end_root, decimal=6)
    assert_almost_equal(plant_out["stem"], end_stem, decimal=6)
    assert_almost_equal(plant_out["leaf"], end_leaf, decimal=6)


def test_nsc_rate_change_per_season_and_part(example_input_params):
    species_object = create_species_object(example_input_params)
    species_param = species_object.calculate_derived_params(example_input_params["BTS"])
    ncs_rate_change = species_param["duration_params"]["nsc_rate_change"]
    # winter_nsc_rate
    # - leaf
    assert_almost_equal(ncs_rate_change["winter_nsc_rate"]["leaf"], 0.003676471)
    # - reproductive
    assert_almost_equal(
        ncs_rate_change["winter_nsc_rate"]["reproductive"], -0.004595588
    )
    # - root
    assert_almost_equal(ncs_rate_change["winter_nsc_rate"]["root"], -0.003676471)
    # - stem
    assert_almost_equal(ncs_rate_change["winter_nsc_rate"]["stem"], -0.00245098)
    # spring_nsc_rate
    # - leaf
    assert_almost_equal(ncs_rate_change["spring_nsc_rate"]["leaf"], -0.015060241)
    # - reproductive
    assert_almost_equal(
        ncs_rate_change["spring_nsc_rate"]["reproductive"], -0.041415663
    )
    # - root
    assert_almost_equal(ncs_rate_change["spring_nsc_rate"]["root"], -0.045180723)
    # - stem
    assert_almost_equal(ncs_rate_change["spring_nsc_rate"]["stem"], -0.006024096)
    # summer_nsc_rate
    # - leaf
    assert_almost_equal(ncs_rate_change["summer_nsc_rate"]["leaf"], -0.02173913)
    # - reproductive
    assert_almost_equal(ncs_rate_change["summer_nsc_rate"]["reproductive"], 0.042119565)
    # - root
    assert_almost_equal(ncs_rate_change["summer_nsc_rate"]["root"], 0.054347826)
    # - stem
    assert_almost_equal(ncs_rate_change["summer_nsc_rate"]["stem"], 0.010869565)
    # fall_nsc_rate
    # - leaf
    assert_almost_equal(ncs_rate_change["fall_nsc_rate"]["leaf"], 0.046875)
    # - reproductive
    assert_almost_equal(ncs_rate_change["fall_nsc_rate"]["reproductive"], 0.076171875)
    # - root
    assert_almost_equal(ncs_rate_change["fall_nsc_rate"]["root"], 0.0625)
    # - stem
    assert_almost_equal(ncs_rate_change["fall_nsc_rate"]["stem"], 0.015625)


def test_select_habit_class(example_input_params):
    species_object = create_species_object(example_input_params)
    dummy_species = example_input_params
    for spec, cls in zip(
        ["forb_herb", "graminoid", "shrub", "tree", "vine"],
        [Forbherb, Graminoid, Shrub, Tree, Vine],
    ):
        dummy_species["BTS"]["plant_factors"]["growth_habit"] = spec
        print(dummy_species["BTS"]["plant_factors"]["growth_habit"])
        assert isinstance(species_object.select_habit_class(dummy_species["BTS"]), cls)


def test_select_form_class(example_input_params):
    species_object = create_species_object(example_input_params)
    dummy_growth_form = example_input_params
    for growth, cls in zip(
        [
            "bunch",
            "colonizing",
            "multiple_stems",
            "rhizomatous",
            "single_crown",
            "single_stem",
            "stoloniferous",
            "thicket_forming",
        ],
        [
            Bunch,
            Colonizing,
            Multiplestems,
            Rhizomatous,
            Singlecrown,
            Singlestem,
            Stoloniferous,
            Thicketforming,
        ],
    ):
        dummy_growth_form["BTS"]["plant_factors"]["growth_form"] = growth
        assert isinstance(
            species_object.select_form_class(dummy_growth_form["BTS"]), cls
        )


def test_select_photosythesis_type(example_input_params):
    species_object = create_species_object(example_input_params)
    dummy_photosythesis = example_input_params
    for photosynthesis_opt, cls in zip(["C3", "C4", "cam"], [C3, C4, Cam]):
        dummy_photosythesis["BTS"]["plant_factors"]["p_type"] = photosynthesis_opt
        assert isinstance(
            species_object.select_photosythesis_type(
                dummy_photosythesis["BTS"], 0.9074
            ),
            cls,
        )


def test_update_dead_biomass(example_input_params, example_plant):
    example_plant_new = example_plant.copy()
    example_plant_new["leaf"] *= 0.5
    example_plant_new["root"] *= 0.5
    example_plant_new["stem"] *= 0.5
    species_object = create_species_object(example_input_params)
    example_plant_new = species_object.update_dead_biomass(example_plant_new, example_plant)
    assert_almost_equal(example_plant_new["dead_leaf"], example_plant_new["leaf"], decimal=6)
    assert_almost_equal(example_plant_new["dead_root"], example_plant_new["root"])
    assert_almost_equal(example_plant_new["dead_stem"], example_plant_new["stem"])


def test_validate_plant_factors_raises_errors(example_input_params):
    # create species class (Note this will run test_validate_plant_factors during init
    species_object = create_species_object(example_input_params)

    # create an unexpected variable
    dummy_plant_factor = {"unexpected-variable-name": []}
    with pytest.raises(KeyError) as excinfo:
        # raising error for try opt_list = plant_factor_options[key]
        species_object.validate_plant_factors(dummy_plant_factor)
    # test that the correct message is sent
    # test_msg =  "Unexpected variable name in species parameter dictionary. Please check input parameter file"
    assert (
        str(excinfo.value)
        == "'Unexpected variable name in species parameter dictionary. Please check input parameter file'"
    )

    # create invalid option
    dummy_plant_type_factor = {"p_type": "fake-p-type"}
    with pytest.raises(ValueError) as excinfo:
        species_object.validate_plant_factors(dummy_plant_type_factor)
    assert str(excinfo.value) == "Invalid p_type option"


def test_validate_duration_params_raises_errors(example_input_params):
    # create Species class (note this will run validate_duration_params during init)
    species_object = create_species_object(example_input_params)

    # ValueError msg for growing_season_start is between 1-365
    start_below_bound = {"growing_season_start": 0}
    start_above_bound = {"growing_season_start": 367}
    with pytest.raises(ValueError) as excinfo:
        species_object.validate_duration_params(start_below_bound)
        species_object.validate_duration_params(start_above_bound)
    assert str(excinfo.value) == "Growing season beginning must be integer values between 1-365"

    # ValueError for growing_season_end is between 1-365 and greater than growing_season_start
    end_below_bound = {"growing_season_start": 144, "growing_season_end": 0}
    end_above_bound = {"growing_season_start": 144, "growing_season_end": 367}
    end_less_start = {"growing_season_start": 305, "growing_season_end": 144}
    with pytest.raises(ValueError) as excinfo:
        species_object.validate_duration_params(end_below_bound)
        species_object.validate_duration_params(end_above_bound)
        species_object.validate_duration_params(end_less_start)
    assert str(excinfo.value) == "Growing season end must be between 1-365 and greater than the growing season beginning"

    # ValueError for senescne being within the growing season
    senescence_less_start = {"growing_season_start": 273, "growing_season_end": 305, "senescence_start": 144}
    senescence_greater_end = {"growing_season_start": 144, "growing_season_end": 273, "senescence_start": 305}
    with pytest.raises(ValueError) as excinfo:
        species_object.validate_duration_params(senescence_less_start)
        species_object.validate_duration_params(senescence_greater_end)
    assert str(excinfo.value) == "Start of senescence must be within the growing season"


def test_litter_decomp_new_biomass_values(example_input_params, example_plant_array):
    # expected values obtained from excel
    expected_new_biomass = {
        'dead_root': np.array([0.011654923, 0.011654923, 0.011654923, 0.011654923, 1.631689185, 1.631689185, 1.631689185, 1.631689185]),
        'dead_stem': np.array([0.011654923, 1.631689185, 0.011654923, 1.631689185, 0.011654923, 1.631689185, 0.011654923, 1.631689185]),
        'dead_leaf': np.array([0.011654923, 0.011654923, 1.631689185, 1.631689185, 0.011654923, 0.011654923, 1.631689185, 1.631689185]),
        'dead_reproductive': np.array([0.815844592, 0.815844592, 0.815844592, 0.815844592, 0.815844592, 0.011654923, 0.011654923, 0.011654923]),
        'dead_root_age': np.array([1095.035397, 268.2368885, 383.4262438, 348.6774074, 1018.381215, 912.6529431, 315.3525663, 983.1220087]),
        'dead_stem_age': np.array([1316.902291, 1454.913069, 689.1424854, 330.2440673, 1363.62031, 458.1701171, 1378.041709, 1365.471028]),
        'dead_leaf_age': np.array([1330.561664, 1353.338509, 1239.787392, 1434.019734, 1349.797247, 736.8478796, 1178.379645, 235.6844825]),
        'dead_reproductive_age': np.array([508.8777074, 1122.713079, 205.4386012, 1074.25249, 661.3439315, 427.4383309, 1262.53064, 1086.549562])
    }

    # initialize class
    species_object = create_species_object(example_input_params)
    # set dt
    species_object.dt = np.timedelta64(1, "D")
    # call litter_decomp
    new_biomass = species_object.litter_decomp(example_plant_array)

    # tests
    for k, expected_value in expected_new_biomass.items():
        assert_allclose(
            new_biomass[k], expected_value, rtol=0.0001
        )


def test_litter_decomp_bad_values_replaced(example_input_params, example_plant_array):
    # initialize class
    species_object = create_species_object(example_input_params)
    # set dt
    species_object.dt = np.timedelta64(1, "D")
    # make a nan value, negative value, and inf value
    bad_epa_values = example_plant_array
    bad_epa_values["dead_root"][0] = -0.0125
    bad_epa_values["dead_stem"][1] = np.nan
    bad_epa_values["dead_leaf"][2] = np.inf

    # run method
    new_biomass = species_object.litter_decomp(bad_epa_values)

    # expected values obtained from excel and replace with 0.0 where bad values were
    expected_new_biomass = {
        'dead_root': np.array([0.0, 0.011654923, 0.011654923, 0.011654923, 1.631689185, 1.631689185, 1.631689185, 1.631689185]),
        'dead_stem': np.array([0.011654923, 0.0, 0.011654923, 1.631689185, 0.011654923, 1.631689185, 0.011654923, 1.631689185]),
        'dead_leaf': np.array([0.011654923, 0.011654923, 0.0, 1.631689185, 0.011654923, 0.011654923, 1.631689185, 1.631689185]),
    }

    # test
    for k, expected_values in expected_new_biomass.items():
        assert_allclose(
            new_biomass[k], expected_values, rtol=0.0001
        )
