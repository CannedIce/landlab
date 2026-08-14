import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal

from landlab.components.genveg.duration import Annual, Deciduous, Evergreen


dt = np.timedelta64(1, 'D')


def test_calc_canopy_area_from_shoot_width(habit_object):
    # zero array returns zero array
    assert_array_almost_equal(
        habit_object._calc_canopy_area_from_shoot_width(shoot_sys_width=np.array([0, 0, 0])),
        np.array([0, 0, 0]),
    )
    # single input returns correct input
    assert_equal(
        habit_object._calc_canopy_area_from_shoot_width(shoot_sys_width=0.325), 0.08295768100885548
    )
    # an array of values
    assert_allclose(
        habit_object._calc_canopy_area_from_shoot_width(
            shoot_sys_width=np.array([0, 0.0004, 0.678, 1.5, 3])
        ),
        np.array(
            [
                0.00000000e00,
                1.25663706e-07,
                3.61034969e-01,
                1.76714587e00,
                7.06858347e00,
            ]
        ),
    )


def test__calc_diameter_from_area(habit_object):
    canopy_area = np.array([0.1, 1.28, 3.7])
    shoot_width = np.array([0.35682, 1.27662, 2.17048])
    pred_shoot_width = habit_object._calc_diameter_from_area(canopy_area)
    assert_array_almost_equal(shoot_width, pred_shoot_width, decimal=5)


def test__calc_canopy_volume(habit_object):
    shoot_width = np.array([0.08, 0.35, 0.72])
    height = np.array([0.1, 0.5, 1.7])
    basal_dia = np.array([0.005, 0.02, 0.1])
    volume = np.array([0.00017868, 0.017004, 0.2672])
    pred_vol = habit_object._calc_canopy_volume(shoot_width, basal_dia, height)
    assert_allclose(pred_vol, volume, rtol=0.0001)


def test_calc_root_sys_width(habit_object):
    shoot_width = np.array([0.08, 0.35, 0.72])
    height = np.array([0.1, 0.5, 1.7])
    basal_dia = np.array([0.005, 0.02, 0.1])
    root_width = np.array([0.080043, 0.084081, 0.144128])
    pred_root_width = habit_object.calc_root_sys_width(shoot_width, basal_dia, height)
    assert_allclose(pred_root_width, root_width, rtol=0.0001)


def test_estimate_abg_biomass_from_cover_graminoid(graminoid_object, example_plant_array):
    abg_biomass = example_plant_array["leaf"] + example_plant_array["stem"]
    example_plant_array["basal_dia"], example_plant_array["shoot_sys_width"], example_plant_array["shoot_sys_height"] = graminoid_object.calc_abg_dims_from_biomass(abg_biomass)
    est_abg_biomass = graminoid_object.estimate_abg_biomass_from_cover(example_plant_array)
    assert_allclose(est_abg_biomass, abg_biomass, rtol=0.0001)


def test_estimate_abg_biomass_from_cover(habit_object, example_plant_array):
    abg_biomass = example_plant_array["leaf"] + example_plant_array["stem"]
    example_plant_array["basal_dia"], example_plant_array["shoot_sys_width"], example_plant_array["shoot_sys_height"] = habit_object.calc_abg_dims_from_biomass(abg_biomass)
    est_abg_biomass = habit_object.estimate_abg_biomass_from_cover(example_plant_array)
    assert_allclose(est_abg_biomass, abg_biomass, rtol=0.0001)


# change this to ensure zero in case of negative
def test_calc_canopy_area_from_shoot_width_raises_error(habit_object):
    with pytest.raises(ValueError):
        habit_object._calc_canopy_area_from_shoot_width(-1.5)
        habit_object._calc_canopy_area_from_shoot_width(np.array([0, 0.004, -0.678, 1.5, 3]))


def test_select_duration_class(habit_object, example_input_params):
    dummy_habit = example_input_params
    dt = np.timedelta64(1, 'D')
    for duration_opt, cls in zip(["annual", "perennial deciduous", "perennial evergreen"], [Annual, Deciduous, Evergreen]):
        dummy_habit["BTS"]["plant_factors"]["duration"] = duration_opt
        assert isinstance(
            habit_object._select_duration_class(
                dummy_habit["BTS"]["grow_params"],
                dummy_habit["BTS"]["duration_params"],
                dt,
                duration_opt, green_parts=("leaf")
            ),
            cls,
        )
