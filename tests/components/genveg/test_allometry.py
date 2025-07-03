import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal


dt = np.timedelta64(1, 'D')


def test__calc2_allometry_coeffs(biomass_minmax):
    x_min = np.array([1.])
    x_max = np.array([3.])
    y_min = np.array([2.])
    y_max = np.array([6.])
    slope = np.array([1])
    y_intercept = np.array([np.log(2)])
    (b, m) = biomass_minmax._calc2_allometry_coeffs(x_min, x_max, y_min, y_max)
    assert_array_almost_equal(y_intercept, b, decimal=5)
    assert_array_almost_equal(slope, m, decimal=5)


def test_apply2_allometry_eq_for_xs(biomass_minmax):
    biomass_minmax.morph_params["empirical_coeffs"]["basal_dia_coeffs"] = {"a": np.log(2), "b": 1}
    abg_biomass = np.array([2.1, 0.2, 15.2])
    pred_basal_diameters = biomass_minmax._apply2_allometry_eq_for_xs(abg_biomass, "basal_dia_coeffs")
    basal_diameters = np.array([1.05, 0.1, 7.6])
    assert_array_almost_equal(pred_basal_diameters, basal_diameters, decimal=5)


def test_apply2_allometry_eq_for_ys(biomass_minmax):
    biomass_minmax.morph_params["empirical_coeffs"]["basal_dia_coeffs"] = {"a": np.log(2), "b": 1}
    basal_diameter = np.array([0.5, 3.5, 12])
    pred_abg_biomass = biomass_minmax._apply2_allometry_eq_for_ys(basal_diameter, "basal_dia_coeffs")
    abg_biomass = np.array([1., 7., 24.])
    assert_array_almost_equal(pred_abg_biomass, abg_biomass, decimal=5)


def test_allometric_coeffs_calculated(biomass_minmax):
    # Check calculation for min-max
    empirical_coeffs = {
        "basal_dia_coeffs": {"a": 6.26936, "b": 1.74048},
        "height_coeffs": {"a": 2.47344, "b": 2.96068},
        "canopy_area_coeffs": {"a": 4.2926, "b": 0.766496},
    }
    for item in empirical_coeffs:
        assert biomass_minmax.morph_params["empirical_coeffs"][item] == pytest.approx(empirical_coeffs[item])


def test_calc_abg_dims_minmax(biomass_minmax):
    # Biomass relationships
    abg_biomass = np.array([2.1, 0.5, 15.2])
    basal_diameter = np.array([0.041761, 0.018309, 0.130217])
    height = np.array([0.5572, 0.3432, 1.0873])
    shoot_sys_width = np.array([0.11132, 0.04365, 0.40486])
    pred_bd, pred_ssw, pred_h = biomass_minmax.calc_abg_dims(abg_biomass, cm=False)
    print(biomass_minmax.morph_params["allometry_method"])
    assert_allclose(pred_bd, basal_diameter, rtol=0.001)
    assert_allclose(pred_h, height, rtol=0.001)
    assert_allclose(pred_ssw, shoot_sys_width, rtol=0.001)


def test_calc_abg_biomass_from_dim(biomass_minmax):
    basal_dia = np.array([0.005, 0.035, 0.12])
    shoot_sys_width = np.array([0.0640, 0.2, 0.48])
    abg_from_bd = np.array([0, 1.5445, 13.1851])
    abg_from_ssw = np.array([0.899, 5.156, 19.7325])
    pred_abg_bd = biomass_minmax.calc_abg_biomass_from_dim(basal_dia, "basal_dia", cm=False)
    assert_allclose(abg_from_bd, pred_abg_bd, rtol=0.001)
    pred_abg_ssw = biomass_minmax.calc_abg_biomass_from_dim((0.25 * np.pi * shoot_sys_width**2), "canopy_area")
    assert_allclose(abg_from_ssw, pred_abg_ssw, rtol=0.001)


def test_allometric_coeffs_are_user_defined(dimensional_user, example_input_params):
    empirical_coeffs = example_input_params["BTS"]["morph_params"]["empirical_coeffs"]
    assert dimensional_user.morph_params["empirical_coeffs"] == empirical_coeffs


def test_allometric_coeffs_are_default(dimensional_default):
    empirical_coeffs = {
        "basal_dia_coeffs": {"a": 0.5093, "b": 0.47},
        "height_coeffs": {"a": np.log(0.232995), "b": 0.619077},
        "canopy_area_coeffs": {"a": np.log(0.23702483 * 0.2329925**0.9459644), "b": 0.72682 + (0.619077 * 0.9459644)},
    }
    for item in empirical_coeffs:
        assert dimensional_default.morph_params["empirical_coeffs"][item] == pytest.approx(empirical_coeffs[item])


def test_calc_abg_dims_default(dimensional_default):
    # Dimensional relationships
    basal_diameter = np.array([0.009, 0.036, 0.05])
    height = np.array([0.218283, 0.514921, 0.631049])
    shoot_sys_width = np.array([0.257388353, 0.639255015, 0.79304121])
    abg_biomass = np.array([1.58372641, 3.03842384, 3.54570077])
    pred_bd, pred_ssw, pred_h = dimensional_default.calc_abg_dims(abg_biomass, cm=True)
    assert_allclose(pred_bd, basal_diameter, rtol=0.001)
    assert_allclose(pred_ssw, shoot_sys_width, rtol=0.001)
    assert_allclose(pred_h, height, rtol=0.001)


def test__calc_shoot_width_from_basal_dia(dimensional_default):
    basal_dia = np.array([0.009, 0.036, 0.05])
    shoot_width = np.array([0.257388353, 0.639255015, 0.79304121])
    pred_shoot_width = dimensional_default._calc_shoot_width_from_basal_dia(basal_dia, cm=True)
    assert_allclose(pred_shoot_width, shoot_width, rtol=0.001)


def test__calc_height_from_basal_dia(dimensional_default):
    basal_dia = np.array([0.009, 0.036, 0.05])
    height = np.array([0.218283, 0.514921, 0.631049])
    pred_height = dimensional_default._calc_height_from_basal_dia(basal_dia, cm=True)
    assert_allclose(pred_height, height, rtol=0.001)


def test__calc_basal_dia_from_shoot_width(dimensional_default):
    basal_dia = np.array([0.009, 0.036, 0.05])
    shoot_width = np.array([0.257388353, 0.639255015, 0.79304121])
    pred_basal_dia = dimensional_default._calc_basal_dia_from_shoot_width(shoot_width, cm=True)
    assert_allclose(pred_basal_dia, basal_dia, rtol=0.001)


"""


def test__calc3_allometry_coeffs(example_input_params):
    h = Multi_Dimensional(params=example_input_params["BTS"])
    x_min = np.array([1.1])
    x_mean = np.array([1.5])
    x_max = np.array([3.])
    y_min = np.array([2.])
    y_mean = np.array([4.])
    y_max = np.array([6.])
    z_min = np.array([5.2])
    z_mean = np.array([9.5])
    z_max = np.array([16.7])
    # Assuming form is ln(z) = a + b ln(x) + c ln(y)
    a_wa = np.array([1.13487])
    b_wa = np.array([0.413507])
    c_wa = np.array([0.684388])
    (a, b, c) = h._calc3_allometry_coeffs(x_min, x_mean, x_max, y_min, y_mean, y_max, z_min, z_mean, z_max)
    assert_array_almost_equal(a, a_wa, decimal=5)
    assert_array_almost_equal(b, b_wa, decimal=5)
    assert_array_almost_equal(c, c_wa, decimal=5)


def test_apply3_allometry_eq_for_ys(example_input_params):
    h = Multi_Dimensional(params=example_input_params["BTS"])
    h.morph_params["canopy_coeffs"] = {"a": 1.13487, "b": 0.413507, "c": 0.684388}
    abg_biomass = np.array([2.1, 0.2, 15.2])
    basal_dia = np.array([0.5, 3.5, 12])
    pred_canopy = h._apply3_allometry_eq_for_ys(basal_dia, abg_biomass, "canopy_coeffs")
    canopy = np.array([0.856126, 0.008508, 2.262885])
    assert_array_almost_equal(pred_canopy, canopy, decimal=6)


def test_apply3_allometry_eq_for_xs(example_input_params):
    h = Multi_Dimensional(params=example_input_params["BTS"])
    h.morph_params["canopy_coeffs"] = {"a": 1.13487, "b": 0.413507, "c": 0.684388}
    abg_biomass = np.array([2.1, 0.2, 15.2])
    canopy_area = np.array([1.28, 0.05, 3.7])
    pred_basal_dia = h._apply3_allometry_eq_for_xs(canopy_area, abg_biomass, "canopy_coeffs")
    basal_dia = np.array([0.256964, 0.186657, 5.318099])
    assert_array_almost_equal(pred_basal_dia, basal_dia, decimal=6)


def test_apply3_allometry_eq_for_zs(example_input_params):
    h = Multi_Dimensional(params=example_input_params["BTS"])
    h.morph_params["canopy_coeffs"] = {"a": 1.13487, "b": 0.413507, "c": 0.684388}
    canopy_area = np.array([1.28, 0.05, 3.7])
    basal_dia = np.array([0.5, 3.5, 12])
    pred_abg = h._apply3_allometry_eq_for_zs(basal_dia, canopy_area, "canopy_coeffs")
    abg = np.array([2.765432, 0.672101, 21.280764])
    assert_array_almost_equal(pred_abg, abg, decimal=6)

"""
