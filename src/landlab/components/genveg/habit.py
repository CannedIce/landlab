import numpy as np

from .check_objects import UnitTestChecks
from .duration import Annual
from .duration import Deciduous
from .duration import Evergreen
from .allometry import Biomass
from .allometry import Dimensional

rng = np.random.default_rng()


# Growth habit classes and selection method
# Growth habit uses duration properties to assign dormancy and emergence methods
class Habit:
    def __init__(self, params, allometry, dt=1, green_parts=(None)):
        self.grow_params = params["grow_params"]
        self.duration_params = params["duration_params"]
        self.morph_params = params["morph_params"]
        self.allometry = allometry
        self.duration = self._select_duration_class(
            self.grow_params,
            self.duration_params,
            dt,
            params["plant_factors"]["duration"],
            green_parts,
        )

    def _calc_canopy_area_from_shoot_width(self, shoot_sys_width):
        UnitTestChecks().is_negative_present(shoot_sys_width, "shoot_sys_width")
        canopy_area = 0.25 * np.pi * shoot_sys_width**2
        return canopy_area

    def _calc_canopy_volume(self, shoot_width, basal_width, shoot_height):
        shoot_rad = shoot_width / 2
        basal_rad = basal_width / 2
        volume = (
            1
            / 3
            * np.pi
            * shoot_height
            * (shoot_rad**2 + shoot_rad * basal_rad + basal_rad**2)
        )
        return volume

    def _calc_root_sys_width(self, shoot_sys_width, basal_width, shoot_sys_height=1):
        volume = self._calc_canopy_volume(shoot_sys_width, basal_width, shoot_sys_height)
        root_sys_width = 0.08 + 0.24 * volume
        root_sys_width[root_sys_width > self.morph_params["root_sys_width"]["max"]] = (
            self.morph_params["root_sys_width"]["max"]
        )
        root_sys_width[root_sys_width < self.morph_params["root_sys_width"]["min"]] = (
            self.morph_params["root_sys_width"]["min"]
        )
        return root_sys_width

    def _calc_diameter_from_area(self, canopy_area):
        return np.sqrt(4 * canopy_area / np.pi)

    def _select_allometry_class(self, species_params, empirical_coeffs):
        allometry_method = species_params["morph_params"]["allometry_method"]
        allometry = {
            "min-max": Biomass(species_params),
            "user_defined": Dimensional(species_params, species_params["morph_params"]["empirical_coeffs"]),
            "default": Dimensional(species_params, empirical_coeffs),
        }
        return allometry[allometry_method]

    def _select_duration_class(
        self,
        species_grow_params,
        duration_params,
        dt,
        duration_val,
        green_parts=(None),
    ):
        duration = {
            "annual": Annual(species_grow_params, duration_params, dt),
            "perennial deciduous": Deciduous(
                species_grow_params, duration_params, dt, green_parts
            ),
            "perennial evergreen": Evergreen(species_grow_params, duration_params, dt),
        }
        return duration[duration_val]

    def _calc_abg_dims_from_biomass(self, abg_biomass):
        # These dimensions are empirically derived allometric relationships.
        basal_dia_cm = height = canopy_area = np.ones_like(abg_biomass)
        filter = np.nonzero(abg_biomass > self.grow_params["abg_biomass"]["min"])
        basal_dia_cm[filter] = self.allometry._apply2_allometry_eq_for_xs(abg_biomass[filter], "basal_coeffs")
        basal_width = basal_dia_cm / 100
        height[filter] = self.allometry._apply2_allometry_eq_for_xs(abg_biomass[filter], "height_coeffs")
        canopy_area[filter] = self.allometry._apply2_allometry_eq_for_xs(abg_biomass[filter], "canopy_coeffs")
        shoot_sys_width = self._calc_diameter_from_area(canopy_area)
        return basal_width, shoot_sys_width, height

    def emerge(self, plants):
        plants = self.duration.emerge(plants)
        return plants

    def _estimate_abg_biomass_from_cover(self, plants):
        # redo this to back calculate percent cover and height to basal diameter and abg biomass
        log_canopy_area = np.log(
            0.25 * np.pi * (plants["shoot_sys_width"])**2,
            out=np.zeros_like(plants["shoot_sys_width"], dtype=np.float64),
            where=(plants["shoot_sys_width"] > 0.0),
        )
        log_abg_biomass = (
            self.morph_params["canopy_area_coeffs"]["a"]
            + self.morph_params["canopy_area_coeffs"]["b"] * log_canopy_area
        )
        est_abg_biomass = np.exp(
            log_abg_biomass,
            out=np.zeros_like(log_abg_biomass, dtype=np.float64),
            where=(plants["basal_width"] > 0.0),
        )
        return est_abg_biomass

    def senesce(self, plants, ns_green_mass, persistent_mass):
        plants = self.duration.senesce(plants, ns_green_mass, persistent_mass)
        return plants

    def set_initial_biomass(self, plants, in_growing_season):
        plants = self.duration.set_initial_biomass(plants, in_growing_season)
        return plants

    def enter_dormancy(self, plants):
        plants = self.duration.enter_dormancy(plants)
        return plants


class Forbherb(Habit):
    def __init__(
        self,
        params,
        dt,
        empirical_coeffs={
            "basal_dia_coeffs": {"a": 2.8558, "b": 1.6226},
            "height_coeffs": {"a": -2.7057, "b": 0.2084},
            "canopy_area_coeffs": {"a": np.log(60), "b": 1.5},
        }
    ):
        # Using functional forms from BiomeE and data from Lu et al 2016
        # for height and basal diameter and relationship from BiomeE
        # for canopy area
        green_parts = ("leaf", "stem")
        allometry = self._select_allometry_class(params, empirical_coeffs)
        super().__init__(params, allometry, dt, green_parts)

    def _calc_abg_dims_from_biomass(self, abg_biomass):
        # These dimensions are empirically derived allometric relationships for grasses.
        basal_area_cm2 = height = canopy_area = np.zeros_like(
            abg_biomass
        )
        filter = np.nonzero(abg_biomass > self.grow_params["abg_biomass"]["min"])
        height[filter] = self._apply2_allometry_eq_for_xs(abg_biomass[filter], "height_coeffs")
        basal_area_cm2[filter] = self._apply3_allometry_eq_for_xs(height[filter], abg_biomass[filter], "basal_area_coeffs")
        basal_width = self._calc_diameter_from_area(basal_area_cm2) / 100
        canopy_area[filter] = self._apply3_allometry_eq_for_xs(height[filter], abg_biomass[filter], "canopy_area_coeffs")
        shoot_sys_width = self._calc_diameter_from_area(canopy_area)
        return basal_width, height, shoot_sys_width


class Graminoid(Habit):
    def __init__(
        self,
        params,
        dt,
        empirical_coeff_options={
            "perennial": {
                "C3": {
                    "basal_dia_coeffs": {"a": 0.4111, "b": 0.4498},
                    "height_coeffs": {"a": np.log(0.232995), "b": 0.619077},
                    "canopy_area_coeffs": {"a": np.log(0.23702483 * 0.2329925**0.9459644), "b": 0.72682 + (0.619077 * 0.9459644)},
                },
                "C4": {
                    "basal_dia_coeffs": {"a": -0.1988, "b": 0.3803},
                    "height_coeffs": {"a": np.log(0.2776634), "b": 0.4176197},
                    "canopy_area_coeffs": {"a": np.log(0.06669907 * 0.2776634**0.2002879), "b": 1.3043469 + (0.4176197 * 0.2002879)},
                },
            },
            "annual": {
                "C3": {
                    "basal_dia_coeffs": {"a": 0.5093, "b": 0.47},
                    "height_coeffs": {"a": np.log(0.1476171), "b": 0.6995105},
                    "canopy_area_coeffs": {"a": np.log(0.12826361 * 0.1476171**0.7576721), "b": 0.7134629 + (0.6995105 * 0.7576721)},
                },
                "C4": {
                    "basal_dia_coeffs": {"a": -0.1571, "b": 0.44},
                    "height_coeffs": {"a": np.log(0.4204882), "b": 0.5194908},
                    "canopy_area_coeffs": {"a": np.log(0.25749493 * 0.4204882**0.5700335), "b": 1.0866763 + (0.5194908 * 0.5700335)},
                },
            },
        },
    ):
        # Graminoid morphology
        # - basal diameter is a function of aboveground biomass
        # - height is a function of basal diameter
        # - canopy area is a function of height and basal diameter (combined)
        # - stem diameter and number of stems is a function of basal diameter
        # Default empirical parameters are directly or derived from Gao 2024
        dur_type = params["plant_factors"]["duration"].split("_")[0]
        p_type = params["plant_factors"]["p_type"]
        if params["morph_params"]["allometry_method"] == "default":
            empirical_coeffs = empirical_coeff_options[dur_type][p_type]
        elif params["morph_params"]["allometry_method"] == "min-max":
            empirical_coeffs = {}
        allometry = self._select_allometry_class(params, empirical_coeffs)
        green_parts = ("leaf", "stem")

        super().__init__(params, allometry, dt, green_parts)

    def _calc_abg_dims_from_biomass(self, abg_biomass):
        # These dimensions are empirically derived allometric relationships for grasses.
        basal_dia_cm = height = canopy_area = np.zeros_like(
            abg_biomass
        )
        filter = np.nonzero(abg_biomass > self.grow_params["abg_biomass"]["min"])
        basal_dia_cm[filter] = self._apply2_allometry_eq_for_xs(abg_biomass[filter], "basal_coeffs")
        basal_width = basal_dia_cm / 100
        height[filter] = self._apply2_allometry_eq_for_ys(basal_dia_cm[filter], "height_coeffs")
        canopy_area[filter] = self._apply3_allometry_eq_for_zs(basal_dia_cm[filter], height[filter], "canopy_area_coeffs")
        shoot_sys_width = self._calc_diameter_from_area(canopy_area)
        return basal_width, height, shoot_sys_width

    def estimate_abg_biomass_from_cover(self, plants):
        # Edit to derive back from percent cover assuming cover for grasses is more reflective of basal diameter than canopy area
        est_abg_biomass = self.allometry._calc_abg_biomass_from_dim(plants["basal_dia"], "basal_dia", cm=True)
        return est_abg_biomass


class Shrub(Habit):
    def __init__(self, params, dt, empirical_coeffs={
            "basal_dia_coeffs": {
                "a": np.log(2500),
                "b": 2.5,
            },
            "canopy_area_coeffs": {
                "a": np.log(120),
                "b": 1.5,
            },
            "height_coeffs": {
                "a": np.log(20),
                "b": 0.5,
            }}):
        green_parts = ("leaf")
        allometry = self._select_allometry_class(params, empirical_coeffs)
        super().__init__(params, allometry, dt, green_parts)
        # Shrub morphology - note these relationships will compound uncertainty
        # - basal diameter is a function of aboveground biomass - in cm
        # - shoot system width is a function of aboveground biomass and basal diameter in m
        # - height is a function of aboveground biomass, basal diameter, amd canopy diameter in m

    def _calc_abg_dims_from_biomass(self, abg_biomass):
        # These dimensions are empirically derived allometric relationships for grasses.
        basal_dia_cm = height = shoot_sys_width = np.zeros_like(
            abg_biomass
        )
        filter = np.nonzero(abg_biomass > self.grow_params["abg_biomass"]["min"])
        basal_dia_cm[filter] = self._apply2_allometry_eq_for_xs(abg_biomass[filter], "basal_coeffs")
        basal_width = basal_dia_cm / 100
        shoot_sys_width[filter] = self._apply3_allometry_eq_for_ys(abg_biomass[filter], basal_dia_cm[filter], abg_biomass[filter], "canopy_area_coeffs")
        height[filter] = self._apply3_allometry_eq_for_ys(shoot_sys_width[filter], abg_biomass[filter], "height_coeffs")
        return basal_width, height, shoot_sys_width


class Tree(Habit):
    def __init__(self, params, dt, empirical_coeffs={}):
        green_parts = ("leaf")
        super().__init__(params, dt, green_parts)


class Vine(Habit):
    def __init__(self, params, dt, empirical_coeffs={}):
        green_parts = ("leaf")
        super().__init__(params, dt, green_parts)
