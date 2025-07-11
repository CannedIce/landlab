"""
Growth component of GenVeg - this is the main driver of vegetation growth and
is driven by a photosynthesis model. Vegetation growth depends on the availability
of carbohydrate produced by photosynthetically active plant parts.
"""

import numpy as np

from landlab.components.genveg.species import Species
from landlab.data_record import DataRecord

rng = np.random.default_rng()


class PlantGrowth(Species):
    """
    Add Intro Stuff here
    _name = "PlantGrowth"
    _unit_agnostic = False
    _cite_as =
    @article{piercygv,
        author = {
            Piercy, C.D.;
            Swannack, T.M.;
            Carrillo, C.C.;
            Russ, E.R.;
            Charbonneau, B. M.;
    }
    #Add all variables to be saved or chosen as outputs here
    _info = {
        "vegetation__total_biomass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units":"g",
            "mapping":"cell",
            "doc": "Total plant biomass for the plant class at the end of the time step"
        },
    }
    """

    def __init__(
        self,
        grid,
        dt,
        rel_time,
        _current_jday,
        species_params={
            "duration_params": {
                "growing_season_start": 91,
                "growing_season_end": 290,
                "senescence_start": 228,
            },
            "grow_params": {
                "respiration_coefficient": [0.015, 0.015, 0.03],
                "glucose_requirement": [1.444, 1.513, 1.463],
                "k_light_extinct": 0.02,
                "light_half_sat": 9,
                "p_max": 0.055,
                "root_to_leaf_coeffs": [0.031, 0.951, 0],
                "root_to_stem_coeffs": [-0.107, 1.098, 0.0216],
                "plant_part_min": [0.01, 0.1, 0.5],
            },
            "mort_params": {
                "s1_days": 365,
                "s1_name": "Mortality factor",
                "s1_pred": [1, 2, 3, 4],
                "s1_rate": [0, 0.1, 0.9, 1],
                "s1_weight": [1000, 1, 1, 1000],
            },
            "plant_factors": {
                "species": "Corn",
                "growth_form": 1,
                "monocot_dicot": "monocot",
                "angio_gymno": "angiosperm",
                "annual_perennial": "annual",
                "p_type": "C3",
            },
            "size_params": {
                "max_height_stem": 2.5,
                "max_mass_stem": 72,
                "max_n_stems": 3,
                "max_plant_density": 1,
            },
            "stor_params": {"r_wint_die": 0.25, "r_wint_stor": 0.25},
        },
        **kwargs,
    ):
        """Instantiate PlantGrowth
        Parameters
        ----------
        grid: RasterModelGrid
            A Landlab ModelGrid

        dt: NumPy time delta, required,
            time step interval

        rel_time: int, required,
            number of time steps elapsed

        _current_jday: int, required
            day of the year assuming Jan 1 is 1

        **kwargs to send to init
            plants: Numpy structured array of individual plants, optional
                with columns
                species: string, plant species names
                pid: int, plant ID
                cell_index: int, index of cell location on grid
                root_biomass: float, title='root', plant live root biomass in g
                leaf_biomass: float, title='stem', plant live leaf biomass in g
                stem_biomass: float, title='stem', plant live stem biomass in g
                storage_biomass: float, title='storage', plant live stem biomass in g
                repro_biomass: float, title='reproductive',
                                plant live reproductive biomass in g
                plant_age: int, plant age in days

            species_params: dict, optional,
                a nested dictionary of named vegetation parameters for the
                species or community and process of interest with below sub-dictionaries
                plant_factors: dict, required,
                    dictionary of plant characteristics describing the
                    species or community of interest with below keys
                    species: string, required,
                        name of species or community used to identify plant
                    growth_form: string, required,
                        USDA plant growth habit,
                        graminoid, forb/herb, shrub, tree, vine
                    monocot_dicot: string, required,
                        should be monocot or dicot
                    angio_gymno: string, required,
                        should be angiosperm or gymnosperm
                    annual_perennial: string, required,
                        plant growth duration, annual (1 year) or
                        perennial (multiple years)
                    p_type: string, required,
                        photosythesis type, either 'C3', 'C4', or 'CAM'
                    leaf_retention: string, required,
                        evergreen or deciduous (annuals are deciduous)
                duration_params: dict, required,
                    dictionary of parameters defining the growing season,
                    growing_season_start: int, required,
                        growing season start day of year,
                        must be between 1-365
                    growing_season_end: int, required,
                        growing season end day of year,
                        must be between 1-365
                    senesecence_start: int, required,
                        start of senescence period after plant reaches peak biomass,
                        must be between gs_start and gs_end
                grow_params: dict, required,
                    dictionary of paramaters required to simulate plant growth

                    respiration_coefficient: dict, required,
                        respiration coefficient with keys
                            'root': float
                            'leaf': float
                            'stem': float
                            'reproductive': float
                    glucose_requirements: dict, required,
                        glucose requirement
                    le_k: float, required,
                        light extinction coefficient
                    hi: float, required,
                        something
                    p_max: float, required,
                        maximum photosyntehtic output
        """
        # Initialize species object to get correct species parameter list
        self._grid = grid
        self.dt = dt
        super().__init__(species_params, self._grid, self.dt)
        self.species_name = self.species_plant_factors["species"]
        self.variable_map = {
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
        self.time_ind = 1
        event_flags = self.set_event_flags(_current_jday)
        _in_growing_season = event_flags.pop("_in_growing_season")
        max_plants = np.round(
            self._grid.number_of_cells
            * self.species_morph_params["max_plant_density"]
            * self._grid.area_of_cell
        ).astype(int)
        print("Max plants")
        print(max_plants)
        self.no_data_scalar = (
            "N/A",
            999999,
            999999,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            999999,
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
            999999,
        )
        self.dtypes = [
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
        mask_scalar = 1
        empty_list = []
        mask = []
        for _ in range(max_plants[0]):
            empty_list.append(self.no_data_scalar)
            mask.append(mask_scalar)
        self.plants = np.ma.array(empty_list, mask=mask, dtype=self.dtypes)
        self.plants[:] = self.no_data_scalar
        self.plants[:] = np.ma.masked
        species_cover = kwargs.get(
            "species_cover",
            np.zeros_like(self._grid["cell"]["vegetation__total_biomass"])
        )
        try:
            init_plants = kwargs.get(
                "plant_array",
                self._init_plants_from_grid(
                    _in_growing_season, species_cover
                ),
            )
        except KeyError:
            msg = "GenVeg requires a pre-populated plant array or a species cover."
            raise ValueError(msg)
        self.n_plants = init_plants.size
        self.plants[: self.n_plants] = init_plants

        self.call = []
        # Create empty Datarecord to store plant data
        # Instantiate data record
        data_vars = {}
        for array_var, record_var in self.variable_map.items():
            data = np.reshape(self.plants[array_var][: self.n_plants], (self.n_plants, 1))
            data_vars[record_var] = (["item_id", "time"], data)
        self.record_plants = DataRecord(
            self._grid,
            time=[rel_time],
            items={
                "grid_element": np.repeat(["cell"], self.n_plants).reshape(
                    self.n_plants, 1
                ),
                "element_id": np.reshape(
                    self.plants["cell_index"][: self.n_plants],
                    (self.n_plants, 1),
                ),
            },
            data_vars=data_vars,
            attrs={
                "vegetation__species": "species name, string",
                "vegetation__root_biomass": "g",
                "vegetation__leaf_biomass": "g",
                "vegetation__stem_biomass": "g",
                "vegetation__repro_biomass": "g",
                "vegetation__dead_root_biomass": "g",
                "vegetation__dead_leaf_biomass": "g",
                "vegetation__dead_stem_biomass": "g",
                "vegetation__dead_repro_biomass": "g",
                "vegetation__total_leaf_area": "sq m",
                "vegetation__live_leaf_area": "sq m",
                "vegetation__shoot_sys_width": "m",
                "vegetation__basal_diameter": "m",
                "vegetation__plant_height": "m",
                "vegetation__root_sys_diameter": "m",
                "vegetation__root_depth": "m",
                "vegetation__plant_age": "days",
                "vegetation__stem_count": "#",
            },
        )
        self.plants["item_id"][: self.n_plants] = self.record_plants.item_coordinates

    def species_plants(self):
        unmasked_rows = np.nonzero(self.plants["pid"] != 999999)
        return self.plants[unmasked_rows]

    def update_plants(self, var_names, pids, var_vals, subarray=None):
        if subarray is None:
            for idx, var_name in enumerate(var_names):
                self.plants[var_name][np.isin(self.plants["pid"], pids)] = var_vals[idx]
            return self.plants
        else:
            for idx, var_name in enumerate(var_names):
                self.plants[subarray][var_name][np.isin(self.plants["pid"], pids)] = var_vals[idx]
            return self.plants

    def add_new_plants(self, new_plants_list, _rel_time):
        # Reassess this. We need the INDEX of the last nanmax PID
        last_pid = np.ma.max(self.plants["pid"])
        pids = np.arange(last_pid + 1, last_pid + 1 + new_plants_list.size)
        new_plants_list["pid"] = pids
        new_plants_list["item_id"] = pids
        (n_new_plants,) = new_plants_list.shape
        start_index = np.flatnonzero(self.plants["pid"] == last_pid).astype(int) + 1
        end_index = n_new_plants + start_index[0]
        self.plants[start_index[0] : end_index] = new_plants_list
        self.n_plants += n_new_plants
        self.record_plants.add_item(
            time=np.array([_rel_time]),
            new_item={
                "grid_element": np.repeat(["cell"], n_new_plants).reshape(
                    n_new_plants, 1
                ),
                "element_id": np.reshape(
                    self.plants["cell_index"][start_index[0] : end_index],
                    (n_new_plants, 1),
                ),
            },
            new_item_spec={
                "vegetation__species": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["species"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__root_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["root_biomass"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__leaf_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["leaf_biomass"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__stem_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["stem_biomass"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__repro_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["repro_biomass"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__dead_root_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["dead_root"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__dead_leaf_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["dead_leaf"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__dead_stem_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["dead_stem"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__dead_repro_biomass": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["dead_reproductive"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__shoot_sys_width": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["shoot_sys_width"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__total_leaf_area": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["total_leaf_area"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
                "vegetation__plant_age": (
                    ["item_id", "time"],
                    np.reshape(
                        self.plants["plant_age"][start_index[0] : end_index],
                        (n_new_plants, 1),
                    ),
                ),
            },
        )
        return self.plants

    def _grow(self, _current_jday, _grid_par, _grid_relative_water_content, _grid_relative_saturation):
        # This is the primary method in PlantGrowth and is run within each
        # GenVeg run_one_step at each timestep. This method applies new environmental
        # conditions daily from the grid, determines what processes run, and implements
        # them in order to update the plant array.

        # set up shorthand aliases and reset
        # look at checking to see if we can use recordmask here
        _last_biomass = self.plants[~self.plants["pid"].mask].copy()
        _new_biomass = self.plants[~self.plants["pid"].mask]
        # Decide what processes happen today
        event_flags = self.set_event_flags(_current_jday)
        processes = {
            "_in_growing_season": self.photosynthesize,
            "_in_senescence_period": self.senesce,
            "_in_reproductive_period": self.disperse,
            "_is_emergence_day": self.emerge,
            "_is_dormant_day": self.enter_dormancy,
        }

        # Run mortality and decompose litter each day
        _new_biomass = self.mortality(_new_biomass)
        _new_biomass = self.litter_decomp(_new_biomass)
        # Limit growth processes only to live plants
        _total_biomass = self.sum_plant_parts(_new_biomass, parts="total")
        filter = np.nonzero(_total_biomass > 0.0)
        _new_live_biomass = _new_biomass[filter]

        # calculate variables needed to run plant processes
        _par = _grid_par[_last_biomass["cell_index"]][filter]
        _relative_water_content = _grid_relative_water_content[
            _last_biomass["cell_index"]
        ][filter]
        _rel_sat = _grid_relative_saturation[_last_biomass["cell_index"]][filter]
        _min_temperature = self._grid["cell"]["air__min_temperature_C"][
            _last_biomass["cell_index"]
        ][filter]
        _max_temperature = self._grid["cell"]["air__max_temperature_C"][
            _last_biomass["cell_index"]
        ][filter]
        _cell_lai = self._grid["cell"]["vegetation__leaf_area_index"][
            _last_biomass["cell_index"]
        ][filter]
        _new_live_biomass = self.respire(
            _min_temperature, _max_temperature, _rel_sat, _new_live_biomass
        )

        # Change this so for positive delta_tot we allocate by size and
        if event_flags["_in_growing_season"]:
            carb_generated_photo = processes["_in_growing_season"](
                _par,
                _min_temperature,
                _max_temperature,
                _cell_lai,
                _relative_water_content,
                _new_live_biomass,
                _current_jday,
            )

            # Future add turnover rate
            _new_live_biomass = self.allocate_biomass_dynamically(
                _new_live_biomass, carb_generated_photo
            )
        _new_live_biomass = self.kill_small_plants(_new_live_biomass)
        event_flags.pop("_in_growing_season")
        # Run all other processes that need to occur
        for process in event_flags.items():
            if process[1]:
                _new_live_biomass = processes[process[0]](
                    _new_live_biomass, _current_jday
                )

        _new_live_biomass["plant_age"] += self.dt.astype(float) * np.ones_like(
            _new_live_biomass["plant_age"]
        )
        _new_biomass[filter] = _new_live_biomass
        _new_biomass = self.update_dead_biomass(_new_biomass, _last_biomass)
        _new_biomass = self.update_morphology(_new_biomass)
        self.plants[~self.plants["pid"].mask] = _new_biomass
        self.plants, self.n_plants = self.remove_plants()

    def _init_plants_from_grid(self, in_growing_season, species_cover):
        """
        This method initializes the plants in the PlantGrowth class
        from the vegetation fields stored on the grid. This method
        is only called if no initial plant array is parameterized
        as part of the PlantGrowth initialization.
        Required parameters are the cell plant cover and a boolean
        indicating if the plants are in the active growing season.
        """
        pidval = 0
        init_plants = self.plants.copy()
        # Loop through grid cells
        for cell_index in range(self._grid.number_of_cells):
            species = self.species_plant_factors["species"]
            cell_cover = species_cover[cell_index]
            cover_area = (
                cell_cover * self._grid.area_of_cell[cell_index] * 0.907
            )
            pidval, init_plants = self.set_initial_cover(cover_area, species, pidval, cell_index, init_plants)
        init_plants = self.set_initial_biomass(init_plants, in_growing_season)
        return init_plants

    def set_event_flags(self, _current_jday):
        """
        This method sets event flags so required processes are run based
        on the day of year.
        """
        durationdict = self.species_duration_params
        flags_to_test = {
            "_in_growing_season": bool(
                (
                    (_current_jday > durationdict["growing_season_start"])
                    & (_current_jday < durationdict["growing_season_end"])
                )
                or (np.isnan(durationdict["growing_season_start"])
                    or np.isnan(durationdict["growing_season_end"])
                    )
            ),
            "_is_emergence_day": bool(
                _current_jday == durationdict["growing_season_start"]
            ),
            "_in_reproductive_period": bool(
                (_current_jday >= durationdict["reproduction_start"])
                & (_current_jday < durationdict["reproduction_end"])
            ),
            "_in_senescence_period": bool(
                (_current_jday >= durationdict["senescence_start"])
                & (_current_jday < durationdict["growing_season_end"])
            ),
            "_is_dormant_day": bool(
                _current_jday == durationdict["growing_season_end"]
            ),
        }
        return flags_to_test

    def kill_small_plants(self, _new_biomass):
        # This method moved live biomass to dead biomass is the plant
        # is too small to grow.
        min_size = self.species_grow_params["min_growth_biomass"]
        total_biomass = self.sum_plant_parts(_new_biomass, parts="growth")
        dead_plants = np.nonzero(total_biomass < min_size)
        if dead_plants[0].size > 0:
            print(str(dead_plants[0].size) + " were too small to survive")

        for part in self.all_parts:
            _new_biomass[part][dead_plants][
                np.isnan(_new_biomass[part][dead_plants])
                | (_new_biomass[part][dead_plants] < 0)
            ] = 0.0
            _new_biomass["dead_" + str(part)][dead_plants] += _new_biomass[part][
                dead_plants
            ]
            _new_biomass[part][dead_plants] = 0.0
            _new_biomass[part][_new_biomass[part] < 0] = 0.0
        return _new_biomass

    def remove_plants(self):
        # Plants that have too little dead biomass remaining to track
        # are removed from the plant array and no longer tracked.
        min_size_dead = 0.1
        min_size_live = self.species_grow_params["min_growth_biomass"]
        total_live_biomass = self.sum_plant_parts(self.plants, parts="growth")
        total_dead_biomass = self.sum_plant_parts(self.plants, parts="dead")
        remove_plants = np.flatnonzero(
            (total_dead_biomass < min_size_dead) & (total_live_biomass < min_size_live)
        )
        self.plants[remove_plants] = self.no_data_scalar
        self.plants[remove_plants] = np.ma.masked
        remove_array_length = np.size(remove_plants)
        self.n_plants -= remove_array_length
        return self.plants, self.n_plants

    def save_plant_output(self, rel_time, opt_save_vars):
        """
        This method saves plant properties at the required time step
        Args:
            rel_time (float): Number of days since start of simulation
            opt_save_vars (list of strings): Optional variables from plant array
                                             to save to record array
        Returns: no explicit return; updates record array with plant data at current
                 relative time step

        """
        self.record_plants.add_record(time=np.array([rel_time]))
        self.record_plants.ffill_grid_element_and_id()
        save_vars = ["species", "root", "leaf", "stem", "reproductive"]
        save_vars.extend(opt_save_vars)
        item_ids = self.plants["item_id"][~self.plants["item_id"].mask]
        for var in save_vars:
            print(var)
            print(self.variable_map[var])
            self.record_plants.dataset[self.variable_map[var]].values[
                item_ids, self.time_ind
            ] = self.plants[var][~self.plants["item_id"].mask]
        self.time_ind += 1
