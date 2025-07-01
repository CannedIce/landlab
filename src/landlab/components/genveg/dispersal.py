import numpy as np

rng = np.random.default_rng()


# Dispersal classes and selection method
class Repro:
    def __init__(self, params):
        grow_params = params["grow_params"]
        disperse_params = params["disperse_params"]
        self.min_size = grow_params["growth_biomass"]["min"]
        self.min_size_for_repro = disperse_params["min_size_dispersal"]


class Clonal(Repro):
    def __init__(self, params):
        disperse_params = params["disperse_params"]
        super().__init__(params)
        self.unit_cost = disperse_params["unit_cost_dispersal"]
        self.max_dist_dispersal = disperse_params["max_dist_runner"]

    def disperse(self, plants):
        """
        This method determines the potential location and cost
        of clonal dispersal. A separate method in
        integrator.py determines if dispersal is successful if the
        potential location is unoccupied.
        """
        max_runner_length = np.zeros_like(plants["root"])
        available_carb = plants["reproductive"] - 2 * self.min_size
        max_runner_length[available_carb > 0] = (
            available_carb[available_carb > 0] / self.unit_cost
        )
        runner_length = rng.uniform(
            low=0.02,
            high=self.max_dist_dispersal,
            size=plants.size,
        )
        pup_azimuth = np.deg2rad(rng.uniform(low=0.01, high=360, size=plants.size))
        filter = np.nonzero(runner_length <= max_runner_length)
        # Save to first position of pup subarray
        plants["dispersal"]["pup_x_loc"][filter] = (
            runner_length[filter] * np.cos(pup_azimuth[filter])
            + plants["x_loc"][filter]
        )
        plants["dispersal"]["pup_y_loc"][filter] = (
            runner_length[filter] * np.sin(pup_azimuth)[filter]
            + plants["y_loc"][filter]
        )
        plants["dispersal"]["pup_cost"][filter] = runner_length[filter] * self.unit_cost
        return plants


class Seed(Repro):
    def __init__(self, params, dt):
        super().__init__(params)
        params = params["disperse_params"]
        self.biomass_to_seedlings = params["mass_to_seedling_rate"]
        self.mean_dispersal_distance = params["mean_seed_distance"]
        self.max_dispersal_distance = params["max_seed_distance"]
        self.dispersal_shape = params["seed_distribution_shape_param"]
        self.seed_size = params["seed_mass"]
        self.seed_efficiency = params["seed_efficiency"]
        self.dt = dt

    def disperse(self, plants, total_biomass):
        """
        This method implements dispersal using an algorithm similar to Vincenot 2016.
        The log-normal distribution shape can be altered to accomodate many patterns of seed
        distribution. Seed production is dependent on plant size and is limited by 
        biomass in the reproductive organ system.
        """
        # Tracking max seeds that can be produced based on reproductive mass available
        max_seeds = plants["reproductive"] * self.seed_size * self.seed_efficiency
        # Partial seedlings are banked to accomodate species with low seedling production rates
        num_seedlings, reserve_part = np.modf(
            (total_biomass * self.biomass_to_seedlings * self.dt.astype(int)),
            out=np.zeros_like(plants["reproductive"]),
            where=(total_biomass >= self.min_size_for_repro)
        )
        num_seedlings[num_seedlings > max_seeds] = max_seeds
        plants["reproductive"] -= num_seedlings * self.seed_size * self.seed_efficiency
        plants["dispersal"]["seedling_reserve"] += reserve_part
        if plants["dispersal"]["seedling_reserve"] >= 1:
            num_seedlings += np.floor(plants["dispersal"]["seedling_reserve"])
            plants["dispersal"]["seedling_reserve"] -= np.floor(plants["dispersal"]["seedling_reserve"])
        # Loop through each plant to assign seedling locations
        for i in range(plants.size):
            dist_from_plant = rng.lognormal(self.mean_dispersal_distance, self.dispersal_shape, num_seedlings[i])
            dist_from_plant[dist_from_plant > self.max_dispersal_distance] = 0.0
            pup_azimuth = np.deg2rad(rng.uniform(low=0.01, high=360, size=num_seedlings[i]))
            plants["dispersal"]["seedling_x_loc"][i][:num_seedlings[i]] = (
                dist_from_plant * np.cos(pup_azimuth) + plants["x_loc"][i]
            )
            plants["dispersal"]["seedling_y_loc"][i][:num_seedlings[i]] = (
                dist_from_plant * np.sin(pup_azimuth) + plants["y_loc"][i]
            )
        return plants


class Random(Repro):
    def __init__(self, params, cell_area, extent, reference):
        super().__init__(params)
        self.daily_col_prob = params["disperse_params"]["daily_colonization_probability"]
        self.cell_area = cell_area
        # Note reference is longitude-latitude and extent is y,x
        self.grid_boundary = (
            reference[1],
            reference[1] + extent[1],
            reference[0],
            reference[0] + extent[0]
        )

    def disperse(self, plants):
        filter = (rng.uniform(0, 1, size=plants["root"].size) > (self.daily_col_prob * self.cell_area))
        plants["dispersal"]["rand_x_loc"][:, 0] = rng.uniform(
            low=self.grid_boundary[0],
            high=self.grid_boundary[1],
            size=plants["root"].size
        )
        plants["dispersal"]["rand_y_loc"][:, 0] = rng.uniform(
            low=self.grid_boundary[2],
            high=self.grid_boundary[3],
            size=plants["root"].size
        )
        plants["dispersal"]["rand_x_loc"][filter] = np.nan
        plants["dispersal"]["rand_y_loc"][filter] = np.nan
        return plants
