import numpy as np
from scipy.optimize import fsolve
rng = np.random.default_rng()
#from landlab.components.genveg import PlantGrowth 

#Duration classes and selection method
class Duration(object):
    def __init__(self, species_grow_params, green_parts):
        self.growdict=species_grow_params
        self.allocation_coeffs=species_grow_params['root_to_leaf_coeffs']+species_grow_params['root_to_stem_coeffs']
        self.green_parts=green_parts

    def set_new_biomass(self, plants):
        print('I create new plants')
        total_biomass_ideal=rng.uniform(low=self.growdict['growth_min_biomass'],high=2*self.growdict['growth_min_biomass'],size=plants.size)
        plants['root_biomass'],plants['leaf_biomass'],plants['stem_biomass']=self._solve_biomass_allocation(total_biomass_ideal, self.allocation_coeffs)
        plants['storage_biomass']=np.zeros_like(plants['root_biomass'])
        plants['repro_biomass']=np.zeros_like(plants['root_biomass'])
        plants['plant_age']=np.zeros_like(plants['root_biomass'])
        return plants
    
    def _solve_biomass_allocation(self, total_biomass, solver_coeffs):
        #Initialize arrays to calculate root, leaf and stem biomass from total
        root=[]
        leaf=[]
        stem=[]
        
        #Loop through grid array
        for total_biomass_in_cell in total_biomass:
            solver_guess = np.full(3,np.log10(total_biomass_in_cell/3))            
            part_biomass_log10=fsolve(self._solverFuncs,solver_guess,(solver_coeffs,total_biomass_in_cell))            
            part_biomass=10**part_biomass_log10
            
            root.append(part_biomass[0])
            leaf.append(part_biomass[1])
            stem.append(part_biomass[2])
        
        #Convert to numpy array
        root=np.array(root)
        leaf=np.array(leaf)
        stem=np.array(stem)      
        return root, leaf, stem

    def _solverFuncs(self,solver_guess,solver_coeffs,total_biomass):
        root_part_log10=solver_guess[0]
        leaf_part_log10=solver_guess[1]
        stem_part_log10=solver_guess[2]
        plant_part_biomass_log10 = np.empty([(3)])

        plant_part_biomass_log10[0]=10**root_part_log10+10**leaf_part_log10+10**stem_part_log10-total_biomass
        plant_part_biomass_log10[1]=solver_coeffs[0]+solver_coeffs[1]*root_part_log10+solver_coeffs[2]*root_part_log10**2-leaf_part_log10
        plant_part_biomass_log10[2]=solver_coeffs[3]+solver_coeffs[4]*root_part_log10+solver_coeffs[5]*root_part_log10**2-stem_part_log10
        
        return plant_part_biomass_log10

class Annual(Duration):
    def __init__(self, species_grow_params):
        green_parts=('root','leaf','stem')
        super().__init__(species_grow_params, green_parts)

    def senesce(self, plants):
        print('I start to lose biomass during senescence periood')
        plants['root_biomass'] = plants['root_biomass'] - (plants['root_biomass'] * 0.02)
        plants['leaf_biomass'] = plants['leaf_biomass'] - (plants['leaf_biomass'] * 0.02)
        plants['stem_biomass'] = plants['stem_biomass'] - (plants['stem_biomass'] * 0.02)
        return plants
    
    def enter_dormancy(self, plants):
        plants['root_biomass'] = np.zeros_like(plants['root_biomass'])
        plants['leaf_biomass'] = np.zeros_like(plants['leaf_biomass'])
        plants['stem_biomass'] = np.zeros_like(plants['stem_biomass']) 
        plants['storage_biomass'] = np.zeros_like(plants['storage_biomass']) 
        plants['repro_biomass'] = np.zeros_like(plants['repro_biomass'])
        return plants
    
    def emerge(self, plants):
        print('I emerge from dormancy')
        plants=self.set_new_biomass(plants)
        return plants

    def set_initial_biomass(self, plants, in_growing_season):
        if in_growing_season:
            plants=self.set_new_biomass(plants)
        return plants
    
class Perennial(Duration):
    def __init__(self, species_grow_params, green_parts):
        super().__init__(species_grow_params, green_parts)
    
    def senesce(self, plants):
        #copied from annual for testing. This needs to be updated
        plants['root_biomass'] = plants['root_biomass'] - (plants['root_biomass'] * 0.02)
        plants['leaf_biomass'] = plants['leaf_biomass'] - (plants['leaf_biomass'] * 0.02)
        plants['stem_biomass'] = plants['stem_biomass'] - (plants['stem_biomass'] * 0.02)
        return plants
    def sum_of_parts(self, plants, parts_list):
        part_sum = np.zeros_like(plants['root_biomass'])
        for part in parts_list:
            part_sum += plants[part]
        return part_sum

    def enter_dormancy(self, plants):
        print('I kill green parts at end of growing season')
        return plants
    
    def set_initial_biomass_all_parts(self, plants, in_growing_season):
        plants['storage_biomass']=rng.uniform(low=self.growdict['plant_part_min']['storage'],high=self.growdict['plant_part_max']['storage'],size=plants.size)
        if in_growing_season:
            plants['plant_age']=rng.uniform(low=0, high=self.max_age, size=plants.size)
            if plants['plant_age'>=self.maturation_age]:
                plants['repro_biomass']=rng.uniform(low=self.growdict['plant_part_min']['reproductive'], high=self.growdict['plant_part_max']['reproductive'], size=plants.size)
        else:
            plants['repro_biomass']=np.full_like(plants['root_biomass'],self.growdict['plant_part_min']['reproductive'])
        total_biomass_ideal=rng.uniform(low=self.growdict['growth_min_biomass'], high=self.growdict['growth_max_biomass'],size=plants.size)
        plants['root_biomass'],plants['leaf_biomass'],plants['stem_biomass']=self._solve_biomass_allocation(total_biomass_ideal, self.allocation_coeffs) 
        return plants

class Evergreen(Perennial):
    def __init__(self, species_grow_params):
        self.keep_green_parts=True

    def emerge(self, plants):
        return plants

    def set_initial_biomass(self, plants, in_growing_season):
        plants=self.set_initial_biomass_all_parts(plants, in_growing_season)
        return plants

class Deciduous(Perennial):
    def __init__(self,species_grow_params, green_parts):
        self.keep_green_parts=False
        super().__init__(species_grow_params, green_parts)
        all_veg_sources=('root','leaf','stem','storage')
        self.persistent_parts=[part for part in all_veg_sources if part not in self.green_parts]

    def senesce(self, plants):
        #copied from annual for testing. This needs to be updated
        print(plants['root_biomass'])
        plants['root_biomass'] = plants['root_biomass'] - (plants['root_biomass'] * 0.02)
        plants['leaf_biomass'] = plants['leaf_biomass'] - (plants['leaf_biomass'] * 0.02)
        plants['stem_biomass'] = plants['stem_biomass'] - (plants['stem_biomass'] * 0.02)
        plants['storage_biomass'] = plants['storage_biomass']
        return plants

    def enter_dormancy(self, plants):
        print('I kill green parts at end of growing season')
        
        new_dormancy_biomass = np.zeros_like(plants['root_biomass'])
        total_mass_to_persistence = np.zeros_like(plants['root_biomass'])
        for part in self.green_parts:
            total_mass_to_persistence += plants[part]
            plants[part] = np.zeros_like(plants[part])

        for part in self.persistent_parts:
            filter = np.where(self.senesce(plants[part]) >= self.set_new_biomass(plants[part]))
            plants[part][filter] = plants[part][filter] + total_mass_to_persistence
        for part in self.persistent_parts:
            filter = np.where(self.senesce(plants[part]) >= self.set_new_biomass(plants[part]))
            new_dormancy_biomass += new_dormancy_biomass[part][filter]
            plants[part][filter] = new_dormancy_biomass 
        for part in self.green_parts:
            plants[part] = np.zeros_like(plants[part])
        return plants  

    
    def emerge(self, plants):
        print('I emerge from dormancy')
        total_mass_persistent_parts=sum(plants[part] for part in self.persistent_parts)
        min_mass_persistent_parts=sum(self.growdict['plant_part_min'][part] for part in self.persistent_parts)
        available_mass=total_mass_persistent_parts-min_mass_persistent_parts
        
        total_mass_new_green=np.zeros_like(plants['root_biomass'])
        new_green_biomass={}
        for part in self.green_parts:
            new_green_biomass[part]=rng.uniform(low=self.growdict['plant_part_min'][part],high=self.growdict['plant_part_min'][part]*2,size=plants.size)
            total_mass_new_green += new_green_biomass[part]
        
        adjusted_total_new_green=np.minimum(available_mass,total_mass_new_green)

        for part in self.green_parts:
            plants[part]=plants[part]+(adjusted_total_new_green/total_mass_new_green)*new_green_biomass[part]

        for part in self.persistent_parts:
            plants[part]=plants[part]-(adjusted_total_new_green*plants[part]/total_mass_persistent_parts)
        return plants

    def set_initial_biomass(self, plants, in_growing_season):
        plants=self.set_initial_biomass_all_parts(plants, in_growing_season)
        if not in_growing_season:
            for part in self.green_parts:
                plants[part]=np.zeros_like(plants[part])
        return plants