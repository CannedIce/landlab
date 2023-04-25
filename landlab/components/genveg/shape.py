
import numpy as np
from scipy import interpolate
#Growth form classes and selection method

#how do these composition classes need to relate to each other? we need to use properties from one composition class to in methods of another. 

class PlantShape(object):
    def __init__(self, morph_params, grow_params):
        self.morph_params=morph_params
        self.grow_params=grow_params
        log10_aspect_ratio=np.array([
            np.log10(self.morph_params['min_abg_aspect_ratio']),
            np.log10(self.morph_params['max_abg_aspect_ratio'])])
        log10_abg_biomass=np.array([
            np.log10((self.grow_params['min_abg_biomass']/1000)*(self.morph_params['max_height']/self.morph_params['min_height'])),
            np.log10(self.grow_params['max_abg_biomass']/1000)
        ])
        self.aspect_ratio_interp_func=interpolate.interp1d(log10_abg_biomass,log10_aspect_ratio,fill_value='extrapolate')
    
    def calc_root_sys_width(self, shoot_sys_width, shoot_sys_height):
        volume = self.calc_crown_volume(shoot_sys_width, shoot_sys_height)
        root_sys_width=0.08+0.24*volume
        return root_sys_width
    
    def calc_crown_area_from_shoot_width(self, shoot_sys_width):
        crown_area=np.pi/4*shoot_sys_width**2
        return crown_area
    
    def calc_vital_volume_from_biomass(self, abg_biomass):
        log_vital_volume=(np.log10(abg_biomass/1000)/np.log10(self.grow_params['max_abg_biomass']/1000))*np.log10(self.morph_params['max_vital_volume'])
        return (10**log_vital_volume)
    
    def calc_crown_volume(self, shoot_sys_width, shoot_sys_height):
        volume=np.pi/4*shoot_sys_width**2*shoot_sys_height
        return volume
    
    def calc_abg_dims_from_biomass(self, abg_biomass):
        #interpolation function not working as expected
        log_aspect_ratio=shoot_sys_width=vital_volume=plant_height=np.zeros_like(abg_biomass)
        filter=np.where(abg_biomass>0)
        log_aspect_ratio[filter]=self.aspect_ratio_interp_func(np.log10(abg_biomass[filter]/1000))
        aspect_ratio=10**log_aspect_ratio
        print(abg_biomass)
        print(aspect_ratio)
        vital_volume[filter]=self.calc_vital_volume_from_biomass(abg_biomass[filter])
        shoot_sys_width=((4*vital_volume)/(np.pi*aspect_ratio))**(1/3)
        plant_height=shoot_sys_width*aspect_ratio

        return shoot_sys_width, plant_height


class Climbing(PlantShape):
    def __init__(self, morph_params, grow_params):
        pass

class Conical(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

    def calc_crown_volume(self, shoot_sys_width, shoot_sys_height):
        volume=np.pi/12*shoot_sys_width**2*shoot_sys_height
        return volume

class Decumbent(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

    def calc_crown_volume(self, shoot_sys_width, shoot_sys_height):
        volume=np.pi/3*shoot_sys_width**2*shoot_sys_height
        return volume

class Erect(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

class Irregular(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

class Oval(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)
    
    def calc_crown_volume(self, shoot_sys_width, shoot_sys_height):
        volume=np.pi/6*shoot_sys_width**2*shoot_sys_height
        return volume

class Prostrate(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

class Rounded(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

    def calc_crown_volume(self, shoot_sys_width, shoot_sys_height):
        volume=np.pi/6*shoot_sys_width**3
        return volume

class Semierect(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)

class Vase(PlantShape):
    def __init__(self, morph_params, grow_params):
        super().__init__(morph_params, grow_params)