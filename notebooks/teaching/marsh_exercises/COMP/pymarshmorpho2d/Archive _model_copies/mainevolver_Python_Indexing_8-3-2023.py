#! /usr/bin/env python
"""pyMarshMorpho2D model."""
import landlab
import numpy
import numpy as np
from landlab import Component
from landlab.components import TidalFlowCalculator
import math
import statistics
import sys
np.set_printoptions(threshold=sys.maxsize)
import warnings
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
warnings.filterwarnings("error")

class mainEvolution(Component):
    """Simulate tidal marsh evolution."""

    _name = "mainEvolution"

    _cite_as = """
    """

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "water_depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Water depth",
        },
        "fully_wet__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Water depth, with depth > 0 everywhere",
        },
        "veg_is_present": {
            "dtype": bool,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "True where marsh vegetation is present",
        },
        "vegetation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "?",
            "mapping": "node",
            "doc": "Some measure of vegetation...?",
        },
        "roughness": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "s/m^1/3",
            "mapping": "node",
            "doc": "Manning roughness coefficient",
        },
        "tidal_flow": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "half tidal range flow velocities",
        },
    }

    def __init__(self, grid,
                 rel_sl_rise_rate = 2.5/1000/365,  # originally 2.74e-6
                 tidal_range = 0.7,
                 tidal_range_for_veg = 0.7,
                 roughness_with_veg = 0.1,
                 roughness_without_veg = 0.02,
                 tidal_period = 12.5/24,
                 model_domain = None
                 ):
        """Initialize the MarshEvolver.

        Parameters
        ----------
        grid : ModelGrid object
            Landlab model grid
        rel_sl_rise_rate : float
            Rate of relative sea-level rise, m/day
        tidal_range : float
            Tidal range, m
        tidal_range_for_veg : float
            Tidal range for vegetation model, m (normally same as tidal range)
        """

        super(mainEvolution, self).__init__(grid)
        self.initialize_output_fields()

        # Get references to fields
        self._elev = self._grid.at_node['topographic__elevation']
        self._water_depth = self._grid.at_node['water_depth']
        self._fully_wet_depth = self._grid.at_node['fully_wet__depth']
        self._veg_is_present = self._grid.at_node['veg_is_present']
        self._vegetation = self._grid.at_node['vegetation']
        self._roughness = self._grid.at_node['roughness']
        self._tidal_flow = self._grid.at_node['tidal_flow']

        # Set parameter values
        self._mean_sea_level = 0.0
        self._rel_sl_rise_rate = rel_sl_rise_rate
        self._tidal_range = tidal_range
        self._tidal_range_for_veg = tidal_range_for_veg
        self._tidal_half_range = tidal_range / 2.0
        self._roughness_with_veg = roughness_with_veg
        self._roughness_without_veg = roughness_without_veg
        self._tidal_period = tidal_period
        self._taucr = 0.2 # a user defined value as part of the mud paramaters in the original code it equals 0.2
        self._taucrVEG = 0.5 # a user defined value as aprt of the vegetation parameters "Critical Shear Stress for vegetated areas. The original value was 0.5
        self._me = 0.1 * pow(10, -4) * 24 * 3600
        self._wsB =  1/1000  # P.wsB=1/1000;%Mud Settling velocity for vegetated area
        self._ws2 = 0.2/1000# P.ws2=0.2/1000;%0.2/1000;% m/s
        self._DoMUD = 1 # base diffusivity of suspended mud [m2/s]. Process not related to tides (e.g. wind and waves, other ocean circulation)
        self._Diffs = 1 # [-]coefficient for tidal dispersion [-]. 1 DO NOT CHANGE
        # self._tidal_period = tidal_period# 12.5/24 # tidal period [day]
        self._rhos = 2650 #sediment density(quartz - mica)[kg / m3]
        self._por2 = 0.7
        self._rbulk = self._rhos*(1-self._por2)
        self._sea_SSC = 60/1000  # 40/1000; %Sea boundary SSC for mud [g/l]
        self._limitdeltaz = 2 # meters
        self._limitmaxup = 1 # meters
        self._min_water_depth = 0.2 # minimuim water depth
        self._Korg = 6/1000/365 # mm/yr organic accretion rate
        self._KBTOT = 0 # something

        # lower and upper limits for veg growth [m]
        # see McKee, K.L., Patrick, W.H., Jr., 1988.
        # these are "dBlo" and "dBup" in matlab original
        self._min_elev_for_veg_growth = -(0.237 * self._tidal_range_for_veg
                                          - 0.092)
        self._max_elev_for_veg_growth = self._tidal_range_for_veg / 2.0

        # default model domain
        self._model_domain = np.reshape(model_domain, np.shape(self._elev))

        # setup index
        self._index = np.array(range(0, len(self._elev)))


    def get_water_depth(self, min_depth=0.01):
        """Calculate the water depth field."""

        depth_at_mean_high_water = np.maximum((-self._elev)
                                              + self._mean_sea_level
                                              + self._tidal_half_range, 0.0)
        self._fully_wet_depth = (0.5 * (depth_at_mean_high_water
                                        + np.maximum(depth_at_mean_high_water
                                                     - self._tidal_range, 0.0)))
        self._hydroperiod = np.minimum(1, np.maximum(0.001, (depth_at_mean_high_water/self._tidal_range)))
        self._fully_wet_depth[self._fully_wet_depth < min_depth] = min_depth
        self._water_depth[:] = self._fully_wet_depth
        self._water_depth[self._elev > (self._mean_sea_level + self._tidal_half_range)] = 0.0
        # self._water_depth[self._water_depth < self._min_water_depth] = self._min_water_depth
        hxxx = self._water_depth[self._water_depth < min_depth]
        relax = 10
        self._water_depth[self._water_depth < min_depth] = np.maximum(self._min_water_depth, min_depth * (1- np.exp(-hxxx * relax))/(1 - np.exp(-min_depth * relax)))


    def update_flow(self):
        from landlab.grid.raster_mappers import map_mean_of_horizontal_links_to_node, map_mean_of_vertical_links_to_node
        from landlab.grid.mappers import map_mean_of_link_nodes_to_link
        from landlab.grid.mappers import map_max_of_node_links_to_node
        """update the flow within the grid"""
        # you need to set the boundaries on the grid
        # for testing we will trun (True, False, True, True
        # self._tidal_flow = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level, tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600, roughness=0.02).calc_tidal_inundation_rate()
        roughnessArray = map_mean_of_link_nodes_to_link(self._grid, self._roughness)
        foo = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level,
                                  tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600,
                                  roughness=roughnessArray, min_water_depth=0.1, scale_velocity=1)
        foo.run_one_step()
        # tst = foo.calc_tidal_inundation_rate()
        # flowHld = map_max_of_node_links_to_node(self._grid, tst)
        self._tidal_flow = foo.calc_tidal_inundation_rate()
        self._ebb_tide_vel = foo._ebb_tide_vel
        self._flood_tide_vel = foo._flood_tide_vel
        self._Uy = map_mean_of_vertical_links_to_node(self._grid, foo._flood_tide_vel) # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor
        self._Ux = map_mean_of_horizontal_links_to_node(self._grid, foo._flood_tide_vel)  # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor

        # flip and mirror
        # self._Uy = (np.flip(np.reshape(self._Uy, self._grid.shape), axis=0))
        # self._Ux = (np.flip(np.reshape(self._Ux, self._grid.shape), axis=0))

        # test
        self._Ux = self._Ux * -1 # 0
        self._Uy = self._Uy * -1 # 0

        # fill in the outer edges of the grid.
        # Landlab leaves these edges as zero due to the core matrix calculations however this causes issues with the matrix inversion step
        # in the morphology calculator
        gridNum = np.reshape(self._index, (self._grid.shape))
        self._Ux[gridNum[:,0]] = self._Ux[gridNum[:,1]] # fill in the left column with data from the next column over
        self._Ux[gridNum[:, self._grid.shape[1]-1]] = self._Ux[gridNum[:, self._grid.shape[1]-2]]
        self._Ux[gridNum[0,:]] = self._Ux[gridNum[1,:]]
        self._Ux[gridNum[-2, :]] = self._Ux[gridNum[-1, :]] # replace the 2nd to bottom row with the bottom row

        # do the same thing for Y
        self._Uy[gridNum[:,0]] = self._Uy[gridNum[:,1]] # fill in the left column with data from the next column over
        self._Uy[gridNum[:, self._grid.shape[1]-1]] = self._Uy[gridNum[:, self._grid.shape[1]-2]]
        self._Uy[gridNum[0,:]] = self._Uy[gridNum[1,:]]
        self._Uy[gridNum[-2, :]] = self._Uy[gridNum[-1, :]]

        self._U = np.sqrt((self._Ux**2) + (self._Uy**2)) # not sure about the units but this matches the matlab output.


    def update_morphology(self, dt):
        from landlab.grid.mappers import map_min_of_link_nodes_to_link
        from landlab.grid.mappers import map_min_of_node_links_to_node
        """Update morphology
        This is currently only the simplest version (without ponding, wave errsion, etc.)"""

        # Thomas' NOTES.
        # The update_morphology function is taking the place of both the "TotalsedimenterosionMUDsine.m" script and the "sedtran.m" script.
        # The python implementation of TotalsedimenterosionMUDsine.m has been completed and validated against the matlab script.
        # Currently the issue is with the sedtran.m implementation.

        ###### erosion base values for all types of erosion (currently only one type of erosion being used.

        # variables to figure out
        # taucr = a user defined value as part of the mud paramaters in the original code it equals 0.2
        # taucrVEG = a user defined value as aprt of the vegetation parameters "Critical Shear Stress for vegetated areas. The original value was 0.5
        # MANN = self._roughness
        # VEG = self._vegetation
        # U = Is a returned value from the tidalFlow Function.  I think it is the tidal velocity.
        # me = P.me=0.1*10^-4*24*3600# a defined value in the parameters for mud P.me=0.1*10^-4*24*3600;  %per day!!!
        # h = Is water depth self._water_depth
        # we are going to have to define a domain. someway.

        # record the starting elevation
        origz = np.copy(self._elev)

        # exclude cells that cant get sediment as they are above water.
        # p = np.where(self._model_domain > 0, np.where(self._water_depth <= 0, False, True),
        #              False)  # original working code. however it didn't remove above water areas from the calculation.
        p = np.where(self._model_domain > 0) #, np.where(self._water_depth <=self._min_water_depth, False, True), False) # double check this logic.  This seems backwards

        fUpeak = math.pi / 2
        taucro = self._elev * 0 + self._taucr
        taucro[self._veg_is_present == 1] = self._taucrVEG


        # tidal current erosion ################################################################################################
        ncyc = 10
        E = 0
        for i in range(-1, ncyc):
            i=i+1
            Utide = self._U * fUpeak * math.sin(i / ncyc * math.pi / 2)
            # print(math.sin(i / ncyc * math.pi / 2))
            # print(Utide[1])
            try:
                watPow = self._water_depth**(-1 / 3) # changed to fully wetted depth
            # if watPow.any() == np.inf:
            #     print("There are infs")
            except:
                 print("Zero water depth detected")
            #     watPow = self._water_depth ** (-1 / 3)
            watPow[watPow == np.inf] = 0 # set inf numbers to zero
            tauC = 1030 * 9.81 * self._roughness**(2) * watPow * Utide**2
            E = E + (1 / (ncyc + 1)) * self._me * (np.sqrt(1 + (tauC/taucro)**2) - 1)
        E[self._model_domain == 2] = 0
        E[E == np.inf] = 0 # clean out any infinite values


        # ## CURRENT-DRIVEN TRANSPORT (Tide and River)######################################################
        # Advection-Diffusion Sediment transport
        WS = (self._elev * 0) + self._ws2
        WS[self._veg_is_present == 1] = self._wsB
        # WS(S==1)=ws2# SHOULD NOT BE NECEEARY BECUASE VEG alreeady set equal to zero where S=1 (see above).  ->Do not add the vegetation settling velocity in the ponds! %WS(S==1)=0.000000000001%E2(S==1)=0

        ###################### Sedtran ############################################
        # This is all modified by Thomas Huff from the original MatLab code
        dx = 5 * 2 # this is possibly a velocity or erosion modifier.
        # (self._DoMUD * 24 * 3600 + self._DiffS * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dxx = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dyy = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Uy * self._Uy) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth


        N, M = self._grid.shape
        S = np.zeros((N * M))
        ilog = []
        jlog = []
        s = []
        for k in [N, -1, 1, -N]: # calculate the gradianets between cells in the x and y direction
            # print(k)
            tmp = self._index[p]
            row, col = np.unravel_index(tmp, shape=(N, M)) # sort this out.
            # indTemp = np.reshape(self._index, (N, M))
            if k == N:
                a = np.where(col + 1 < M, True, False)
                q = tmp + 1
            if k == -N:
                a = np.where(col - 1 >= 0, True, False)
                q = tmp - 1 # originally tmp was tmp[p]
            if k == -1:
                a = np.where(row - 1 >= 0, True, False)
                q = tmp - M
            if k == 1:
                a = np.where(row + 1 < N, True, False)
                q = tmp + M


            # numerical array that cooresponds to the index values covered by water
            parray = self._index[p]

            # filter a added 8-1-2023
            # remove no land cells from a
            # a[parray[a]] = np.where(self._model_domain[parray[a]] != 1,
            #              np.where(self._model_domain[parray[a]] != 2,
            #                       np.where(self._model_domain[parray[a]] == 10, True, False), True), True)
            a[q[a]] = np.where(a[q[a]] == True, np.where(self._model_domain[q[a]] != 1,
                         np.where(self._model_domain[q[a]] != 2,
                                  np.where(self._model_domain[q[a]] == 10, True, False), True), True), False)


            # calculate DD value
            if (k == N) | (k == -N):
                D = Dyy
            else:
                D = Dxx

            numeric = 1 # This will need to be set by a hyperparameter in the future
            if numeric == 1:
                try:
                    DD = (D[parray[a]] + D[q[a]])/2
                except:
                    print("There was an issues with the DD calculation")
            else:
                DD = np.minimum(D[parray[a]], D[q[a]])
            try:
                value = DD / self._water_depth[parray[a]] / self._hydroperiod[parray[a]]
            except:
                print("There was an issue with the value calculation")

            # calculate a mask for flux entering or leaving a given node.
            Fin = np.copy(self._elev)
            Fin[:] = 0
            Fout = np.copy(Fin)
            # There are some challenges with getting the indexes to match and this is how I am getting around it.
            Fin[parray[a][np.in1d(parray[a], self._index[self._model_domain == 1])]] = 1
            tmpInd = q[a]
            Fout[self._index[tmpInd[self._model_domain[q[a]] == 1]]] = 1



            ########################################################################################################
            # matlab code currently not implemented
            # %river flow component
            # if computeriver==1
            # if (k==N);UR=URy(p(a));up=find(UR>0);F=UR(up);end %East-west
            # if (k==-N);UR=URy(p(a));up=find(UR<0);F=-UR(up);end
            # if (k==1);UR=URx(p(a));up=find(UR>0);F=UR(up);end  %North-south
            # if (k==-1);UR=URx(p(a));up=find(UR<0);F=-UR(up);end
            # value(up)=value(up)+F*3600*24/dx;
            # end
            #
            #######################################################################################################
            # residualcurrents = 1
            # if residualcurrents == 1:
            #     # print("Calculating residual currents")
            #     # tidal residual currents and transport.
            #     # (I imposed no residual currents are the open boundary to avoid
            #     # calculating the fluxes to get the mass balance at 100%)
            #     if k == N:
            #         UR = self._Ux[parray[a]]
            #         up = np.where(UR > 0, True, False)
            #         F = UR[up]
            #     if k == -N:
            #         UR = self._Ux[parray[a]]
            #         up = np.where(UR < 0, True, False)
            #         F = -UR[up]
            #     if k == 1:
            #         UR = self._Uy[parray[a]]
            #         up = np.where(UR > 0, True, False)
            #         F = UR[up]
            #     if k == -1:
            #         UR = self._Uy[parray[a]]
            #         up = np.where(UR < 0, True, False)
            #         F = -UR[up]
            #     value[up] = value[up] + F * 3600 * 24 / dx
            #######################################################################################################
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # %%%%%%%%%%%%%%%%%%%%%BED EVOLUTION DIVERGENCE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # fix this section to add in bed creep ponds... Pond creep seems to do more than just for ponds.....
            # EVOLUTION OF z
            # z=znew;  # NEED TO RE-UDPATED FROM THE Y1
            # Qs2=E / (ws2*3600*24) * self._U * np.maximum(0,self._water_depth) # kg/m/s %[hcorrected]=getwaterdepth(Trange,Hsurge,msl,z,kro,hpRIV);  %VEG=(z-msl)>dBlo;
            # znew=bedcreepponds(z,A,Active,A*0+1,crMUD,crMARSH,dx,dt,VEG,S,Qs2,rbulk2,alphaMUD)  # MUD CREEP  MARSH
            # deltaz=z-znew
            # deltaz(A==2)=0 # DO NOT UPDATE THE BOUNDARY
            # z=z-deltaz

            # build a list of values entering a node
            S[parray[a]] = S[parray[a]] + (value * Fin[parray[a]])
            try:
                ilog = list(ilog) + list(q[a])
                jlog = list(jlog) + list(parray[a])
            except:
                print("There was an issue with ilog or jlog creation")
            # build a list of values exiting the node
            s = list(s) + list(-value * Fout[q[a]])
        settling = np.copy(self._elev) * 0
        settling[p] = 24 * 3600 * WS[p] / self._water_depth[p] * self._hydroperiod[p]
        settling[self._index[self._model_domain == 2]] = 0
        S[self._index[self._model_domain == 2]] = 1
        aa = self._index[self._model_domain == 10]
        S[parray[aa]] = 1 # impose the b.c.
        del(aa)

        ################################################################################
        # %sea boundary
        # a=find(A(p)==2);%find the co b.c.
        # settling(a)=0;%do not settle in the b.c. (to not change the SSC)
        # S(p(a))=1;%to impose the b.c.
        #
        # %river boundary
        # if MUD==1; %if mud, handlew this as an imposed SSC
        # a=find(A(p)==10);%find the co b.c.
        # settling(a)=0;%do not settle in the b.c. (to not change the SSC)
        # S(p(a))=1;%to impose the b.c.
        # end
        #################################################################################

        try:
            ilog = list(ilog) + list(parray)
            jlog = list(jlog) + list(parray)
            s = list(s) + list(S[p] + settling[p])
        except:
            print("Error with ilog or jlog")
        ds2 = sparse.csc_array((s, (ilog, jlog)), shape = (N*M, N*M)) #, shape = (N, M))
        rhs = np.array(E)
        rhs[:] = 0
        rhs[p] = E[p]
        aa = self._index[self._model_domain == 2]
        try:
            rhs[aa] = (60/1000) * self._water_depth[aa] * self._hydroperiod[aa]
        except:
            print('rhs creation issue')
        MUD = 1 # This is just for testing.  Make sure to remove this later on
        if MUD == 1:
            aa = self._index[self._model_domain == 10]
            rhs[aa] = np.nan * self._water_depth[aa] * self._hydroperiod[aa]
            del(aa)
        try:
            P = scipy.sparse.linalg.spsolve(ds2, rhs) # was working with .lsqr
        except:
            print("Matrix solution was singular. Reverting to lsqr to solve matrix inversion")
            P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim = 5000)[0]

        SSM = np.array(P) # apply the P value to the SSM varible
        EmD = np.copy(self._elev)
        EmD[:] = 0 # zero out a new array
        EmD[p] = (E[p] - SSM[p]*settling[p])/self._rbulk
        EmD[self._model_domain == 2] = 0 # impose boundary conditions
        self._elev = self._elev - (dt * EmD) # calculate the change in elevation.
        self._elev[self._model_domain == 2] = -10 # set the boundary areas to -10
        self._elev[self._model_domain == 50] = 100  # set the boundary areas to -10


        #######################################################################################################
        # add organic accretion
        noPond = np.where(S == 0, True, False)
        self._vegetation[noPond] = self._vegetation[noPond] * S[noPond]
        AccreteOrganic = 1
        if AccreteOrganic == 1:
            self._elev = self._elev + self._vegetation * self._Korg * dt # put organic on mud!!!
        self._KBTOT = self._KBTOT + sum(self._vegetation[self._model_domain == 1]) * self._Korg * dt
        #######################################################################################################


        # return the maxdeltaz and maxup values for the loop computation in "run_one_step
        mxDelta = np.percentile(abs(self._elev - origz), 99)
        mxchup = np.maximum(0, np.maximum(0, self._elev-origz)) * ((self._elev)>(self._mean_sea_level+self._tidal_range/2))
        mxUp = np.max(mxchup[:])
        return(mxDelta, mxUp)

        # np.reshape(self._elev, (500,300))

        # organic accretion has not been added yet.
    def update_vegetation(self):
        """Update vegetation."""
        height_above_msl = self._elev - self._mean_sea_level
        self._veg_is_present[:] = (height_above_msl
                                   > self._min_elev_for_veg_growth)
        self._vegetation = (4 * (height_above_msl
                                 - self._max_elev_for_veg_growth)
                            * (self._min_elev_for_veg_growth
                               - height_above_msl)
                            / (self._min_elev_for_veg_growth
                               - self._max_elev_for_veg_growth) ** 2)
        self._vegetation[height_above_msl > self._max_elev_for_veg_growth] = 0.0
        self._vegetation[height_above_msl < self._min_elev_for_veg_growth] = 0.0

    def update_roughness(self):
        """Update Manning's n values."""
        self._roughness[:] = self._roughness_without_veg
        self._roughness[self._veg_is_present] = self._roughness_with_veg

    def run_one_step(self, timeStep, round, model_domain):
        """Advance in time."""
        # print("\n")
        # print(f'Working on round {round}')

        if round == 0:
            t = 1
            dto = 0.00001
        else:
            dto = 365
        dti = 0
        dt = dto
        while dti < dto:
            firstattempt=1
            maxdeltaz = self._limitdeltaz + 1
            maxup = self._limitmaxup + 1

            while maxdeltaz > self._limitdeltaz or maxup > self._limitmaxup: # changed | to or
                if firstattempt != 1:
                    dt = dt / 2 * np.minimum((self._limitdeltaz / maxdeltaz), (self._limitmaxup / maxup))
                    # print(f'dt bing reduced. dt value is now {dt}')
                firstattempt = 0
                if round <= 1:
                    dt = np.minimum(0.5*365, dt)
                    # print(f' -----> updated dt value {dt}')
                # Update sea level
                self._mean_sea_level = self._mean_sea_level + (self._rel_sl_rise_rate * dt)

                # water depth
                self.get_water_depth()

                # vegetation
                self.update_vegetation()

                # roughness
                self.update_roughness()

                # tidal_flow
                self.update_flow()
                # bed morphology
                maxdeltaz, maxup = self.update_morphology(dt)
            dti = dti+dt
            dt = np.minimum((dt*2), (np.maximum(0, dto-dti)))
            # print(f'<---- updated dti value {dti}')
            # print(f'<---- updated dt value {dt}')
