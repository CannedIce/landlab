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
                 rel_sl_rise_rate=0.004 / 365,  # originally 2.74e-6
                 tidal_range=0.7,
                 tidal_range_for_veg = 0.7,
                 roughness_with_veg=0.1,
                 roughness_without_veg=0.02,
                 tidal_period = 12.5/24
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

        # grid boundaries
        self._grid.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                                       top_is_closed= True,
                                                       left_is_closed=True,
                                                       bottom_is_closed=False)


        # lower and upper limits for veg growth [m]
        # see McKee, K.L., Patrick, W.H., Jr., 1988.
        # these are "dBlo" and "dBup" in matlab original
        self._min_elev_for_veg_growth = -(0.237 * self._tidal_range_for_veg
                                          - 0.092)
        self._max_elev_for_veg_growth = self._tidal_range_for_veg / 2.0

        # default model domain
        self._model_domain = np.copy(self._elev)
        self._model_domain[self._model_domain == -10] = 2
        self._model_domain[self._model_domain == -1] = 1

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


    def update_flow(self):
        from landlab.grid.raster_mappers import map_mean_of_horizontal_links_to_node, map_mean_of_vertical_links_to_node
        """update the flow within the grid"""
        # you need to set the boundaries on the grid
        # for testing we will trun (True, False, True, True
        # self._tidal_flow = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level, tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600, roughness=0.02).calc_tidal_inundation_rate()
        foo = TidalFlowCalculator(self._grid, mean_sea_level=0, tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600, roughness=0.02)
        foo.run_one_step()
        self._tidal_flow = foo.calc_tidal_inundation_rate()
        self._ebb_tide_vel = foo._ebb_tide_vel
        self._flood_tide_vel = foo._flood_tide_vel
        self._Uy = map_mean_of_horizontal_links_to_node(self._grid, foo._flood_tide_vel) # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor
        self._Ux = map_mean_of_vertical_links_to_node(self._grid, foo._flood_tide_vel)  # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor
        self._U = np.sqrt((pow(self._Ux,2) + pow(self._Uy, 2))) # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor

        # print(self._tidal_range)
        # print(self._tidal_period)
        # print(self._mean_sea_level)
        # # print(self._tidal_flow*0.5*10000) # not sure what the units are here.
        # print(self._flood_tide_vel)
        # # print(self._tidal_flow*.5/self._water_depth*10000)
        # # print(self._water_depth)
        # stop


    def update_morphology(self):
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

        p = np.where(self._model_domain > 0, True, False) # exclude cells that cant get sediment.


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
            watPow = self._water_depth**(-1 / 3)
            # watPow[watPow == np.inf] = 0
            tauC = 1030 * 9.81 * self._roughness**(2) * watPow * np.power(Utide, 2)
            E = (E + 1) / (ncyc + 1) * self._me * (np.sqrt(1 + (tauC/taucro)**2) - 1)
        Etide = E
        # E2 = Etide


        # ## CURRENT-DRIVEN TRANSPORT (Tide and River)######################################################
        # (2)Total sediment resupension MUD
        # figure out how to call the total sed code... Do we build an integrated function and put it here?
        # E2, E2tide = totalsedimenterosionMUDsine(U,MANN,VEG,fTide,NaN,Uwave_sea,NaN,Tp_sea,NaN,NaN,1,NaN,taucr,taucrVEG,me,h,Zlev,TrangeVEG,computeSeaWaves,0,0)
        # E2[A==2]=0  # needed for b.c.

        # Advection-Diffusion Sediment transport
        WS = (self._elev * 0) + self._ws2
        WS[self._veg_is_present == 1] = self._wsB
        # print(f'WS is {WS}')
        # WS(S==1)=ws2# SHOULD NOT BE NECEEARY BECUASE VEG alreeady set equal to zero where S=1 (see above).  ->Do not add the vegetation settling velocity in the ponds! %WS(S==1)=0.000000000001%E2(S==1)=0

        ###################### Sedtran ############################################
        # This is all modified by Thomas Huff from the original MatLab code
        # varibles to define
        # DoMUD = 1;%base diffusivity of suspedned mud [m2/s]. Process not related to tides (e.g. wind and waves, other ocean circulation)
        # Diffs = 1; %[-]coefficient for tidal dispersion [-]. 1 DO NOT CHANGE
        # Ttide = self._tide_period
        # Ux =
        # dx =
        # h =

        # now handle the sediment diffusion across the grid.
        # diffusion = # (DoMUD*24*3600 +DiffS*self._tidal_period/2.*(abs(Ux.*Ux))*(24*3600).^2)/(dx^2).*h

        # floodedGrid = self._elev[self._fully_wet_depth > 0.1] # figure out how to only apply
        #
        #
        # this at the wetted areas.
        # ld = landlab.components.LinearDiffuser(self._grid, linear_diffusivity = self._DoMUD)
        # gdoff = np.copy(hd.fixed_grad_offsets)
        # gdoff[self._fully_wet_depth < 0.1] = self._elev # reset elevation in areas that are not properly wetted
        dx = 5 * 2
        tst = abs(self._Ux * self._Ux) * (24 * 3600)
        # (self._DoMUD * 24 * 3600 + self._DiffS * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dxx = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dyy = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Uy * self._Uy) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth

        # calculations are the same as matlab up to this point.
        DxxMin = map_min_of_node_links_to_node(self._grid, map_min_of_link_nodes_to_link(self._grid, Dxx))
        DyyMin = map_min_of_node_links_to_node(self._grid, map_min_of_link_nodes_to_link(self._grid, Dyy))

        N, M = self._grid.shape
        S = np.zeros((N * M))
        ilog = []
        jlog = []
        s = []
        for k in [N, -1, 1, -N]:
            print(k)
            tmp = numpy.array(range(0, N * M))
            q = tmp[p] + k # the translated cell
            a0 = np.where(q > 0, True, False)
            a1 = np.where(q < (N * M), True, False)
            a2 = np.where(a1 == True, a0, False)

            a = np.where(p == a2, True, False)
            del(a0)
            del(a1)
            del(a2)

            # This just builds a numberical array to coordspond the boolian p values
            ptmp = []
            rnd = -1
            for parray in p:
                rnd += 1
                if parray == True:
                    ptmp.append(rnd)
            parray = np.array(ptmp)

            # calculate DD value
            if k == N | k == -N:
                D = Dyy
            else:
                D = Dxx

            numeric = 0
            if numeric  == 1:
                DD = (D[parray[a]] + D[q[a]])/2
            else:
                DD = np.minimum(D[parray[a]], D[q[a]])

            value = DD / self._water_depth[parray[a]] / self._hydroperiod[parray[a]]


            Fin = self._elev
            Fin[:] = 0
            Fout = Fin
            # There are some challenges with getting the indexes to match and this is how I am getting around it.
            tmpInd = parray[a]
            Fin[self._index[tmpInd[self._model_domain[parray[a]] == 1]]] = 1
            tmpInd = q[a]
            Fout[self._index[tmpInd[self._model_domain[q[a]] == 1]]] = 1




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
            # if residualcurrents==1;% tidal residual currents and transport.
            #    % (I imposed no residual currents are the opne boundary to avoid
            #    % calcuitating the fluxes to get the mass balance at 100%)
            # if (k==N);UR=UresY(p(a));up=find(UR>0);F=UR(up);end
            # if (k==-N);UR=UresY(p(a));up=find(UR<0);F=-UR(up);end
            # if (k==1);UR=UresX(p(a));up=find(UR>0);F=UR(up);end
            # if (k==-1);UR=UresX(p(a));up=find(UR<0);F=-UR(up);end
            # value(up)=value(up)+F*3600*24/dx;
            # end

            S[parray[a]] = S[parray[a]] + (value * Fin[parray[a]])
            ilog = list(ilog) + list(self._index[q[a]])
            jlog = list(jlog) + list(self._index[parray[a]])
            s = list(s) + list(-value * Fout[q[a]])
        settling = self._elev * 0
        settling[p] = 24 * 3600 * WS[p] / self._water_depth[p] * self._hydroperiod[p]
        settling[self._model_domain == 2] = 0


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

        ilog = list(ilog) + list(self._index[p])
        jlog = list(jlog) + list(self._index[p])
        s = list(s) + list(S[p] + settling[p])
        import scipy
        from scipy import sparse
        from scipy.sparse.linalg import LinearOperator
        ds2 = sparse.csc_matrix((s, (ilog, jlog)))#, shape = (N, M))
        rhs = np.array(E[p])
        # foo = np.vstack([rhs, np.zeros(len(rhs))])
        P = scipy.sparse.linalg.spsolve(ds2, rhs)


    # START WORKING FROM HERE ON WEDNESDAY.


        valuex = DxxMin / self._water_depth / self._hydroperiod
        valuey = DyyMin / self._water_depth / self._hydroperiod

        pfin = np.where(self._model_domain == 1, True, False)
        Fin = np.copy(self._elev)
        Fin[:] = 0
        Fin[pfin] = 1
        Fout = Fin # this is just for testing the first loop

        S = numpy.zeros(len(valuex))
        S = S + valuex * Fin #exit from that cell
        S = S + valuey * Fin  # exit from that cell

        sOutx = -valuex * Fout
        sOuty = -valuey * Fout

        D = np.sqrt(Dxx**2 + Dyy**2)

        # what needs to be done.....
        # We have calcualted E and D in the MM2D equation now we need to incorporate sediment transfer between nodes.
        # we have the flow velocities in both the x and y directions and we can calculate the divergence bwtween these values.
        from landlab.grid.divergence import calc_flux_div_at_node
        grad = self._grid.calc_grad_at_link(D)
        #S = calc_flux_div_at_node(self._grid, -grad)
        z = self._grid.calc_grad_at_link(self._elev)
        zNode = calc_flux_div_at_node(self._grid, z)
        const = 1
        S = const*D*zNode
        value = D / self._water_depth / self._hydroperiod
        from landlab.utils import get_core_node_matrix
        mat, rhs = get_core_node_matrix(
            self._grid,
            S,
        )
        rhs[:,0] = E[self.grid.core_nodes] # it is possible that this should be D

        from scipy.sparse.linalg import spsolve
        testfoo = spsolve(mat, rhs)
        # print(testfoo)
        # stop

        # print(self._U)
        # print(value)
        # stop





        ######################################################################################################
        # Function to this point has been double checked with Matlab.  Below this point.... There be monsters. And none of it is correct

        # get minimum surrounding velocities
        # DDx = map_min_of_node_links_to_node(self._grid, map_min_of_link_nodes_to_link(self._grid, Dxx))
        # DDy = map_min_of_node_links_to_node(self._grid, map_min_of_link_nodes_to_link(self._grid, Dyy))
        #
        # valueX = DDx / self._water_depth / self._hydroperiod
        # valueY = DDy / self._water_depth / self._hydroperiod
        # Fin = np.copy(self._elev)
        # Fin[:] = 0
        # Fin[self._model_domain == 1] = 1
        #
        # S = (valueX*Fin) + (valueY*Fin)
        #
        # # I am very unsure about the next bit of code
        # sx = (-valueX * Fin)
        # sy = (-valueY * Fin)

        settling = self._elev * 0
        settling[p] = 24*3600*WS[p]/self._water_depth[p]*self._hydroperiod[p]
        # print(settling)
        # stop
        settling = S + settling # s should likely be replaced with D

        # # setting up rhs
        # a = np.where(self._model_domain == 2, True, False)
        # rhs = E
        # rhs[a] = self._sea_SSC * self._water_depth[a] * self._hydroperiod[a]
        # import scipy
        # rnd = -1
        # for i in [S, sx, sy, sSettling]: # This implementation is completely wrong.
        #     rnd += 1
        #     hld = np.vstack([i, np.ones(len(i))])#, sx, sy, sSettling])
        #     # rhs = np.vstack([rhs, rhs, rhs, rhs])
        #     # print(hld.shape)
        #     # print(rhs.shape)
        #     m,c = np.linalg.lstsq(hld.T, rhs.T)[0]
        #     vals = m * i + c
        #     if rnd == 0:
        #         P = vals
        #     else:
        #         P = vals * P
        # # still not sure about the above code.  This maybe completely wrong! I am not sure about the combination of multiple vectors
        #
        SSM = np.copy(self._elev)
        SSM[:] = 0
        SSM[self._grid.core_nodes] = testfoo

        EmD = np.copy(self._elev)
        EmD[:] = 0 # zero out a new array
        EmD[p] = (E[p] - SSM*settling)/self._rbulk
        print(EmD)

        # Ux = map_mean_of_vertical_links_to_node(self._grid, Dhd)
        # Uy = map_mean_of_horizontal_links_to_node(self._grid, Dhd)
        #
        # DDx = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(Ux * Ux) * (
        #             24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        # DDy = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(Uy * Uy) * (
        #             24 * 3600) ** 2) / (dx ** 2) * self._water_depth

        #print(D.shape)

        # Section to increase accretion at the edge of a channel. CURRENTLY NOT IMPLEMENTED
        # This section basically looks at adjacent pixels to the channel and adds some additional accretion there in mud emvironments and less in sand

        # if i in range(0,1):
        #     if i  == 0:
        #         D = Dxx
        #     else:
        #         D = Dyy
        #
        #     numeric = 0
        #     if numeric==1:
        #         DD=(D(p(a))+D(q(a)))/2 # THE ORIGINAL FOR MARSH makes more accumulation on the sides of the channels ie narrower channels
        #     else:
        #         DD=min(D(p(a)),D(q(a))) # less levees on the sides. Especially with SAND

        # currently missing sections for river flow and residualcurrents

        # # boundary conditions imposed SSC
        # a = np.copy(self._model_domain)
        # a = np.where(a == 2, True, False) # 2 signifies boundary conditions.
        # # rhs = self._model_domain * 0 # create an empty grid of zero values.
        # rhs = E # [self._model_domain>0]
        # rhs[a] = self._sea_SSC * self._water_depth[a] * self._hydroperiod[a]
        # print(a)
        # print(rhs)
        # stop
        #
        # #summary of the material that exits the cell
        # p = np.copy(self._elev)
        # # print(p)
        # p = np.where(p < self._fully_wet_depth, True, False)
        # # print(24*3600*WS[p])
        # # print(max(self._water_depth[p]))
        # # print(max(self._hydroperiod[p]))
        # settling= (24*3600*WS)/self._water_depth*self._hydroperiod # .*(h(p)>3);
        # settling[settling == np.inf] = 0 #fix inf values
        # settling[p == False] = 0
        # # settling[a] = 1 # no setting in boundary areas.
        # # print(f'Settling is {max(settling)}')
        #
        # # print(f'rhs is {rhs}')
        # SSM = settling/rhs
        # # fix inf values
        # SSM[SSM == np.inf] = 0
        # SSM[SSM == np.nan] = 0
        #
        # EmD = np.copy(self._elev)
        # EmD[:] = 0
        # EmD[p] = (E[p]-(SSM[p] * settling[p]))/self._rbulk

        # SSC2 = SSM/self._water_depth

        # update bed elevation
        # print("\n")
        # print("*"*60)
        # print(f'The E value is {statistics.mean(-E)}')
        # print(f'The WS value is {statistics.mean(WS)}')
        # impose boundary conditions
        EmD[self._model_domain == 2] = 0
        # print(np.reshape(EmD, (500,300))[:,1])
        # stop
        self._elev = self._elev - (1 * EmD) #(1 * EmD/100)
        # print("*" * 60)
        # print(f'mean of EmD is {statistics.mean(EmD)}')

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

    def run_one_step(self, dt):
        """Advance in time."""

        # Update sea level
        self._mean_sea_level = self._mean_sea_level +  (self._rel_sl_rise_rate * dt)

        # water depth
        self.get_water_depth()

        # vegetation
        self.update_vegetation()

        # roughness
        self.update_roughness()

        # tidal_flow
        self.update_flow()

        # bed morphology
        self.update_morphology()