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

        # grid boundaries
        self._grid.set_closed_boundaries_at_grid_edges(right_is_closed=True,
                                                       top_is_closed= False,
                                                       left_is_closed=True,
                                                       bottom_is_closed=True)


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
        from landlab.grid.mappers import map_max_of_node_links_to_node
        """update the flow within the grid"""
        # you need to set the boundaries on the grid
        # for testing we will trun (True, False, True, True
        # self._tidal_flow = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level, tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600, roughness=0.02).calc_tidal_inundation_rate()
        foo = TidalFlowCalculator(self._grid, mean_sea_level=self._mean_sea_level, tidal_range=self._tidal_range, tidal_period = self._tidal_period*24*3600, roughness=0.02)
        foo.run_one_step()
        # tst = foo.calc_tidal_inundation_rate()
        # flowHld = map_max_of_node_links_to_node(self._grid, tst)
        self._tidal_flow = foo.calc_tidal_inundation_rate()
        self._ebb_tide_vel = foo._ebb_tide_vel
        self._flood_tide_vel = foo._flood_tide_vel
        self._Uy = map_mean_of_horizontal_links_to_node(self._grid, foo._flood_tide_vel) # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor
        self._Ux = map_mean_of_vertical_links_to_node(self._grid, foo._flood_tide_vel)  # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor
        # test
        # self._Ux = self._Ux * 10
        # self._Uy = self._Uy * 10

        # fill in the outer edges of the grid.  Landlab leaves these edges as zero due to the core matrix calculations however this causes issues with the matrix inversion step
        # in the morphology calculator
        gridNum = np.reshape(self._index, (self._grid.shape))
        self._Ux[gridNum[:,0]] = self._Ux[gridNum[:,1]] # fill in the left column with data from the next column over
        self._Ux[gridNum[:, self._grid.shape[1]-1]] = self._Ux[gridNum[:, self._grid.shape[1]-2]]
        self._Ux[gridNum[0,:]] = self._Ux[gridNum[1,:]]
        # self._Ux[gridNum[-2, :]] = self._Ux[gridNum[-1, :]] # replace the 2nd to bottom row with the bottom row

        self._Uy[gridNum[:,0]] = self._Uy[gridNum[:,1]] # fill in the left column with data from the next column over
        self._Uy[gridNum[:, self._grid.shape[1]-1]] = self._Uy[gridNum[:, self._grid.shape[1]-2]]
        self._Uy[gridNum[0,:]] = self._Uy[gridNum[1,:]]
        # self._Uy[gridNum[-2, :]] = self._Uy[gridNum[-1, :]]


        self._U = np.sqrt((pow(self._Ux,2) + pow(self._Uy, 2))) # not sure about the units but this matches the matlab output. Thomas added the *2/100 conversion factor

        # print(self._tidal_range)
        # print(self._tidal_period)
        # print(self._mean_sea_level)
        # # print(self._tidal_flow*0.5*10000) # not sure what the units are here.
        # print(self._flood_tide_vel)
        # # print(self._tidal_flow*.5/self._water_depth*10000)
        # # print(self._water_depth)
        # stop


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

        p = np.where(self._model_domain > 0, np.where(self._water_depth <=0, False, True), False) # exclude cells that cant get sediment as they are above water.


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
            tauC = 1030 * 9.81 * self._roughness**(2) * watPow * np.power(Utide, 2)
            E = E + (1 / (ncyc + 1)) * self._me * (np.sqrt(1 + (tauC/taucro)**2) - 1)
        Etide = E
        E[self._model_domain == 2] = 0
        E[E == np.inf] = 0
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
        # (self._DoMUD * 24 * 3600 + self._DiffS * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dxx = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Ux * self._Ux) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth
        Dyy = (self._DoMUD * 24 * 3600 + self._Diffs * self._tidal_period / 2 * np.abs(self._Uy * self._Uy) * (24 * 3600) ** 2) / (dx ** 2) * self._water_depth

        # calculations are the same as matlab up to this point.
        # DxxMin = map_min_of_node_links_to_node(self._grid, map_min_of_link_nodes_to_link(self._grid, Dxx))
        # DyyMin = map_min_of_node_links_to_node(self._grid, map_min_of_link_nodes_to_link(self._grid, Dyy))

        N, M = self._grid.shape
        S = np.zeros((N * M))
        ilog = []
        jlog = []
        s = []
        for k in [N, -1, 1, -N]:
            # print(k)
            tmp = self._index[p]# numpy.array(range(0, N * M))
             # the translated cell

            # add a control filter to limit the extents of the filter array
            # qFilter = np.where(q <= N * M, True, False)

            # indexLog = []
            # lg = -1
            # for row in range(N):
            #     for col in range(M):
            #         lg = lg + 1
            #         if k == N:
            #             if (col+2) <= M:
            #                 indexLog
            #             a = np.where(col+2 <= M)


            # figure this section out
            row, col = np.unravel_index(tmp, shape=(N, M)) # sort this out.
            # indTemp = np.reshape(self._index, (N, M))
            if k == N:
                a = np.where(col + 1 < M, True, False)
                q = tmp + 1
                # qa = indTemp[:,0:M-1].flatten()
                # pa = indTemp[:,1:M].flatten()
            if k == -N:
                a = np.where(col - 1 >= 0, True, False)
                q = tmp - 1 # originally tmp was tmp[p]

                # fsub = indTemp[:, 1:M]
                # a = np.reshape(fsub, (N, M-1))
                # qa = indTemp[:,1:M].flatten()
                # pa = indTemp[:,0:M-1].flatten()
            if k == -1:
                a = np.where(row - 1 >= 0, True, False)
                q = tmp - M
                # fsub = indTemp[1:N, :]
                # a = np.reshape(fsub, (N, M-1))
                # qa = indTemp[1:N,:].flatten()
                # pa = indTemp[0:N-1, :].flatten()
            if k == 1:
                a = np.where(row + 1 < N, True, False)
                q = tmp + M
                # fsub = indTemp[0:N-1, :]
                # a = np.reshape(fsub, (N, M-1))
                # qa = indTemp[0:N-1,:].flatten()
                # pa = indTemp[1:N, :].flatten()


            # sanity filter application
            # aFilter = np.where(q < (N * M), True, False)
            # a = np.where(aFilter == True, a, False)
            # a = []
            # if k == N:
            #     rnd = rnd + 1
            #     for rw in range(N):
            #         for col in range(M):
            #             if col + 1 < M:
            #                 a.append(rnd)
            #
            #
            #
            # a0 = np.where(q > 0, True, False)
            # a1 = np.where(q < (N * M), True, False)
            # a2 = np.where(a1 == True, a0, False)

            # pInd = np.copy(self._index[p])
            # a = q[a2]
            # a = np.where(p == a2, True, False)
            # del(a0)
            # del(a1)
            # del(a2)

            # This just builds a numberical array to coordspond the boolian p values
            # ptmp = []
            # rnd = -1
            # for pr in p:
            #     rnd += 1
            #     if pr == True:
            #         ptmp.append(int(rnd))
            #     else:
            #         ptmp.append(np.nan)
            parray = self._index[p] #np.array(ptmp)

            # calculate DD value
            if (k == N) | (k == -N):
                D = Dyy
                # set the second the last row to the same as the bottom row
                # This is compensating for an edge effect issue with the Flow calculations
                # D[-M*2:-M] = Dyy[-M:]
            else:
                D = Dxx
                # set the second the last row to the same as the bottom row
                # This is compensating for an edge effect issue with the Flow calculations
                # D[-M*2:-M] = Dxx[-M:]


            # There is an issue with the DD calculation. In Matlab D(q(a)) == D(p(a)) but that isn't true for python.
            numeric = 1
            if numeric == 1:
                # print(f'len p array {len(parray[a])}')
                # print(f'len q {len(q[a])}')
                try:
                    DD = (D[parray[a]] + D[q[a]])/2
                except:
                    print("BLAAA")

            else:
                DD = np.minimum(D[parray[a]], D[q[a]])
            try:
                value = DD / self._water_depth[parray[a]] / self._hydroperiod[parray[a]]
            except:
                print("Here")


            Fin = np.copy(self._elev)
            Fin[:] = 0
            Fout = np.copy(Fin)
            # There are some challenges with getting the indexes to match and this is how I am getting around it.
            Fin[parray[a][np.in1d(parray[a], self._index[self._model_domain == 1])]] = 1
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
            # matrix indexing to match matlabs matrix
            # matLabIndex = np.zeros((N,M))
            # ind = -1
            # for c in range(M):
            #     for r in range(N):
            #         ind +=1
            #         matLabIndex[r, c] = ind
            # matLabIndex = matLabIndex.flatten()
            S[parray[a]] = S[parray[a]] + (value * Fin[parray[a]])
            try:
                ilog = list(ilog) + list(q[a])
                jlog = list(jlog) + list(parray[a])
            except:
                print("ilog")
            s = list(s) + list(-value * Fout[q[a]])
        settling = np.copy(self._elev) * 0
        settling[p] = 24 * 3600 * WS[p] / self._water_depth[p] * self._hydroperiod[p]
        settling[self._index[self._model_domain == 2]] = 0
        S[self._index[self._model_domain == 2]] = 1
        aa = self._index[self._model_domain == 10]
        S[p[aa]] = 1 # impose the b.c.
        del(aa)




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
        try:
            ilog = list(ilog) + list(parray)
            jlog = list(jlog) + list(parray)
            s = list(s) + list(S[p] + settling[p])
        except:
            print("Error with ilog or jlog")
        import scipy
        from scipy import sparse
        from scipy.sparse.linalg import LinearOperator
        ds2 = sparse.csc_array((s, (ilog, jlog)), shape = (N*M, N*M)) #, shape = (N, M))
        rhs = np.array(E)
        rhs[:] = 0
        rhs[p] = E[p]
        aa = self._index[self._model_domain == 2]
        try:
            rhs[aa] = (60/1000) * self._water_depth[aa] * self._hydroperiod[aa]
        except:
            print('rhs creation issue')
        del (aa)
        MUD = 1 # This is just for testing.  Make sure to remove this later on
        if MUD == 1:
            aa = self._index[self._model_domain == 10]
            rhs[aa] = np.nan * self._water_depth[aa] * self._hydroperiod[aa]
            del(aa)
        # inc = -1
        # for mi in matLabIndex:
        #     inc +=1
        #     rhs[inc] = rhs[int(mi)]
        # foo = np.vstack([rhs, np.zeros(len(rhs))])
        # print("Running matrix calculations.")
        # import scikits.umfpack
        #rhsML = np.copy(rhs[matLabIndex.astype(int)])
        try:
            P = scipy.sparse.linalg.spsolve(ds2, rhs) # was working with .lsqr
        except:
            print("Matrix solution was singular. Reverting to lsqr to solve matrix inversion")
            P = scipy.sparse.linalg.lsqr(ds2, rhs, iter_lim = 5000)[0]

        # convert to python matrix ordering
        # PyP = np.copy(P)
        # PyP[matLabIndex.astype(int)] = np.copy(P)
        # from scipy.linalg import qr
        # def solve_minnonzero(A, b):
        #     x1, res, rnk, s = scipy.sparse.linalg.lsqr(A, b)
        #     if rnk == A.shape[1]:
        #         return x1  # nothing more to do if A is full-rank
        #     Q, R, P = qr(A.T, mode='full', pivoting=True)
        #     Z = Q[:, rnk:].conj()
        #     C = np.linalg.solve(Z[rnk:], -x1[rnk:])
        #     return x1 + Z.dot(C)

        #P = solve_minnonzero(ds2, rhs)

        # from scipy.optimize import nnls
        # P = nnls(ds2, rhs)
        SSM = np.array(P)

        # print('Building EmD array')
        EmD = np.copy(self._elev)
        EmD[:] = 0 # zero out a new array
        EmD[p] = (E[p] - SSM[p]*settling[p])/self._rbulk
        # print(EmD)
        # impose boundary conditions
        EmD[self._model_domain == 2] = 0
        # EmD[np.isnan(EmD)] = 0 # set na to zero

        ref = np.copy(self._elev[self._model_domain == 2]) # the boundary areas to no get a change in elevation.
        # print("Calculating elevation.")
        self._elev = self._elev - (dt * EmD) #(1 * EmD/100)
        self._elev[self._model_domain == 2] = -10

        # return the maxdeltaz value for the loop computation
        return(np.percentile(abs(self._elev - origz), 99))

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
            while maxdeltaz > self._limitdeltaz | maxup > self._limitmaxup:
                if firstattempt != 1:
                    dt = dt / 2 * np.minimum((self._limitdeltaz / maxdeltaz), (self._limitmaxup / maxup))
                firstattempt = 0
                if round <= 2:
                    dt = np.minimum(0.5*365, dt)
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
                maxdeltaz = self.update_morphology(dt)
            dti = dti+dt
            dt = np.minimum((dt*2), (np.maximum(0, dto-dti)))
