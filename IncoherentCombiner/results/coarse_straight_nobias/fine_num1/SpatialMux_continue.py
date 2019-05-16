import emopt
from emopt.misc import NOT_PARALLEL, run_on_master, COMM
from emopt.adjoint_method import AdjointMethod, AdjointMethodMO

import numpy as np
from math import pi
import copy

from petsc4py import PETSc
from mpi4py import MPI

import h5py

####################################################################################
# Combiner class
####################################################################################
class SpatialMuxAdjointMethod(AdjointMethod):

    def __init__(self, sim, spacer, waveguide, lower, indices, xs, ys, ZminD,
                 ZmaxD, fom_domain, mode_match):
        super(SpatialMuxAdjointMethod, self).__init__(sim, step=1e-7)
        self.spacer = spacer
        self.waveguide = waveguide
        self.lower = lower
        self.indices = indices
        self.xs = xs
        self.ys = ys
        self.x_current = np.copy(xs)
        self.y_current = np.copy(ys)
        self.mode_match = mode_match
        self.fom_domain = fom_domain
        self.ZminD = ZminD
        self.ZmaxD = ZmaxD

        self.roc_min = 0.15
        self.k_roc = 10
        self.A_roc = 0.0025

        self.step_roc = 1e-8

    def get_current_points(self, params):
        # required to make the grad_y calculation simpler
        x = np.copy(self.xs)
        y = np.copy(self.ys)
        ids = self.indices

        N = len(ids)
        for i in range(N):
            #if ids[i] == x.size-1:
            #    x[ids[i]] = x[0] + params[0]
            #    y[ids[i]] = y[0] + params[N]
            #else:
            #    x[ids[i]] = x[ids[i]] + params[i]
            #    y[ids[i]] = y[ids[i]] + params[i+N]

            x[ids[i]] = x[ids[i]] + params[i]
            y[ids[i]] = y[ids[i]] + params[i+N]
        

        #if NOT_PARALLEL:
        #    import matplotlib.pyplot as plt
        #    fff = plt.figure()
        #    axxx = fff.add_subplot(111)
        #    axxx.plot(x, y, '-o')
        #    plt.show()

        return x, y

    def update_system(self, params):
        x, y = self.get_current_points(params)

        self.x_current = x
        self.y_current = y

        self.spacer.set_points(x,y)
        self.waveguide.set_points(x,y)
        self.lower.set_points(x,y)

    def calc_roc_fom(self, params):
        x, y = self.get_current_points(params)

        roc_fom = 0.0
        ids = self.indices
        N = len(ids)

        ##### TEMP ####
        #x_store = []
        #y_store = []
        #ids_store = []

        ##### TEMP ####

        for i in range(N):
            #if(ids[i] == 0):
            #    x1 = x[-2]; x2 = x[ids[i]]; x3 = x[ids[i]+1]
            #    y1 = y[-2]; y2 = y[ids[i]]; y3 = y[ids[i]+1]
            #elif(ids[i] == x.size-1):
            #    x1 = x[ids[i]-1]; x2 = x[ids[i]]; x3 = x[1]
            #    y1 = y[ids[i]-1]; y2 = y[ids[i]]; y3 = y[1]
            #else:
            #    x1 = x[ids[i]-1]; x2 = x[ids[i]]; x3 = x[ids[i]+1]
            #    y1 = y[ids[i]-1]; y2 = y[ids[i]]; y3 = y[ids[i]+1]

            x1 = x[ids[i]-1]; x2 = x[ids[i]]; x3 = x[ids[i]+1]
            y1 = y[ids[i]-1]; y2 = y[ids[i]]; y3 = y[ids[i]+1]

            roc = emopt.fomutils.radius_of_curvature(x1, x2, x3, y1, y2, y3)

            penalty = emopt.fomutils.step(self.roc_min - roc, self.k_roc,
                                          A=self.A_roc)
            roc_fom += penalty
            #if(penalty>=0.0025):
            #    x_store.append([x1, x2, x3])
            #    y_store.append([y1,y2,y3])
            #    ids_store.append([ids[i]-1, ids[i], ids[i]+1])

        return roc_fom

    @run_on_master
    def calc_fom(self, sim, params):
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]
        Psrc = sim.source_power

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom/Psrc + self.calc_roc_fom(params)

    def calc_dFdx(self, sim, params):
        Psrc = sim.source_power

        fom_no_norm = self.calc_fom(sim, params) * Psrc

        if(NOT_PARALLEL):
            dFdEx = -1*self.mode_match.get_dFdEx() / Psrc
            dFdEy = -1*self.mode_match.get_dFdEy() / Psrc
            dFdEz = -1*self.mode_match.get_dFdEz() / Psrc
            dFdHx = -1*self.mode_match.get_dFdHx() / Psrc
            dFdHy = -1*self.mode_match.get_dFdHy() / Psrc
            dFdHz = -1*self.mode_match.get_dFdHz() / Psrc
        else:
            dFdEx = None; dFdEy = None; dFdEz = None
            dFdHx = None; dFdHy = None; dFdHz = None

        adjoint_sources = \
            emopt.fomutils.power_norm_dFdx_3D(sim, fom_no_norm,
                                              self.fom_domain,
                                              dFdEx, dFdEy, dFdEz,
                                              dFdHx, dFdHy, dFdHz)
        return adjoint_sources

    def calc_grad_y(self, sim, params):
        Np = len(params)
        grad_y = np.zeros(Np)
        for i in range(Np):
            param_i = params[i]
            params[i] = param_i - self.step_roc/2.0
            roc_fom0 = self.calc_roc_fom(params)
            params[i] = param_i + self.step_roc/2.0
            roc_fom1 = self.calc_roc_fom(params)

            grad_y[i] = (roc_fom1 - roc_fom0) / self.step_roc

            params[i] = param_i

        return grad_y

    def get_update_boxes(self, sim, params):
        x, y = self.get_current_points(params)

        dx = sim.dx
        dy = sim.dy
        dz = sim.dz

        ids = self.indices
        N = len(ids)

        update_boxes = []
        for i in range(N):
            if(ids[i]==0):
                xmin = np.min([x[-2], x[ids[i]], x[ids[i]+1]])-2*dx
                xmax = np.max([x[-2], x[ids[i]], x[ids[i]+1]])+2*dx
                ymin = np.min([y[-2], y[ids[i]], y[ids[i]+1]])-2*dy
                ymax = np.max([y[-2], y[ids[i]], y[ids[i]+1]])+2*dy
                ubs = [xmin, xmax, ymin, ymax, self.ZminD-2*dz, self.ZmaxD+2*dz]
            elif(ids[i]==x.size-1):
                xmin = np.min([x[ids[i]-1], x[ids[i]], x[1]])-2*dx
                xmax = np.max([x[ids[i]-1], x[ids[i]], x[1]])+2*dx
                ymin = np.min([y[ids[i]-1], y[ids[i]], y[1]])-2*dy
                ymax = np.max([y[ids[i]-1], y[ids[i]], y[1]])+2*dy
                ubs = [xmin, xmax, ymin, ymax, self.ZminD-2*dz, self.ZmaxD+2*dz]
            else:
                xmin = np.min([x[ids[i]-1], x[ids[i]], x[ids[i]+1]])-2*dx
                xmax = np.max([x[ids[i]-1], x[ids[i]], x[ids[i]+1]])+2*dx
                ymin = np.min([y[ids[i]-1], y[ids[i]], y[ids[i]+1]])-2*dy
                ymax = np.max([y[ids[i]-1], y[ids[i]], y[ids[i]+1]])+2*dy
                ubs = [xmin, xmax, ymin, ymax, self.ZminD-2*dz, self.ZmaxD+2*dz]

            update_boxes.append(ubs)

        update_boxes_initial = copy.deepcopy(update_boxes)
        for u in update_boxes_initial:
            update_boxes.append(u) # accounts for y params

        return update_boxes

###################################################################################
# Multi-objective class
###################################################################################
class MultiObjective(AdjointMethodMO):
    def calc_total_fom(self, foms):
        return np.mean(foms)

    def calc_total_gradient(self, foms, grads):
        gradient = np.mean(grads, axis=0)
        return gradient


###################################################################################
# Definitions
###################################################################################
def opt_finish_iteration(params, sim1, am1, sim2, am2, fom_history,
                         fom1_history, fom2_history, eff_history, field_domain):
    fom1 = -1*am1.calc_fom(sim1, params)
    fom2 = -1*am2.calc_fom(sim2, params)

    fom = 0.5*(fom1 + fom2)
    penalty = am1.calc_roc_fom(params)

    print('Current FOM: %0.4E' % (-1*fom))
    print('SIM1: %0.4E' % (-1*fom1))
    print('SIM2: %0.4E' % (-1*fom2))
    print('Efficiency: %0.4E' % (-1*(fom+penalty)))
    # update FOM history
    fom_history.append(fom)
    fom1_history.append(fom1)
    fom2_history.append(fom2)
    eff_history.append(fom+penalty)

    foms = {'FOM':fom_history, 'FOM1':fom1_history, 'FOM2':fom2_history,
            'Efficiency':eff_history}

    # get fields and calc |E| for plotting
    Ex1 = sim1.saved_fields[1][0]
    Ey1 = sim1.saved_fields[1][1]
    Ez1 = sim1.saved_fields[1][2]
    E1 = np.squeeze(np.sqrt(np.abs(Ex1)**2 + np.abs(Ey1)**2 + np.abs(Ez1)**2))

    Ex2 = sim2.saved_fields[1][0]
    Ey2 = sim2.saved_fields[1][1]
    Ez2 = sim2.saved_fields[1][2]
    E2 = np.squeeze(np.sqrt(np.abs(Ex2)**2 + np.abs(Ey2)**2 + np.abs(Ez2)**2))


    # get permittivity distribution to vis structure
    eps_grid = sim1.eps.get_values_in(field_domain, squeeze=True)

    # generate plot
    emopt.io.plot_iteration(np.real(E1), np.real(eps_grid),
                            sim1.X_real, sim1.Y_real, foms,
                            fname='current_result1.pdf')
    emopt.io.plot_iteration(np.real(E2), np.real(eps_grid),
                            sim2.X_real, sim2.Y_real, foms,
                            fname='current_result2.pdf')


    dout = {}
    dout['foms'] = fom_history
    dout['params'] = params
    dout['X'] = sim1.X_real
    dout['Y'] = sim1.Y_real

    dout['eps'] = eps_grid

    dout['Ex1'] = Ex1
    dout['Ey1'] = Ey1
    dout['Ez1'] = Ez1

    dout['Ex2'] = Ex2
    dout['Ey2'] = Ey2
    dout['Ez2'] = Ez2

    emopt.io.save_results('results', dout)

class SpatialMux(object):
    def __init__(self, wavelength):
        Xsim = 12.0
        Ysim = 5.0
        Zsim = 2.0
        dx = 0.04
        dy = 0.04
        dz = 0.04


        ####################################################################################
        # Setup simulations
        ####################################################################################
        w_pml = dx * 15

        X = Xsim + 2*w_pml
        Y = Ysim + 2*w_pml
        Z = Zsim + 2*w_pml


        sim1 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-5, min_rindex=1.0)
        sim1.src_ramp_time = sim1.Nlambda * 20
        sim1.Nmax = sim1.Nlambda * 1000

        sim1.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
        sim1.bc = '000'


        sim2 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-5, min_rindex=1.0)
        sim2.src_ramp_time = sim2.Nlambda * 20
        sim2.Nmax = sim2.Nlambda * 1000

        sim2.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
        sim2.bc = '000'


        X = sim1.X
        Y = sim1.Y
        Z = sim1.Z

        X_real = sim1.X_real
        Y_real = sim1.Y_real
        Z_real = sim1.Z_real
        ####################################################################################
        # Define the geometry/materials
        ####################################################################################
        w_wg = 0.55
        w_wg_out = 0.9
        wg_spacing = 1.0
        wg_shift = -0.2
        Lio = 2.0

        h_spacer = 0.04
        h_waveguide = 0.18
        h_lowerRidge = 0.5

        n_InP = 3.4
        n_SiO2 = 1.444

        position1 = Lio+w_pml+3.0
        position2 = X-Lio-w_pml-1.0

        #x_pts = []
        #y_pts = []

        #x_pts.append(np.array([0,0,w_pml+Lio]))
        #y_pts.append(Y/2+np.array([wg_spacing/2,
        #                          wg_spacing/2+w_wg,
        #                          wg_spacing/2+w_wg]))

        ##x_pts.append(np.array([w_pml+Lio, 0, 0, w_pml+Lio]))
        ##y_pts.append(Y/2+np.array([wg_spacing/2,
        ##                           wg_spacing/2,
        ##                           wg_spacing/2+w_wg,
        ##                           wg_spacing/2+w_wg]))

        #x_pts.append(np.array([position2]))
        #y_pts.append(Y/2+np.array([w_wg_out/2+wg_shift]))


        #x_pts.append(np.array([X-Lio-w_pml, X, X, X-Lio-w_pml]))
        #y_pts.append(Y/2+np.array([w_wg_out/2+wg_shift, w_wg_out/2+wg_shift,
        #                           -w_wg_out/2+wg_shift,-w_wg_out/2+wg_shift]))


        #x_pts.append(np.array([position2]))
        #y_pts.append(Y/2+np.array([-w_wg_out/2+wg_shift]))

        #x_pts.append(np.array([position1-5*1.5*dx]))
        #y_pts.append(np.array([Y/2-12*1.5*dx]))


        #x_pts.append(np.array([w_pml+Lio, 0, 0, w_pml+Lio]))
        #y_pts.append(Y/2+np.array([-wg_spacing/2-w_wg,
        #                           -wg_spacing/2-w_wg,
        #                           -wg_spacing/2,
        #                           -wg_spacing/2]))
        #x_pts.append(np.array([position1-10*1.5*dx]))
        #y_pts.append(np.array([Y/2-3*1.5*dx]))

        #x_pts.append(np.array([position1+10*1.5*dx, position1]))
        #y_pts.append(Y/2+np.array([-3*1.5*dx, 3*1.5*dx]))
        #
        #x_pts.append(np.array([w_pml+Lio]))
        #y_pts.append(Y/2+np.array([wg_spacing/2]))


        #inds_intra = [False, True, False, True, True, False, True, True, False]
        #inds_inter = [False, True, True, True, True, True, True, True, True]


        #x, y, inds = SpatialMux.create_designable_polygon(x_pts, y_pts,
        #                                                   inds_intra=inds_intra,
        #                                                   inds_inter=inds_inter,
        #                                                   ds=3*dx)

        ## add last point
        #x = np.append(x, x[0])
        #y = np.append(y, y[0])
        #inds = np.append(inds, inds[-1]+1)

        blabla = h5py.File('optResult.h5','r')
        x = blabla['x']
        y = blabla['y']
        inds = blabla['inds']

        x = np.copy(x)
        y = np.copy(y)
        inds = np.array([i for i in inds],dtype='int16')
        # remove bad points
        #for i in range(y.size):
        #    if x[i] >= 5.5 and x[i]<=5.6 and y[i]>3.1:
        #        y[i] = y[i] +0.1

        #x = np.concatenate((x[:162],x[176:]))
        #y = np.concatenate((y[:162],y[176:]))

        #inds = np.concatenate((inds[:163],inds[175:]))
        #inds[163:] = inds[163:]-12



        inds = [i for i in range(x.size) if x[i]>=2.61 and x[i]<=10.59]

        roc_bla = 0.2

        make_round = [False for i in x]
        for i in inds: make_round[i] = True
        #make_round[11] = True
        make_round[14] = True

        x, y = emopt.geometry.fillet(x, y, roc_bla, make_round=make_round,
                                     points_per_90=12)

        #for i in range(x.size):
        #    if x[i] >=5.2 and x[i] <= 5.8 and y[i]>=2.8 and y[i]<=3.2:
        #        x[i] += 0.3
        #        y[i] -= 0.1
        x, y = emopt.geometry.populate_lines(x, y, ds=dx,
                                                     refine_box=[w_pml+Lio,
                                                                 X-w_pml-Lio,
                                                                 0, 6])
        inds = [i for i in range(x.size) if (x[i]>w_pml+Lio and
                                                     x[i]<X-w_pml-Lio)]

        x_new = x
        y_new = y
        inds_new = inds
        #inds = inds_new

        if NOT_PARALLEL:
            #print inds
            import matplotlib.pyplot as plt
            f=plt.figure()
            ax = f.add_subplot(111)
            ax.plot(x_new,y_new,'-ro')
            ax.plot(x_new[inds_new], y_new[inds_new], 'ko')
            plt.axis("equal")
            plt.show()



        spacer = emopt.grid.Polygon()
        spacer.set_points(x, y)
        spacer.layer = 2; spacer.material_value = n_SiO2**2

        waveguide = emopt.grid.Polygon()
        waveguide.set_points(x, y)
        waveguide.layer = 2; waveguide.material_value = n_InP**2

        lower = emopt.grid.Polygon()
        lower.set_points(x, y)
        lower.layer = 2; lower.material_value = n_SiO2**2

        # background air
        bg_air = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
        bg_air.layer = 3; bg_air.material_value = 1.0

        # background substrate
        bg_substrate = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
        bg_substrate.layer = 3; bg_substrate.material_value = n_SiO2**2


        eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)

        eps.add_primitive(spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
        eps.add_primitive(waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
        eps.add_primitive(lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)

        eps.add_primitive(bg_substrate, 0, Z/2-h_waveguide/2-h_lowerRidge)
        eps.add_primitive(bg_air, Z/2-h_waveguide/2-h_lowerRidge, Z)
        ZminD = Z/2-h_waveguide/2-h_lowerRidge
        ZmaxD = Z/2+h_waveguide/2


        mu = emopt.grid.ConstantMaterial3D(1.0)

        sim1.set_materials(eps, mu)
        sim1.build()
        sim2.set_materials(eps, mu)
        sim2.build()

        ###############################################################################
        # Setup the sources
        ###############################################################################
        input_slice = emopt.misc.DomainCoordinates(w_pml+5*dx, w_pml+5*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

        #slices= emopt.grid.DomainCoordinates(X/2, X/2, w_pml, Y-w_pml, w_pml, Z-w_pml,
        #                                    dx, dy, dz)
        #slices2 = emopt.grid.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml,
        #                                       Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)
        slices3 =emopt.grid.DomainCoordinates(0, X, 0, Y, Z/2, Z/2, dx, dy, dz)



        mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.4, neigs=6)

        mode.bc = '00'
        mode.build()
        mode.solve()

        Jxs = []
        Jys = []
        Jzs = []
        Mxs = []
        Mys = []
        Mzs = []
        phases = []

        eps_arr = eps.get_values_in(slices3, squeeze=True)

        if(NOT_PARALLEL):
            import matplotlib.pyplot as plt
            f2=plt.figure()
            ax=f2.add_subplot(111)
            pos = ax.imshow(np.real(eps_arr), extent=[0, X, 0, Y], vmin=0, vmax=15,
                      cmap='inferno')
            f2.colorbar(pos, ax=ax)
            plt.show()

        for i in range(2):
            Jx0, Jy0, Jz0, Mx0, My0, Mz0 = mode.get_source(i, dx, dy, dz)

            Jxs.append(Jx0)
            Jys.append(Jy0)
            Jzs.append(Jz0)
            Mxs.append(Mx0)
            Mys.append(My0)
            Mzs.append(Mz0)
            Eym = mode.get_field_interp(i, 'Ey')
            phases.append(np.angle(Eym[10,10]))
            if (NOT_PARALLEL):
                print mode.neff[i]
        #        import matplotlib.pyplot as plt
        #        Eym=Eym.squeeze()
        #        vmin = np.min(np.real(Eym))
        #        vmax = np.max(np.real(Eym.real))
        #        ax = f.add_subplot(4,1,i+1)
        #        ax.imshow(np.real(Eym), extent=[0, Y, 0, Z], vmin=vmin, vmax=-vmin,
        #                  cmap='seismic')
        #
        #if(NOT_PARALLEL):
        #    plt.show()

        phaseFundamental = phases[0] - phases[1]

        JxFund = Jxs[0] + Jxs[1]*np.exp(1j * phaseFundamental)
        JyFund = Jys[0] + Jys[1]*np.exp(1j * phaseFundamental)
        JzFund = Jzs[0] + Jzs[1]*np.exp(1j * phaseFundamental)
        MxFund = Mxs[0] + Mxs[1]*np.exp(1j * phaseFundamental)
        MyFund = Mys[0] + Mys[1]*np.exp(1j * phaseFundamental)
        MzFund = Mzs[0] + Mzs[1]*np.exp(1j * phaseFundamental)

        Jx2 = Jxs[0] - Jxs[1]*np.exp(1j * phaseFundamental)
        Jy2 = Jys[0] - Jys[1]*np.exp(1j * phaseFundamental)
        Jz2 = Jzs[0] - Jzs[1]*np.exp(1j * phaseFundamental)
        Mx2 = Mxs[0] - Mxs[1]*np.exp(1j * phaseFundamental)
        My2 = Mys[0] - Mys[1]*np.exp(1j * phaseFundamental)
        Mz2 = Mzs[0] - Mzs[1]*np.exp(1j * phaseFundamental)





        src1 = [JxFund, JyFund, JzFund, MxFund, MyFund, MzFund]
        src2 = [Jx2, Jy2, Jz2, Mx2, My2, Mz2]

        src1 = COMM.bcast(src1, root=0)
        src2 = COMM.bcast(src2, root=0)

        sim1.set_sources(src1, input_slice)
        sim2.set_sources(src2, input_slice)


        ###############################################################################
        ## Mode match for optimization
        ################################################################################
        fom_slice = emopt.misc.DomainCoordinates(X-w_pml-5*dx, X-w_pml-5*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

        fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45, neigs=6)

        fom_mode.bc = '00'
        fom_mode.build()
        fom_mode.solve()

        Exm0 = fom_mode.get_field_interp(0, 'Ex')
        Eym0 = fom_mode.get_field_interp(0, 'Ey')
        Ezm0 = fom_mode.get_field_interp(0, 'Ez')
        Hxm0 = fom_mode.get_field_interp(0, 'Hx')
        Hym0 = fom_mode.get_field_interp(0, 'Hy')
        Hzm0 = fom_mode.get_field_interp(0, 'Hz')

        Exm1 = fom_mode.get_field_interp(1, 'Ex')
        Eym1 = fom_mode.get_field_interp(1, 'Ey')
        Ezm1 = fom_mode.get_field_interp(1, 'Ez')
        Hxm1 = fom_mode.get_field_interp(1, 'Hx')
        Hym1 = fom_mode.get_field_interp(1, 'Hy')
        Hzm1 = fom_mode.get_field_interp(1, 'Hz')


        if(NOT_PARALLEL):
            Nz, Ny = Exm0.shape
            Exm0 = np.reshape(Exm0, (Nz, Ny, 1))
            Eym0 = np.reshape(Eym0, (Nz, Ny, 1))
            Ezm0 = np.reshape(Ezm0, (Nz, Ny, 1))
            Hxm0 = np.reshape(Hxm0, (Nz, Ny, 1))
            Hym0 = np.reshape(Hym0, (Nz, Ny, 1))
            Hzm0 = np.reshape(Hzm0, (Nz, Ny, 1))
            Exm1 = np.reshape(Exm1, (Nz, Ny, 1))
            Eym1 = np.reshape(Eym1, (Nz, Ny, 1))
            Ezm1 = np.reshape(Ezm1, (Nz, Ny, 1))
            Hxm1 = np.reshape(Hxm1, (Nz, Ny, 1))
            Hym1 = np.reshape(Hym1, (Nz, Ny, 1))
            Hzm1 = np.reshape(Hzm1, (Nz, Ny, 1))
            print fom_mode.neff[0]
            print fom_mode.neff[1]
            print fom_mode.neff[2]

        mode_match1 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm0, Eym0, Ezm0, Hxm0,
                                               Hym0, Hzm0)
        mode_match2 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm1, Eym1, Ezm1, Hxm1,
                                              Hym1, Hzm1)
        ###############################################################################
        # Setup the AdjointMethod objected needed for gradient calculations
        ###############################################################################
        field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, Z/2,
                                                     Z/2, dx, dy, dz)


        sim1.field_domains = [fom_slice, field_monitor]
        am1 = SpatialMuxAdjointMethod(sim1, spacer, waveguide, lower, inds, x, y, ZminD, ZmaxD, fom_slice, mode_match1)

        sim2.field_domains = [fom_slice, field_monitor]
        am2 = SpatialMuxAdjointMethod(sim2, spacer, waveguide, lower, inds, x, y, ZminD, ZmaxD, fom_slice, mode_match2)

        am = MultiObjective([am1, am2])

        params = np.zeros(2*len(inds))

        self.am = am
        self.am1 = am1
        self.am2 = am2
        self.sim1 = sim1
        self.sim2 = sim2
        self.params0 = params
        self.field_monitor = field_monitor

    @staticmethod
    def create_designable_polygon(x_pts, y_pts, inds_intra, inds_inter, ds):
        x = np.array([])
        y = np.array([])
        inds = np.array([], 'int16')

        for i in range(len(x_pts)):
            if inds_inter[i]:
                if i==0:
                    xs, ys = emopt.grid.Polygon.populate_lines([x_pts[-1][-1],
                                                               x_pts[0][0]],
                                                               [y_pts[-1][-1],
                                                                y_pts[0][0]],
                                                               ds)
                    xs = xs[1:-1]
                    ys = ys[1:-1]
                else:
                    xs, ys = emopt.grid.Polygon.populate_lines([x_pts[i-1][-1],
                                                               x_pts[i][0]],
                                                               [y_pts[i-1][-1],
                                                               y_pts[i][0]], ds)
                    xs = xs[1:-1]
                    ys = ys[1:-1]
                N1 = x.size
                x = np.concatenate((x,xs))
                y = np.concatenate((y,ys))
                N2 = x.size
                inds_temp = np.arange(N1, N2)
                inds = np.concatenate((inds, inds_temp))

            if inds_intra[i]:
                xs = np.array([])
                ys = np.array([])

                for j in range(len(x_pts[i])):
                    if j==0:
                        xf = np.array([x_pts[i][0]])
                        yf = np.array([y_pts[i][0]])
                    else:
                        xf, yf = emopt.grid.Polygon.populate_lines([x_pts[i][j-1],
                                                                    x_pts[i][j]],
                                                                    [y_pts[i][j-1],y_pts[i][j]],
                                                                    ds)
                        xf = xf[1:]
                        yf = yf[1:]
                    xs = np.concatenate((xs,xf))
                    ys = np.concatenate((ys,yf))
                N1 = x.size
                x = np.concatenate((x,xs))
                y = np.concatenate((y,ys))
                N2 = x.size
                inds_temp = np.arange(N1,N2)
                inds = np.concatenate((inds, inds_temp))

            else:
                x = np.concatenate((x,x_pts[i]))
                y = np.concatenate((y,y_pts[i]))

        return x, y, inds



if __name__ == '__main__':
    mux = SpatialMux(1.55)
    am = mux.am
    am1 = mux.am1
    am2 = mux.am2

    sim1 = mux.sim1
    sim2 = mux.sim2
    params = mux.params0
    field_monitor = mux.field_monitor

    print len(params)
    
    #params_test = np.copy(params)
    #params_test[:] = 0.1

    #am1.get_current_points(params_test)

    #check_indices = np.arange(0, len(params), 5)
    #am.check_gradient(params, indices = check_indices)

    fom_history = []
    fom1_history = []
    fom2_history = []
    eff_history = []
    callback = lambda p: opt_finish_iteration(p, sim1, am1, sim2, am2,
                                              fom_history, fom1_history,
                                              fom2_history, eff_history, field_monitor)

    ###############################################################################
    # Setup and run the simulation
    ###############################################################################
    opt = emopt.optimizer.Optimizer(am, params, Nmax=100, opt_method='L-BFGS-B',
                                   callback_func=callback)
    fom, pfinal = opt.run()

