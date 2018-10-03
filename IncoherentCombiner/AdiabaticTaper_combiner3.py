import emopt
from emopt.misc import NOT_PARALLEL, run_on_master, COMM
from emopt.adjoint_method import AdjointMethod, AdjointMethodMO

import numpy as np
from math import pi
import copy

from petsc4py import PETSc
from mpi4py import MPI

####################################################################################
# Combiner class
####################################################################################
class IncoherentCombinerAdjointMethod(AdjointMethod):

    def __init__(self, sim, spacer, waveguide, lower, indices, xs, ys, ZminD,
                 ZmaxD, fom_domain, mode_match):
        super(IncoherentCombinerAdjointMethod, self).__init__(sim, step=1e-5)
        self.spacer = spacer
        self.waveguide = waveguide
        self.lower = lower
        self.indices = indices
        self.xs = xs
        self.ys = ys
        self.mode_match = mode_match
        self.fom_domain = fom_domain
        self.ZminD = ZminD
        self.ZmaxD = ZmaxD

    def update_system(self, params):
        # params = [position1, position2, Hole_x, Hole_y, Hole_length, Hole_height]

        N = len(params)/2;

        for i in range(len(self.indices)):
            self.spacer.set_point(self.indices[i], params[i], params[i+N])
            self.waveguide.set_point(self.indices[i], params[i], params[i+N])
            self.lower.set_point(self.indices[i], params[i], params[i+N])


    @run_on_master
    def calc_fom(self, sim, params):
        Ex, Ey, Ez, Hx, Hy, Hz = sim.saved_fields[0]
        Psrc = sim.source_power

        self.mode_match.compute(Ex, Ey, Ez, Hx, Hy, Hz)
        fom = -1*self.mode_match.get_mode_match_forward(1.0)

        return fom/Psrc

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
        """ Our FOM does not depend explicitly on deisgn parameters """
        return np.zeros(len(params))

    def get_update_boxes(self, sim, params):
        x = self.waveguide.xs
        y = self.waveguide.ys

        dx = sim.dx
        dy = sim.dy
        dz = sim.dz

        ids = self.indices
        N = len(ids)

        update_boxes = []
        for i in range(N):
            if(i<N-1):
                xmin = np.min([x[ids[i]-1], x[ids[i]], x[ids[i]+1]])-2*dx
                xmax = np.max([x[ids[i]-1], x[ids[i]], x[ids[i]+1]])+2*dx
                ymin = np.min([y[ids[i]-1], y[ids[i]], y[ids[i]+1]])-2*dy
                ymax = np.max([y[ids[i]-1], y[ids[i]], y[ids[i]+1]])+2*dy
                ubs = [xmin, xmax, ymin, ymax, self.ZminD-2*dz, self.ZmaxD+2*dz]
            else:
                xmin = np.min([x[ids[i]-1], x[ids[i]], x[0]])-2*dx
                xmax = np.max([x[ids[i]-1], x[ids[i]], x[0]])+2*dx
                ymin = np.min([y[ids[i]-1], y[ids[i]], y[0]])-2*dy
                ymax = np.max([y[ids[i]-1], y[ids[i]], y[0]])+2*dy
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
# Class to compute superposed mode sources
###################################################################################
#class SuperposedMode(ModeFullVector)
#    


###################################################################################
# Definitions
###################################################################################

####################################################################################
# Simulation parameters
####################################################################################
Xsim = 7000.0
Ysim = 5000.0
Zsim = 2000.0
dx = 40.0
dy = 40.0
dz = 40.0

wavelength = 1550.0


####################################################################################
# Setup simulations
####################################################################################
w_pml = dx * 15

X = Xsim + 2*w_pml
Y = Ysim + 2*w_pml
Z = Zsim + 2*w_pml


sim1 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-6, min_rindex=1.0)
sim1.src_ramp_time = sim1.Nlambda * 20
sim1.Nmax = sim1.Nlambda * 1000

sim1.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
sim1.bc = '000'


sim2 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-6, min_rindex=1.0)
sim2.src_ramp_time = sim2.Nlambda * 20
sim2.Nmax = sim2.Nlambda * 1000

sim2.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
sim2.bc = '000'




X = sim1.X
Y = sim1.Y
Z = sim1.Z

####################################################################################
# Define the geometry/materials
####################################################################################
w_wg = 550.0
w_wg_out = 900.0
wg_spacing = 1200.0
L_in = X/2
L_out = X/2

h_spacer = 40.0
h_waveguide = 180.0
h_lowerRidge = 500.0


n_InP = 3.4
n_SiO2 = 1.444


# taper
position1 = w_pml+10*dx+2*X/5
position2 = 4*X/5

#Hole_x = 3*X/5-200
#Hole_y = Y/2+6*dx
#Hole_length = 1.8*w_wg
#Hole_height = w_wg/2


endIn = w_pml+10*dx


xs = np.array([])
ys = np.array([])


x_pts = []
y_pts = []

#Fixed points
x_pts.append(np.array([endIn, 0, 0, endIn]))
x_pts.append(np.array([position2, X, X, position2]))
x_pts.append(np.array([endIn, 0 , 0, endIn]))

y_pts.append(np.array([wg_spacing/2, wg_spacing/2, wg_spacing/2+w_wg,
                       wg_spacing/2+w_wg])+Y/2)
y_pts.append(np.array([w_wg_out/2, w_wg_out/2, -w_wg_out/2, -w_wg_out/2])+Y/2)
y_pts.append(np.array([-wg_spacing/2-w_wg, -wg_spacing/2-w_wg, -wg_spacing/2,
                       -wg_spacing/2])+Y/2)


# Create changeable points (parameters)

indices = [];

ds = 2*dx

x_p = []
y_p = []

for i in range(len(x_pts)-1):
    xf, yf = emopt.grid.Polygon.populate_lines([x_pts[i][-1],x_pts[i+1][0]],
                                               [y_pts[i][-1],y_pts[i+1][0]], ds)

    indices.append(xf.size-3)
    x_p.append(xf[1:-2])
    y_p.append(yf[1:-2])

    xs = np.concatenate((xs, x_pts[i], xf[1:-2]))
    ys = np.concatenate((ys, y_pts[i], yf[1:-2]))


xf, yf = emopt.grid.Polygon.populate_lines([endIn, position1, endIn],
                                           [Y/2-wg_spacing/2, Y/2,
                                            Y/2+wg_spacing/2], ds)
indices.append(xf.size-3)
xs = np.concatenate((xs, x_pts[-1], xf[1:-2]))
ys = np.concatenate((ys, y_pts[-1], yf[1:-2]))

x_p.append(xf[1:-2])
y_p.append(yf[1:-2])
print indices
print len(x_p[0]), len(x_p[1]), len(x_p[2])

indices2 = np.array([], dtype='int32')
indices2=np.concatenate((indices2, x_pts[0].size +
                         np.array(range(indices[0]))))
indices2=np.concatenate((indices2, indices2.size + x_pts[0].size + x_pts[1].size +
                         np.array(range(indices[1]))))
indices2=np.concatenate((indices2, indices2.size + x_pts[0].size +
                         x_pts[1].size + x_pts[2].size +
                         np.array(range(indices[2]))))


spacer = emopt.grid.Polygon()
spacer.set_points(xs, ys)
spacer.layer = 2; spacer.material_value = n_SiO2**2

waveguide = emopt.grid.Polygon()
waveguide.set_points(xs, ys)
waveguide.layer = 2; waveguide.material_value = n_InP**2

lower = emopt.grid.Polygon()
lower.set_points(xs, ys)
lower.layer = 2; lower.material_value = n_SiO2**2



#hole= emopt.grid.Rectangle(Hole_x, Hole_y, Hole_length, Hole_height)
#hole.layer = 1; hole.material_value = 1.0





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

#eps.add_primitive(hole, Z/2-h_waveguide/2-h_lowerRidge, Z)


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
input_slice = emopt.misc.DomainCoordinates(18*dx, 18*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

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


for i in range(4):
    Jx0, Jy0, Jz0, Mx0, My0, Mz0 = mode.get_source(i, dx, dy, dz)

    Jxs.append(Jx0)
    Jys.append(Jy0)
    Jzs.append(Jz0)
    Mxs.append(Mx0)
    Mys.append(My0)
    Mzs.append(Mz0)
    Eym = mode.get_field_interp(i, 'Ey')
    phases.append(np.angle(Eym[10,10]))
#    if (NOT_PARALLEL):
#        print mode.neff[i]
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


##############################################################################
# View each simulation
##############################################################################

#sim1.solve_forward()
#sim2.solve_forward()
#field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y,
#                                             Z/2, Z/2, dx, dy, dz)
#Ey = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)


###############################################################################
# Mode match for optimization
###############################################################################
#fom_slice = emopt.misc.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)
#
#fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45, neigs=6)
#
#fom_mode.bc = '00'
#fom_mode.build()
#fom_mode.solve()
#
#
#
#Exm0 = fom_mode.get_field_interp(0, 'Ex')
#Eym0 = fom_mode.get_field_interp(0, 'Ey')
#Ezm0 = fom_mode.get_field_interp(0, 'Ez')
#Hxm0 = fom_mode.get_field_interp(0, 'Hx')
#Hym0 = fom_mode.get_field_interp(0, 'Hy')
#Hzm0 = fom_mode.get_field_interp(0, 'Hz')
#
#Exm1 = fom_mode.get_field_interp(1, 'Ex')
#Eym1 = fom_mode.get_field_interp(1, 'Ey')
#Ezm1 = fom_mode.get_field_interp(1, 'Ez')
#Hxm1 = fom_mode.get_field_interp(1, 'Hx')
#Hym1 = fom_mode.get_field_interp(1, 'Hy')
#Hzm1 = fom_mode.get_field_interp(1, 'Hz')
#
#
#if(NOT_PARALLEL):
#    Nz, Ny = Exm0.shape
#    Exm0 = np.reshape(Exm0, (Nz, Ny, 1))
#    Eym0 = np.reshape(Eym0, (Nz, Ny, 1))
#    Ezm0 = np.reshape(Ezm0, (Nz, Ny, 1))
#    Hxm0 = np.reshape(Hxm0, (Nz, Ny, 1))
#    Hym0 = np.reshape(Hym0, (Nz, Ny, 1))
#    Hzm0 = np.reshape(Hzm0, (Nz, Ny, 1))
#    Exm1 = np.reshape(Exm1, (Nz, Ny, 1))
#    Eym1 = np.reshape(Eym1, (Nz, Ny, 1))
#    Ezm1 = np.reshape(Ezm1, (Nz, Ny, 1))
#    Hxm1 = np.reshape(Hxm1, (Nz, Ny, 1))
#    Hym1 = np.reshape(Hym1, (Nz, Ny, 1))
#    Hzm1 = np.reshape(Hzm1, (Nz, Ny, 1))
#
#
#
#Ex_1 = sim1.get_field_interp('Ex', domain=fom_slice)
#Ey_1 = sim1.get_field_interp('Ey', domain=fom_slice)
#Ez_1 = sim1.get_field_interp('Ez', domain=fom_slice)
#Hx_1 = sim1.get_field_interp('Hx', domain=fom_slice)
#Hy_1 = sim1.get_field_interp('Hy', domain=fom_slice)
#Hz_1 = sim1.get_field_interp('Hz', domain=fom_slice)
#
#
#
#Ex_2 = sim2.get_field_interp('Ex', domain=fom_slice)
#Ey_2 = sim2.get_field_interp('Ey', domain=fom_slice)
#Ez_2 = sim2.get_field_interp('Ez', domain=fom_slice)
#Hx_2 = sim2.get_field_interp('Hx', domain=fom_slice)
#Hy_2 = sim2.get_field_interp('Hy', domain=fom_slice)
#Hz_2 = sim2.get_field_interp('Hz', domain=fom_slice)
#
#if(NOT_PARALLEL):
#    print Ex_1.shape
#
#    Nz, Ny = Ex_1.shape
#    Ex_1 = np.reshape(Ex_1, (Nz, Ny, 1))
#    Ey_1 = np.reshape(Ey_1, (Nz, Ny, 1))
#    Ez_1 = np.reshape(Ez_1, (Nz, Ny, 1))
#    Hx_1 = np.reshape(Hx_1, (Nz, Ny, 1))
#    Hy_1 = np.reshape(Hy_1, (Nz, Ny, 1))
#    Hz_1 = np.reshape(Hz_1, (Nz, Ny, 1))
#    Ex_2 = np.reshape(Ex_2, (Nz, Ny, 1))
#    Ey_2 = np.reshape(Ey_2, (Nz, Ny, 1))
#    Ez_2 = np.reshape(Ez_2, (Nz, Ny, 1))
#    Hx_2 = np.reshape(Hx_2, (Nz, Ny, 1))
#    Hy_2 = np.reshape(Hy_2, (Nz, Ny, 1))
#    Hz_2 = np.reshape(Hz_2, (Nz, Ny, 1))



#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#
#    Eym0_bla=np.squeeze(Eym0)
#    Ey_1_bla=np.squeeze(Ey_1)
#    Eym1_bla=np.squeeze(Eym1)
#    Ey_2_bla=np.squeeze(Ey_2)
#
#    f5=plt.figure()
#    ax = f5.add_subplot(221)
#    vmax = np.max(np.abs(Eym0_bla))
#    pos = ax.imshow(np.abs(Eym0_bla), extent=[0, Y, 0, Z], vmin=0, vmax=vmax,
#              cmap='inferno')
#    f5.colorbar(pos, ax=ax)
#
#    ax = f5.add_subplot(222)
#    vmax = np.max(np.abs(Ey_1_bla))
#    bla = ax.imshow(np.abs(Ey_1_bla), extent=[0, Y, 0, Z], vmin=0, vmax=vmax,
#              cmap='inferno')
#    f5.colorbar(bla, ax=ax)
#
#
#    ax = f5.add_subplot(223)
#    vmax = np.max(np.abs(Eym1_bla))
#    bla = ax.imshow(np.abs(Eym1_bla), extent=[0, Y, 0, Z], vmin=0, vmax=vmax,
#              cmap='inferno')
#    f5.colorbar(bla, ax=ax)
#
#    ax = f5.add_subplot(224)
#    vmax = np.max(np.abs(Ey_2_bla))
#    bla = ax.imshow(np.abs(Ey_2_bla), extent=[0, Y, 0, Z], vmin=0, vmax=vmax,
#              cmap='inferno')
#    f5.colorbar(bla, ax=ax)
#
#
#    plt.show()
#
#
#
#    mode_match1 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm0, Eym0, Ezm0, Hxm0,
#                                           Hym0, Hzm0)
#    Psrc1 = sim1.source_power
#    mode_match1.compute(Ex_1, Ey_1, Ez_1, Hx_1, Hy_1, Hz_1)
#    value1 = mode_match1.get_mode_match_forward(1.0)
#    value1 = value1/Psrc1
#
#    mode_match2 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm1, Eym1, Ezm1, Hxm1,
#                                          Hym1, Hzm1)
#    Psrc2 = sim2.source_power
#    mode_match2.compute(Ex_2, Ey_2, Ez_2, Hx_2, Hy_2, Hz_2)
#    value2 = mode_match2.get_mode_match_forward(1.0)
#    value2 = value2/Psrc2
#    print("mode match fundamental = ", value1)
#    print("mode match second order = ", value2)



#Ey = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#
#    vmax = np.max(np.abs(Ey))
#    f = plt.figure()
#    ax1 = f.add_subplot(211)
#    ax1.imshow(np.abs(Ey), extent = [0,X,0,Y], vmin=0,
#              vmax=vmax, cmap='inferno')
#
#Ey = sim2.get_field_interp('Ey', domain=field_monitor, squeeze=True)
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#
#    vmax = np.max(np.abs(Ey))
#    ax2 = f.add_subplot(212)
#    ax2.imshow(np.abs(Ey), extent = [0,X,0,Y], vmin=0,
#              vmax=vmax, cmap='inferno')
#    plt.show()
#




################################################################################
## Mode match for optimization
################################################################################
fom_slice = emopt.misc.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

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



mode_match1 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm0, Eym0, Ezm0, Hxm0,
                                       Hym0, Hzm0)
mode_match2 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm1, Eym1, Ezm1, Hxm1,
                                      Hym1, Hzm1)
###############################################################################
# Setup the AdjointMethod objected needed for gradient calculations
###############################################################################

sim1.field_domains = [fom_slice]
am1 = IncoherentCombinerAdjointMethod(sim1, spacer, waveguide, lower, indices2,
                                      xs, ys, ZminD, ZmaxD, fom_slice, mode_match1)

sim2.field_domains = [fom_slice]
am2 = IncoherentCombinerAdjointMethod(sim2, spacer, waveguide, lower, indices2,
                                      xs, ys, ZminD, ZmaxD, fom_slice, mode_match2)

ams = []
ams.append(am1)
ams.append(am2)

am = MultiObjective(ams)

params = np.array(np.concatenate((xs[indices2],ys[indices2])))

#am.check_gradient(params)

###############################################################################
# Setup and run the simulation
###############################################################################
opt = emopt.optimizer.Optimizer(am, params, Nmax=5, opt_method='L-BFGS-B')
fom, pfinal = opt.run()

am.fom(pfinal)

if(NOT_PARALLEL):
    Ex111 = sim1.get_field_interp('Ex', domain=fom_slice)
    Ey111 = sim1.get_field_interp('Ey', domain=fom_slice)
    Ez111 = sim1.get_field_interp('Ez', domain=fom_slice)
    Hx111 = sim1.get_field_interp('Hx', domain=fom_slice)
    Hy111 = sim1.get_field_interp('Hy', domain=fom_slice)
    Hz111 = sim1.get_field_interp('Hz', domain=fom_slice)

    Psrc = sim1.source_power
    mode_match1.compute(Ex111, Ey111, Ez111, Hx111, Hy111, Hz111)
    fom1 = -1*mode_match1.get_mode_match_forward(1.0)
    result111=fom1/Psrc

    Ex222 = sim2.get_field_interp('Ex', domain=fom_slice)
    Ey222 = sim2.get_field_interp('Ey', domain=fom_slice)
    Ez222 = sim2.get_field_interp('Ez', domain=fom_slice)
    Hx222 = sim2.get_field_interp('Hx', domain=fom_slice)
    Hy222 = sim2.get_field_interp('Hy', domain=fom_slice)
    Hz222 = sim2.get_field_interp('Hz', domain=fom_slice)

    Psrc = sim2.source_power
    mode_match2.compute(Ex222, Ey222, Ez222, Hx222, Hy222, Hz222)
    fom2 = -1*mode_match2.get_mode_match_forward(1.0)
    result222=fom2/Psrc

    print result111
    print result222







field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, Z/2,
                                             Z/2, dx, dy, dz)

Ey11 = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)
Ey22 = sim2.get_field_interp('Ey', domain=field_monitor, squeeze=True)
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    #Ey = np.concatenate([Ey[::-1],Ey], axis=0)

    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
    vmax = np.max(np.abs(Ey11))
    f=plt.figure()
    ax1 = f.add_subplot(311)
    ax1.imshow(np.abs(Ey11), extent=[0,X-2*w_pml,0,Y-2*w_pml],vmin=0, vmax=vmax,
                cmap='inferno')

    vmax = np.max(np.abs(Ey22))
    ax2 = f.add_subplot(312)
    ax2.imshow(np.abs(Ey22), extent=[0, X-2*w_pml, 0, Y-2*w_pml], vmin=0,
               vmax=vmax, cmap='inferno')

    vmax = np.max(np.abs(eps_arr))
    ax3 = f.add_subplot(313)
    ax3.imshow(np.abs(eps_arr), extent=[0, X-2*w_pml, 0, Y-2*w_pml], vmin=0,
               vmax=vmax, cmap='seismic')

    plt.show()
