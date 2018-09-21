import emopt
from emopt.misc import NOT_PARALLEL, run_on_master, COMM
from emopt.adjoint_method import AdjointMethod, AdjointMethodMO

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI

####################################################################################
# Combiner class
####################################################################################
class IncoherentCombinerAdjointMethod(AdjointMethod):

    def __init__(self, sim, combiner_spacer, combiner_waveguide, combiner_lower, fom_domain, mode_match):
        super(IncoherentCombinerAdjointMethod, self).__init__(sim, step=1e-5)
        self.combiner_spacer = combiner_spacer
        self.combiner_waveguide = combiner_waveguide
        self.combiner_lower = combiner_lower
        self.mode_match = mode_match
        self.fom_domain = fom_domain

    def update_system(self, params):
        self.combiner_spacer.height = params[0]
        self.combiner_spacer.width = params[1]
        self.combiner_waveguide.height = params[0]
        self.combiner_waveguide.width = params[1]
        self.combiner_lower.height = params[0]
        self.combiner_lower.width = params[1]

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
def findRelativePhase(A,B):
    # Assumes A and B are of equal shape
    phaseA = []
    phaseB = []
    for x in np.nditer(A, flags=["refs_ok"]):
        phaseBla = np.angle(x)
	phaseA.append(phaseBla)

    for x in np.nditer(B, flags=["refs_ok"]):
        phaseBla = np.angle(x)
	phaseB.append(phaseBla)

    phase = [x-y for x,y in zip(phaseA,phaseB)]

    if all(x==phase[0] for x in phase):
        return phase[0]
    else:
        raise ValueError

####################################################################################
# Simulation parameters
####################################################################################
X = 10000.0
Y = 6000.0
Z = 3000.0
dx = 40.0
dy = 40.0
dz = 40.0

wavelength = 1550.0

####################################################################################
# Setup simulations
####################################################################################
w_pml = dx * 15

sim1 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-6, min_rindex=1.0)
sim1.src_ramp_time = sim1.Nlambda * 20
sim1.Nmax = sim1.Nlambda * 500


sim1.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
sim1.bc = '000'


sim2 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-6, min_rindex=1.0)
sim2.src_ramp_time = sim2.Nlambda * 20
sim2.Nmax = sim2.Nlambda * 500

sim2.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
sim2.bc = '000'


X = sim1.X
Y = sim1.Y
Z = sim1.Z

####################################################################################
# Define the geometry/materials
####################################################################################
w_wg = 800.0
L_in = X/2
L_out = X/2
L_combiner = 5000.0
w_combiner = 2800.0
h_spacer = 40.0
h_waveguide = 180.0
h_lowerRidge = 500.0
h_substrate = 660.0
wg_spacing = 800.0

n_InP = 3.4
n_SiO2 = 1.444

## Ridge spacer
#wg_in_top_spacer = emopt.grid.Rectangle(X/4, Y/2+wg_spacing/2+w_wg/2, L_in, w_wg)
#wg_in_top_spacer.layer = 1; wg_in_top_spacer.material_value = n_SiO2**2
#
#wg_in_bottom_spacer = emopt.grid.Rectangle(X/4, Y/2-wg_spacing/2-w_wg/2, L_in, w_wg)
#wg_in_bottom_spacer.layer = 1; wg_in_bottom_spacer.material_value = n_SiO2**2
#
#wg_out_spacer = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg)
#wg_out_spacer.layer = 1; wg_out_spacer.material_value = n_SiO2**2
#
#combiner_spacer = emopt.grid.Rectangle(X/2, Y/2, L_combiner, w_combiner)
#combiner_spacer.layer = 1; combiner_spacer.material_value = n_SiO2**2
#
## Ridge waveguide
#wg_in_top_waveguide = emopt.grid.Rectangle(X/4, Y/2+wg_spacing/2+w_wg/2, L_in, w_wg)
#wg_in_top_waveguide.layer = 1; wg_in_top_waveguide.material_value = n_InP**2
#
#wg_in_bottom_waveguide = emopt.grid.Rectangle(X/4, Y/2-wg_spacing/2-w_wg/2, L_in, w_wg)
#wg_in_bottom_waveguide.layer = 1; wg_in_bottom_waveguide.material_value = n_InP**2
#
#wg_out_waveguide = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg)
#wg_out_waveguide.layer = 1; wg_out_waveguide.material_value = n_InP**2
#
#combiner_waveguide = emopt.grid.Rectangle(X/2, Y/2, L_combiner, w_combiner)
#combiner_waveguide.layer = 1; combiner_waveguide.material_value = n_InP**2
#
## Ridge lower
#wg_in_top_lower = emopt.grid.Rectangle(X/4, Y/2+wg_spacing/2+w_wg/2, L_in, w_wg)
#wg_in_top_lower.layer = 1; wg_in_top_lower.material_value = n_SiO2**2
#
#wg_in_bottom_lower = emopt.grid.Rectangle(X/4, Y/2-wg_spacing/2-w_wg/2, L_in, w_wg)
#wg_in_bottom_lower.layer = 1; wg_in_bottom_lower.material_value = n_SiO2**2
#
#wg_out_lower = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg)
#wg_out_lower.layer = 1; wg_out_lower.material_value = n_SiO2**2
#
#combiner_lower = emopt.grid.Rectangle(X/2, Y/2, L_combiner, w_combiner)
#combiner_lower.layer = 1; combiner_lower.material_value = n_SiO2**2
#
## background air
#bg_air = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
#bg_air.layer = 2; bg_air.material_value = 1.0
#
## background substrate
#bg_substrate = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
#bg_substrate.layer = 2; bg_substrate.material_value = n_SiO2**2
#
#
#
#eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)
#
#eps.add_primitive(wg_in_top_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
#eps.add_primitive(wg_in_top_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
#eps.add_primitive(wg_in_top_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)
#
#eps.add_primitive(wg_in_bottom_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
#eps.add_primitive(wg_in_bottom_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
#eps.add_primitive(wg_in_bottom_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)
#
#eps.add_primitive(wg_out_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
#eps.add_primitive(wg_out_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
#eps.add_primitive(wg_out_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)
#
#eps.add_primitive(combiner_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
#eps.add_primitive(combiner_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
#eps.add_primitive(combiner_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)



pts_x_static1 = np.array([
    0, X/2-L_combiner/2, X/2-L_combiner/2,
    X/2+L_combiner/2, X/2+L_combiner/2, X,
    X, X/2+L_combiner/2, X/2+L_combiner/2,
    X/2-L_combiner/2, X/2-L_combiner/2, 0,
    0, X/2-L_combiner/2, X/2-L_combiner,
    0
    ])
pts_y_static1 = np.array([
    Y/2+wg_spacing/2+w_wg, Y/2+wg_spacing/2+w_wg, Y/2+w_combiner/2,
    Y/2+w_combiner/2, Y/2+w_wg/2, Y/2+w_wg/2,
    Y/2-w_wg/2, Y/2-w_wg/2, Y/2-w_combiner/2,
    Y/2-w_combiner/2, Y/2-wg_spacing/2-w_wg, Y/2-wg_spacing/2-w_wg,
    Y/2-wg_spacing/2, Y/2-wg_spacing/2, Y/2+wg_spacing/2,
    Y/2+wg_spacing/2
    ])

pts_x_static1 = np.array([
    0, X/2-L_combiner/2, X/2-L_combiner/2,
    X/2+L_combiner/2, X/2+L_combiner/2, X,
    X, X/2+L_combiner/2, X/2+L_combiner/2,
    X/2-L_combiner/2, X/2-L_combiner/2, 0,
    0, X/2-L_combiner/2, X/2-L_combiner,
    0
    ])
pts_y_static1 = np.array([
    Y/2+wg_spacing/2+w_wg, Y/2+wg_spacing/2+w_wg, Y/2+w_combiner/2,
    Y/2+w_combiner/2, Y/2+w_wg/2, Y/2+w_wg/2,
    Y/2-w_wg/2, Y/2-w_wg/2, Y/2-w_combiner/2,
    Y/2-w_combiner/2, Y/2-wg_spacing/2-w_wg, Y/2-wg_spacing/2-w_wg,
    Y/2-wg_spacing/2, Y/2-wg_spacing/2, Y/2+wg_spacing/2,
    Y/2+wg_spacing/2
    ])

pts_x_static1 = np.array([
    0, X/2-L_combiner/2, X/2-L_combiner/2,
    X/2+L_combiner/2, X/2+L_combiner/2, X,
    X, X/2+L_combiner/2, X/2+L_combiner/2,
    X/2-L_combiner/2, X/2-L_combiner/2, 0,
    0, X/2-L_combiner/2, X/2-L_combiner,
    0
    ])
pts_y_static1 = np.array([
    Y/2+wg_spacing/2+w_wg, Y/2+wg_spacing/2+w_wg, Y/2+w_combiner/2,
    Y/2+w_combiner/2, Y/2+w_wg/2, Y/2+w_wg/2,
    Y/2-w_wg/2, Y/2-w_wg/2, Y/2-w_combiner/2,
    Y/2-w_combiner/2, Y/2-wg_spacing/2-w_wg, Y/2-wg_spacing/2-w_wg,
    Y/2-wg_spacing/2, Y/2-wg_spacing/2, Y/2+wg_spacing/2,
    Y/2+wg_spacing/2
    ])


spacer = emopt.grid.Polygon()
spacer.set_points(pts_x, pts_y)
spacer.material_value = n_SiO2**2
spacer.layer = 1

waveguide = emopt.grid.Polygon()
waveguide.set_poitns(pts_x,pts_y)
waveguide.material_value = n_InP**2
waveguide.layer = 1

ridge = emopt.grid.Polygon()
ridge.set_poitns(pts_x,pts_y)
ridge.material_value = n_SiO2**2
ridge.layer = 1






eps.add_primitive(bg_substrate, 0, Z/2-h_waveguide/2-h_lowerRidge)
eps.add_primitive(bg_air, Z/2-h_waveguide/2-h_lowerRidge, Z)


mu = emopt.grid.ConstantMaterial3D(1.0)

sim1.set_materials(eps, mu)
sim1.build()

sim2.set_materials(eps, mu)
sim2.build()

###############################################################################
# Setup the sources
###############################################################################
input_slice = emopt.misc.DomainCoordinates(w_pml+3*dx, w_pml+3*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

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

for i in range(4):
    Jx0, Jy0, Jz0, Mx0, My0, Mz0 = mode.get_source(i, dx, dy, dz)
    Jxs.append(Jx0)
    Jys.append(Jy0)
    Jzs.append(Jz0)
    Mxs.append(Mx0)
    Mys.append(My0)
    Mzs.append(Mz0)
    Eym = mode.get_field_interp(i, 'Ey')
    phases.append(np.angle(Eym[0,0]))

phaseFundamental = phases[0] - phases[1]
phaseHigher = phases[2] - phases[3]


JxFund = Jxs[0] + Jxs[1]*np.exp(1j * phaseFundamental)
JyFund = Jys[0] + Jys[1]*np.exp(1j * phaseFundamental)
JzFund = Jzs[0] + Jzs[1]*np.exp(1j * phaseFundamental)
MxFund = Mxs[0] + Mxs[1]*np.exp(1j * phaseFundamental)
MyFund = Mys[0] + Mys[1]*np.exp(1j * phaseFundamental)
MzFund = Mzs[0] + Mzs[1]*np.exp(1j * phaseFundamental)

JxHigher = Jxs[2] - Jxs[3]*np.exp(1j * phaseHigher)
JyHigher = Jys[2] - Jys[3]*np.exp(1j * phaseHigher)
JzHigher = Jzs[2] - Jzs[3]*np.exp(1j * phaseHigher)
MxHigher = Mxs[2] - Mxs[3]*np.exp(1j * phaseHigher)
MyHigher = Mys[2] - Mys[3]*np.exp(1j * phaseHigher)
MzHigher = Mzs[2] - Mzs[3]*np.exp(1j * phaseHigher)








src1 = [JxFund, JyFund, JzFund, MxFund, MyFund, MzFund]
src2 = [JxHigher, JyHigher, JzHigher, MxHigher, MyHigher, MzHigher]

src1 = COMM.bcast(src1, root=0)
src2 = COMM.bcast(src2, root=0)


sim1.set_sources(src1, input_slice)
sim2.set_sources(src2, input_slice)

##############################################################################
# View each simulation
##############################################################################

#sim1.solve_forward()
#
#field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y,
#                                             Z/2, Z/2, dx, dy, dz)
#Ey = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)
#
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#
#    vmax = np.max(np.abs(Ey))
#    f = plt.figure()
#    ax1 = f.add_subplot(211)
#    ax1.imshow(np.abs(Ey), extent = [0,X,0,Y], vmin=0,
#              vmax=vmax, cmap='plasma')
#
#field_monitor2 = emopt.misc.DomainCoordinates(X/2, X/2, 0, Y, 0, Z, dx, dy, dz)
#
#Ey = sim1.get_field_interp('Ey', domain=field_monitor2, squeeze=True)
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#
#    vmax = np.max(np.real(Ey))
#    f2 = plt.figure()
#    ax1 = f2.add_subplot(211)
#    ax1.imshow(np.real(Ey), extent = [0,Y,0,Z], vmin=-vmax,
#              vmax=vmax, cmap='seismic')
#
#
#
#
#sim2.solve_forward()
#
#Ey = sim2.get_field_interp('Ey', domain=field_monitor, squeeze=True)
#
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#
#    vmax = np.max(np.abs(Ey))
#    ax2 = f.add_subplot(212)
#    ax2.imshow(np.abs(Ey), extent = [0,X,0,Y], vmin=0,
#              vmax=vmax, cmap='plasma')
#
#
#
#Ey = sim2.get_field_interp('Ey', domain=field_monitor2, squeeze=True)
#
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#
#    vmax = np.max(np.real(Ey))
#    ax2 = f2.add_subplot(212)
#    ax2.imshow(np.real(Ey), extent = [0,Y,0,Z], vmin=-vmax,
#              vmax=vmax, cmap='seismic')
#    plt.show()



###############################################################################
# Mode match for optimization
###############################################################################
fom_slice = emopt.misc.DomainCoordinates(X-w_pml-3*dx, X-w_pml-3*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

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
params = np.array([w_combiner, L_combiner])

sim1.field_domains = [fom_slice]
am1 = IncoherentCombinerAdjointMethod(sim1, combiner_spacer,combiner_waveguide,combiner_lower, fom_slice, mode_match1)

sim2.field_domains = [fom_slice]
am2 = IncoherentCombinerAdjointMethod(sim2, combiner_spacer,combiner_waveguide,combiner_lower, fom_slice, mode_match2)

ams = []
ams.append(am1)
ams.append(am2)

am = MultiObjective(ams)

am.check_gradient(params)

###############################################################################
# Setup and run the simulation
###############################################################################
opt = emopt.optimizer.Optimizer(am, params, Nmax=10, opt_method='L-BFGS-B')
fom, pfinal = opt.run()

am.fom(pfinal)

field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y, Z/2,
                                             Z/2, dx, dy, dz)

Ey1 = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)
Ey2 = sim2.get_field_interp('Ey', domain=field_monitor, squeeze=True)
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    Ey1 = np.concatenate([Ey1[::-1],Ey1], axis=0)

    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
    vmax = np.max(np.real(Ey1))
    f=plt.figure()
    ax1 = f.add_subplot(211)
    ax1.imshow(np.real(Ey1), extent=[0,X,0,Y],vmin=-vmax, vmax=vmax, cmap='seismic')
    
    Ey2 = np.concatenate([Ey2[::-1],Ey2], axis=0)

    vmax = np.max(np.real(Ey2))
    ax2 = f.addsubplot(212)
    ax2.imshow(np.real(Ey2), extent=[0,X,0,Y], vmin=-vmax, vmax =vmax, cmap='seismic')
    
    plot.show()
