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

    def __init__(self, sim, combiner, fom_domain, mode_match):
        super(IncoherentCombinerAdjointMethod, self).__init__(sim, step=1e-5)
        self.combiner = combiner
        self.mode_match = mode_match
        self.fom_domain = fom_domain

    def update_system(self, params):
        self.combiner.height = params[0]
        self.combiner.width = params[1]

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
        return np.zeros(len(parms))


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
X = 4000.0
Y = 6000.0
Z = 4000.0
dx = 60.0
dy = 20.0
dz = 20.0

wavelength = 1550.0


####################################################################################
# Setup simulations
####################################################################################
w_pml = dx * 15

sim1 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=5e-6, min_rindex=1.0)
sim1.src_ramp_time = sim1.Nlambda * 20
sim1.Nmax = sim1.Nlambda * 10000

sim1.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
sim1.bc = '000'

#
#sim2 = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=5e-6, min_rindex=1.0)
#sim2.src_ramp_time = sim2.Nlambda * 20
#sim2.Nmax = sim2.Nlambda * 10000
#
#sim2.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
#sim2.bc = '000'




X = sim1.X
Y = sim1.Y
Z = sim1.Z

####################################################################################
# Define the geometry/materials
####################################################################################
w_wg = 550.0
w_wg_out = 900.0

L_in = X/2
L_out = X/2

h_spacer = 40.0
h_waveguide = 180.0
h_lowerRidge = 500.0

h_taper = X-1000.0-2*w_pml

n_InP = 3.4
n_SiO2 = 1.444

# Ridge spacer
wg_in_top_spacer = emopt.grid.Rectangle(X/4, Y/2, L_in, w_wg)
wg_in_top_spacer.layer = 1; wg_in_top_spacer.material_value = n_SiO2**2

wg_out_spacer = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg_out)
wg_out_spacer.layer = 1; wg_out_spacer.material_value = n_SiO2**2



wg_in_top_waveguide = emopt.grid.Rectangle(X/4, Y/2, L_in, w_wg)
wg_in_top_waveguide.layer = 1; wg_in_top_waveguide.material_value = n_InP**2

wg_out_waveguide = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg_out)
wg_out_waveguide.layer = 1; wg_out_waveguide.material_value = n_InP**2



wg_in_top_lower = emopt.grid.Rectangle(X/4, Y/2, L_in, w_wg)
wg_in_top_lower.layer = 1; wg_in_top_lower.material_value = n_SiO2**2

wg_out_lower = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg_out)
wg_out_lower.layer = 1; wg_out_lower.material_value = n_SiO2**2


# taper
x_taper = np.array([X/2-h_taper/2, 0, 0, X/2-h_taper/2, X/2+h_taper/2, X, X,
                    X/2+h_taper/2])
#y_taper = np.array([Y/2-w_wg/2, Y/2-w_wg/2, Y/2+w_wg/2, Y/2+w_wg/2,
#                    Y/2+w_wg_out/2, Y/2+w_wg_out/2, Y/2-w_wg_out/2, Y/2-w_wg_out/2])

y_taper= np.array([Y/2-w_wg/2, Y/2-w_wg/2, Y/2+w_wg/2, Y/2+w_wg/2, Y/2+w_wg/2,
                   Y/2+w_wg/2, Y/2-w_wg_out+w_wg/2, Y/2-w_wg_out+w_wg/2 ])

taper_spacer = emopt.grid.Polygon()
taper_spacer.set_points(x_taper, y_taper)
taper_spacer.layer = 1; taper_spacer.material_value = n_SiO2**2

taper_waveguide = emopt.grid.Polygon()
taper_waveguide.set_points(x_taper, y_taper)
taper_waveguide.layer = 1; taper_waveguide.material_value = n_InP**2

taper_lower = emopt.grid.Polygon()
taper_lower.set_points(x_taper, y_taper)
taper_lower.layer = 1; taper_lower.material_value = n_SiO2**2



# background air
bg_air = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
bg_air.layer = 2; bg_air.material_value = 1.0

# background substrate
bg_substrate = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
bg_substrate.layer = 2; bg_substrate.material_value = n_SiO2**2



eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)

#eps.add_primitive(wg_in_top_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
#eps.add_primitive(wg_in_top_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
#eps.add_primitive(wg_in_top_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)
#
#eps.add_primitive(wg_out_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
#eps.add_primitive(wg_out_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
#eps.add_primitive(wg_out_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)

eps.add_primitive(taper_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
eps.add_primitive(taper_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
eps.add_primitive(taper_lower, Z/2-h_waveguide/2-h_lowerRidge, Z/2-h_waveguide/2)



eps.add_primitive(bg_substrate, 0, Z/2-h_waveguide/2-h_lowerRidge)
eps.add_primitive(bg_air, Z/2-h_waveguide/2-h_lowerRidge, Z)


mu = emopt.grid.ConstantMaterial3D(1.0)

sim1.set_materials(eps, mu)
sim1.build()

#sim2.set_materials(eps, mu)
#sim2.build()




###############################################################################
# Setup the sources
###############################################################################
input_slice = emopt.misc.DomainCoordinates(18*dx, 18*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)
slices= emopt.grid.DomainCoordinates(X/2, X/2, w_pml, Y-w_pml, w_pml, Z-w_pml,
                                    dx, dy, dz)
slices2 = emopt.grid.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml,
                                       Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)
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

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt
    f = plt.figure()

for i in range(4):
    Jx0, Jy0, Jz0, Mx0, My0, Mz0 = mode.get_source(i, dx, dy, dz)
    #Jx0, Jy0, Jz0, Mx0, My0, Mz0 = powerNormalizeMode(Jx0, Jy0, Jz0, Mx0, My0,
     #                                                Mz0, np.array([1, 0 ,0]),
      #                                                dy, dz)

    Jxs.append(Jx0)
    Jys.append(Jy0)
    Jzs.append(Jz0)
    Mxs.append(Mx0)
    Mys.append(My0)
    Mzs.append(Mz0)
    Eym = mode.get_field_interp(i, 'Ey')
    phases.append(np.angle(Eym[0,0]))
    if (NOT_PARALLEL):
        print mode.neff[i]
        import matplotlib.pyplot as plt
        Eym=Eym.squeeze()
        vmin = np.min(np.real(Eym))
        vmax = np.max(np.real(Eym.real))
        ax = f.add_subplot(4,1,i+1)
        ax.imshow(np.real(Eym), extent=[0, Y, 0, Z], vmin=vmin, vmax=-vmin, cmap='seismic')

if(NOT_PARALLEL):
    plt.show()

#phaseFundamental = findRelativePhase(Jxs[0], Jxs[1])
#phaseHigher = findRelativePhase(Jxs[2], Jxs[3])

phaseFundamental = phases[0] - phases[1]
phaseHigher = phases[2] - phases[3]

#JxFund = Jxs[0] + Jxs[1]*np.exp(1j * phaseFundamental)
#JyFund = Jys[0] + Jys[1]*np.exp(1j * phaseFundamental)
#JzFund = Jzs[0] + Jzs[1]*np.exp(1j * phaseFundamental)
#MxFund = Mxs[0] + Mxs[1]*np.exp(1j * phaseFundamental)
#MyFund = Mys[0] + Mys[1]*np.exp(1j * phaseFundamental)
#MzFund = Mzs[0] + Mzs[1]*np.exp(1j * phaseFundamental)
#
#JxHigher = Jxs[2] - Jxs[3]*np.exp(1j * phaseHigher)
#JyHigher = Jys[2] - Jys[3]*np.exp(1j * phaseHigher)
#JzHigher = Jzs[2] - Jzs[3]*np.exp(1j * phaseHigher)
#MxHigher = Mxs[2] - Mxs[3]*np.exp(1j * phaseHigher)
#MyHigher = Mys[2] - Mys[3]*np.exp(1j * phaseHigher)
#MzHigher = Mzs[2] - Mzs[3]*np.exp(1j * phaseHigher)

JxFund = Jxs[0]
JyFund = Jys[0]
JzFund = Jzs[0]
MxFund = Mxs[0]
MyFund = Mys[0]
MzFund = Mzs[0]

JxHigher = Jxs[2]
JyHigher = Jys[2]
JzHigher = Jzs[2]
MxHigher = Mxs[2]
MyHigher = Mys[2]
MzHigher = Mzs[2]



src1 = [JxFund, JyFund, JzFund, MxFund, MyFund, MzFund]
#src2 = [JxHigher, JyHigher, JzHigher, MxHigher, MyHigher, MzHigher]

src1 = COMM.bcast(src1, root=0)
#src2 = COMM.bcast(src2, root=0)


sim1.set_sources(src1, input_slice)
#sim2.set_sources(src2, input_slice)


##############################################################################
# View each simulation
##############################################################################

sim1.solve_forward()
#sim2.solve_forward()
field_monitor = emopt.misc.DomainCoordinates(0, X, 0, Y,
                                             Z/2, Z/2, dx, dy, dz)
Ey = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)


###############################################################################
# Mode match for optimization
###############################################################################
fom_slice = emopt.misc.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)

fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45, neigs=6)

fom_mode.bc = '00'
fom_mode.build()
fom_mode.solve()


print("Got Here")

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



Ex_1 = sim1.get_field_interp('Ex', domain=fom_slice)
Ey_1 = sim1.get_field_interp('Ey', domain=fom_slice)
Ez_1 = sim1.get_field_interp('Ez', domain=fom_slice)
Hx_1 = sim1.get_field_interp('Hx', domain=fom_slice)
Hy_1 = sim1.get_field_interp('Hy', domain=fom_slice)
Hz_1 = sim1.get_field_interp('Hz', domain=fom_slice)



#Ex_2 = sim2.get_field_interp('Ex', domain=fom_slice)
#Ey_2 = sim2.get_field_interp('Ey', domain=fom_slice)
#Ez_2 = sim2.get_field_interp('Ez', domain=fom_slice)
#Hx_2 = sim2.get_field_interp('Hx', domain=fom_slice)
#Hy_2 = sim2.get_field_interp('Hy', domain=fom_slice)
#Hz_2 = sim2.get_field_interp('Hz', domain=fom_slice)

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



if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    Eym0_bla=np.squeeze(Eym0)
    Ey_1_bla=np.squeeze(Ey_1)

    f5=plt.figure()
    ax = f5.add_subplot(211)
    vmax = np.max(np.abs(Eym0_bla))
    pos = ax.imshow(np.abs(Eym0_bla), extent=[0, Y, 0, Z], vmin=0, vmax=vmax,
              cmap='inferno')
    f5.colorbar(pos, ax=ax)

    ax = f5.add_subplot(212)
    vmax = np.max(np.abs(Ey_1_bla))
    bla = ax.imshow(np.abs(Ey_1_bla), extent=[0, Y, 0, Z], vmin=0, vmax=vmax,
              cmap='inferno')
    f5.colorbar(bla, ax=ax)
    plt.show()


    mode_match1 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm0, Eym0, Ezm0, Hxm0,
                                           Hym0, Hzm0)
    Psrc1 = sim1.source_power
    mode_match1.compute(Ex_1, Ey_1, Ez_1, Hx_1, Hy_1, Hz_1)
    value1 = mode_match1.get_mode_match_forward(1.0)
    value1 = value1/Psrc1

    #mode_match2 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm1, Eym1, Ezm1, Hxm1,
    #                                      Hym1, Hzm1)
    #Psrc2 = sim2.source_power
    #mode_match2.compute(Ex_2, Ey_2, Ez_2, Hx_2, Hy_2, Hz_2)
    #value2 = mode_match2.get_mode_match_forward(1.0)
    #value2 = value2/Psrc2
    print("mode match fundamental = ", value1)
    #print("mode match second order = ", value2)



Ey = sim1.get_field_interp('Ey', domain=field_monitor, squeeze=True)

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt
    eps_arr = eps.get_values_in(field_monitor, squeeze=True)

    vmax = np.max(np.real(Ey))
    f = plt.figure()
    ax1 = f.add_subplot(111)
    ax1.imshow(np.real(Ey), extent = [0,X,0,Y], vmin=-vmax,
              vmax=vmax, cmap='seismic')
    plt.show()
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

#
#

#
#
################################################################################
## Mode match for optimization
################################################################################
#fom_slice = emopt.misc.DomainCoordinates(X-w_pml-4*dx, X-w_pml-4*dx, w_pml, Y-w_pml, w_pml, Z-w_pml, dx, dy, dz)
#
#fom_mode = emopt.modes.ModeFullVector(wavelength, eps, mu, fom_slice, n0=3.45, neigs=6)
#
#fom_mode.bc = '00'
#fom_mode.build()
#fom_mode.solve()
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
#mode_match1 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm0, Eym0, Ezm0, Hxm0,
#                                       Hym0, Hzm0)
#mode_match2 = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm1, Eym1, Ezm1, Hxm1,
#                                      Hym1, Hzm1)
################################################################################
## Setup the AdjointMethod objected needed for gradient calculations
################################################################################
#params = np.array([w_combiner, L_combiner])
#
#sim1.field_domains = [fom_slice]
#        am1 = IncoherentCombinerAdjointMethod(sim1, combiner, fom_slice,
#                                              mode_match1)
#
#sim2.field_domains = [fom_slice]
#        am2 = IncoherentCombinerAdjointMethod(sim2, combiner, fom_slice,
#                                              mode_match2)
#
#ams = []
#ams.append(am1)
#ams.append(am2)
#
#am = MultiObjective(ams)
#
#am.check_gradient(params)
#
################################################################################
## Setup and run the simulation
################################################################################
#opt = emopt.optimizer.Optimizer(am, params, Nmax=10, opt_method='L-BFGS-B')
#fom, pfinal = opt.run()
#
#am.fom(pfinal)
#
#field_monitor = emopt.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, Z/2,
#                                             Z/2, dx, dy, dz)
#
#Ey = sim.get_field_interp('Ey', domain=field_monitor, squeeze=True)
#if(NOT_PARALLEL):
#    import matplotlib.pyplot as plt
#
#    Ey = np.concatenate([Ey[::-1],Ey], axis=0)
#
#    eps_arr = eps.get_values_in(field_monitor, squeeze=True)
#    vmax = np.max(np.real(Ey))
#    f=plt.figure()
#    ax1 = f.add_subplot(111)
#    ax1.imshow(np.real(Ey), extent=[0,X-2*wpml,0,Y-2*w_pml],vmin=-vmax, vmax=vmax, cmap='seismic')
#    plt.show()
