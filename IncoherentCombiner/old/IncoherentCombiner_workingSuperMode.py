import emopt
from emopt.misc import NOT_PARALLEL, run_on_master
from emopt.adjoint_method import AdjointMethod

import numpy as np
from math import pi

from petsc4py import PETSc
from mpi4py import MPI



####################################################################################
# Simulation parameters
####################################################################################
X = 5000.0
Y = 5000.0
Z = 2500.0
dx = 20.0
dy = 20.0
dz = 20.0

wavelength = 1550.0

####################################################################################
# Setup simulation
####################################################################################
sim = emopt.fdtd.FDTD(X, Y, Z, dx, dy, dz, wavelength, rtol=1e-6, min_rindex=1.0)
w_pml = dx * 15
sim.src_ramp_time = sim.Nlambda * 20
sim.Nmax = sim.Nlambda * 250

sim.w_pml = [w_pml, w_pml, w_pml, w_pml, w_pml, w_pml]
sim.bc = '000'

X = sim.X
Y = sim.Y
Z = sim.Z

####################################################################################
# Define the geometry/materials
####################################################################################
w_wg = 800.0
L_in = X/2
L_out = X/2
L_combiner = 2400.0
w_combiner = 2800.0
h_spacer = 40.0
h_waveguide = 180.0
h_lowerRidge = 500.0
h_substrate = 660.0 
wg_spacing = 600.0



n_InP = 3.4
n_SiO2 = 1.444

# Ridge spacer
wg_in_top_spacer = emopt.grid.Rectangle(X/4, Y/2+wg_spacing/2+w_wg/2, L_in, w_wg)
wg_in_top_spacer.layer = 1; wg_in_top_spacer.material_value = n_SiO2**2

wg_in_bottom_spacer = emopt.grid.Rectangle(X/4, Y/2-wg_spacing/2-w_wg/2, L_in, w_wg)
wg_in_bottom_spacer.layer = 1; wg_in_bottom_spacer.material_value = n_SiO2**2

wg_out_spacer = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg)
wg_out_spacer.layer = 1; wg_out_spacer.material_value = n_SiO2**2

combiner_spacer = emopt.grid.Rectangle(X/2, Y/2, L_combiner, w_combiner)
combiner_spacer.layer = 1; combiner_spacer.material_value = n_SiO2**2

# Ridge waveguide
wg_in_top_waveguide = emopt.grid.Rectangle(X/4, Y/2+wg_spacing/2+w_wg/2, L_in, w_wg)
wg_in_top_waveguide.layer = 1; wg_in_top_waveguide.material_value = n_InP**2

wg_in_bottom_waveguide = emopt.grid.Rectangle(X/4, Y/2-wg_spacing/2-w_wg/2, L_in, w_wg)
wg_in_bottom_waveguide.layer = 1; wg_in_bottom_waveguide.material_value = n_InP**2

wg_out_waveguide = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg)
wg_out_waveguide.layer = 1; wg_out_waveguide.material_value = n_InP**2

combiner_waveguide = emopt.grid.Rectangle(X/2, Y/2, L_combiner, w_combiner)
combiner_waveguide.layer = 1; combiner_waveguide.material_value = n_InP**2

# Ridge lower
wg_in_top_lower = emopt.grid.Rectangle(X/4, Y/2+wg_spacing/2+w_wg/2, L_in, w_wg)
wg_in_top_lower.layer = 1; wg_in_top_lower.material_value = n_SiO2**2

wg_in_bottom_lower = emopt.grid.Rectangle(X/4, Y/2-wg_spacing/2-w_wg/2, L_in, w_wg)
wg_in_bottom_lower.layer = 1; wg_in_bottom_lower.material_value = n_SiO2**2

wg_out_lower = emopt.grid.Rectangle(3*X/4, Y/2, L_out, w_wg)
wg_out_lower.layer = 1; wg_out_lower.material_value = n_SiO2**2

combiner_lower = emopt.grid.Rectangle(X/2, Y/2, L_combiner, w_combiner)
combiner_lower.layer = 1; combiner_lower.material_value = n_SiO2**2

# background air
bg_air = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
bg_air.layer = 2; bg_air.material_value = 1.0

# background substrate
bg_substrate = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
bg_substrate.layer = 2; bg_substrate.material_value = n_SiO2**2



eps = emopt.grid.StructuredMaterial3D(X, Y, Z, dx, dy, dz)

eps.add_primitive(wg_in_top_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
eps.add_primitive(wg_in_top_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
eps.add_primitive(wg_in_top_lower, h_substrate, Z/2-h_waveguide/2)

eps.add_primitive(wg_in_bottom_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
eps.add_primitive(wg_in_bottom_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
eps.add_primitive(wg_in_bottom_lower, h_substrate, Z/2-h_waveguide/2)

eps.add_primitive(wg_out_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
eps.add_primitive(wg_out_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
eps.add_primitive(wg_out_lower, h_substrate, Z/2-h_waveguide/2)

eps.add_primitive(combiner_spacer, Z/2+h_waveguide/2, Z/2+h_waveguide/2+h_spacer)
eps.add_primitive(combiner_waveguide, Z/2-h_waveguide/2, Z/2+h_waveguide/2)
eps.add_primitive(combiner_lower, h_substrate, Z/2-h_waveguide/2)


eps.add_primitive(bg_substrate, 0, h_substrate)
eps.add_primitive(bg_air, h_substrate, Z)


mu = emopt.grid.ConstantMaterial3D(1.0)

sim.set_materials(eps, mu)
sim.build()

###############################################################################
# Setup the sources
###############################################################################
input_slice = emopt.misc.DomainCoordinates(w_pml+4*dx, w_pml+4*dx, w_pml+dy, Y-w_pml-dy, w_pml+dz, Z-w_pml-dz, dx, dy, dz)

mode = emopt.modes.ModeFullVector(wavelength, eps, mu, input_slice, n0=3.4, neigs=8)

mode.bc = '00'
mode.build()
mode.solve()

sim.set_sources(mode, input_slice)

Eym_inputer = []
normal = np.array([1, 0, 0])
xhat = np.array([1, 0, 0])
yhat = np.array([0, 1, 0])
zhat = np.array([0, 0, 1])

x_dot_s = xhat.dot(normal)
y_dot_s = yhat.dot(normal)
z_dot_s = zhat.dot(normal)
phase = []

for i in range(8):
    Exm = mode.get_field_interp(i, 'Ex')
    Eym = mode.get_field_interp(i, 'Ey')
    Ezm = mode.get_field_interp(i, 'Ez')
    Hxm = mode.get_field_interp(i, 'Hx')
    Hym = mode.get_field_interp(i, 'Hy')
    Hzm = mode.get_field_interp(i, 'Hz')

    Pxm = Eym * np.conj(Hzm) - Ezm * np.conj(Hym)
    Pym = -Exm * np.conj(Hzm) + Ezm * np.conj(Hxm)
    Pzm = Exm * np.conj(Hym) - Eym * np.conj(Hxm)

    Pm = dx*dy*np.sum(x_dot_s * Pxm + y_dot_s * Pym + z_dot_s * Pzm)
    Eym_input00 = Eym / Pm

    phase.append(np.arctan2(np.imag(Eym_input00[0,0]),np.real(Eym_input00[0,0])))

    Eym_inputer.append(Eym_input00)
    

Eym_input=[]

Eym_input.append(Eym_inputer[0])
Eym_input.append(Eym_inputer[1])
Eym_input.append(Eym_inputer[2])
Eym_input.append(Eym_inputer[3])
Eym_input.append(Eym_inputer[0]+Eym_inputer[1] * np.exp(1j * phase[0] - 1j *
                                                     phase[1]) )
Eym_input.append(Eym_inputer[2]-Eym_inputer[3] * np.exp(1j * phase[2] - 1j *
                                                     phase[3]))


if(NOT_PARALLEL):
    import matplotlib.pyplot as plt
    f = plt.figure()
    eps_arr = eps.get_values_in(input_slice)
    eps_arr = eps_arr.real
    eps_arr = np.squeeze(eps_arr)

    for i in range(6):
        vmin = -np.max(np.real(Eym_input[i]))
        vmax = np.max(np.real(Eym_input[i]))

        ax = f.add_subplot(6,1,i+1)
        im = ax.imshow(np.real(Eym_input[i]), extent=[0, Y, 0, Z], vmin=vmin,
                       vmax=vmax, cmap='seismic', origin='lower')
        ax.contour(eps_arr, levels=[1.0, n_SiO2**2, n_InP**2], extent=[0, Y, 0,
                                                                      Z],
                   linewidths=[1,], colors='w', alpha=0.5)
        f.colorbar(im, ax=ax)

    plt.show()

#
#
#
#
#
#
#
#
#
#
#if(NOT_PARALLEL):
#    Nz, Ny = Exm.shape
#    Exm = np.reshape(Exm, (Nz, Ny, 1))
#    Eym = np.reshape(Eym, (Nz, Ny, 1))
#    Ezm = np.reshape(Ezm, (Nz, Ny, 1))
#    Hxm = np.reshape(Hxm, (Nz, Ny, 1))
#    Hym = np.reshape(Hym, (Nz, Ny, 1))
#    Hzm = np.reshape(Hzm, (Nz, Ny, 1))
#
#
#mode_match = emopt.fomutils.ModeMatch([1,0,0], dy, dz, Exm, Eym, Ezm, Hxm, Hym, Hzm)
#
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
