"""Demonstrate how to set up a simple simulation in emopt consisting of a
waveguide which is excited by a dipole current located at the center of the
waveguide.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python simple_waveguide.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""

import emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
import matplotlib
matplotlib.use('Agg')

####################################################################################
#Simulation Region parameters
####################################################################################
X = 15.0
Y = 10.0
dx = 0.04
dy = 0.04
wavelength = 1.55

sim = emopt.fdtd_2d_cyl.FDTD_TE(X,Y,dx,dy,wavelength, rtol=1e-5, min_rindex=1.44,
                      nconv=100)
sim.Nmax = 1000*sim.Ncycle
sim.courant_num = 0.9
w_pml = dx * 50 # set the PML width

# we use symmetry boundary conditions at y=0 to speed things up. We
# need to make sure to set the PML width at the minimum y boundary is set to
# zero. Currently, FDTD cannot compute accurate gradients using symmetry in z
# :(
#sim.w_pml = [w_pml, w_pml, 0, w_pml]
sim.w_pml = [w_pml, w_pml, w_pml, 0]
sim.bc = 'HH'

# get actual simulation dimensions
X = sim.X
Y = sim.Y


####################################################################################
# Setup system materials
####################################################################################
# Materials
n0 = 1.44
n1 = 3.0

# set a background permittivity of 1
eps_background = emopt.grid.Rectangle(X/2, 0, 2*X, 2*Y)
eps_background.layer = 2
eps_background.material_value = n0**2

# Create a high index waveguide through the center of the simulation
h_wg = 0.5
#waveguide = emopt.grid.Rectangle(X/2, Y/2, 2*X, h_wg)
#waveguide = emopt.grid.Rectangle(X/2, 0.0, 1.0, 1.0)
#waveguide.layer = 1
#waveguide.material_value = n1**2

eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
#eps.add_primitive(waveguide)
eps.add_primitive(eps_background)

mu = emopt.grid.ConstantMaterial2D(1.0)

# set the materials used for simulation
sim.set_materials(eps, mu)

####################################################################################
# setup the sources
####################################################################################
# setup the sources -- just a dipole in the center of the waveguide
#src_domain = emopt.misc.DomainCoordinates(X/2, X/2, Y/2, Y/2, 0, 0, dx, dy, 1.0)

#src_domain = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, 0, dx, dy, 1.0)
#src_domain = emopt.misc.DomainCoordinates(X/2, X/2, Y/2, Y/2, 0, 0, dx, dy, 1.0)
#src_domain = emopt.misc.DomainCoordinates(X/2, X/2, 0, 0, 0, 0, dx, dy, 1.0)
src_domain = emopt.misc.DomainCoordinates(X/2, X/2, Y, Y, 0, 0, dx, dy, 1.0)
#Jz = np.zeros([M,N], dtype=np.complex128)
#Mx = np.zeros([M,N], dtype=np.complex128)
#My = np.zeros([M,N], dtype=np.complex128)

M = src_domain.Nx
N = src_domain.Ny
print((M,N))

Jz = np.zeros([N,M], dtype=np.complex128)
Mx = np.zeros([N,M], dtype=np.complex128)
My = np.zeros([N,M], dtype=np.complex128)

Mx[0,0] = 1.0

src = [Jz, Mx, My]
sim.set_sources(src, src_domain)

####################################################################################
# Build and simulate
####################################################################################
sim.build()
sim.solve_forward()

# Get the fields we just solved for
# define a plane using a DomainCoordinates with no z-thickness
sim_area = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, 0, dx, dy, 1.0)
#Ez = sim.get_field_interp('Ez', sim_area)
Hy = sim.get_field_interp('Hy', sim_area)
#E2 = (Ez*Ez.conj()).real

# Simulate the field.  Since we are running this using MPI, we only generate
# plots in the master process (otherwise we would end up with a bunch of
# plots...). This is accomplished using the NOT_PARALLEL flag which is defined
# by emopt.misc
if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    extent = sim_area.get_bounding_box()[0:4]

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(Hy.real, extent=extent,
                            vmin=-np.max(Hy.real)/1.0,
                            vmax=np.max(Hy.real)/1.0,
                            #cmap='jet',interpolation='nearest')
                            cmap='jet')

    # Plot the waveguide boundaries
    #ax.plot(extent[0:2], [Y/2-h_wg/2, Y/2-h_wg/2], 'k-')
    #ax.plot(extent[0:2], [Y/2+h_wg/2, Y/2+h_wg/2], 'k-')

    ax.set_title('E$_z$', fontsize=18)
    ax.set_xlabel('x [um]', fontsize=14)
    ax.set_ylabel('y [um]', fontsize=14)
    f.colorbar(im)
    plt.savefig('simple_waveguide_fdtd_cyl.pdf')
