"""Demonstrate how to set up a simple simulation in emopt consisting of a
waveguide which is excited by a dipole current located at the center of the
waveguide.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python simple_waveguide.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""

#import emopt
import emopt2 as emopt
from emopt.misc import NOT_PARALLEL

import numpy as np
#import matplotlib
#matplotlib.use('Agg')

#Ez = []
#Hx = []
#Hy = []
Hz = []
Ex = []
Ey = []

####################################################################################
#Simulation Region parameters
####################################################################################
X = 10.0
Y = 5.0
dx = 0.01
dy = 0.01
wavelength = 1.55

sim = emopt.fdtd_2d.FDTD_TE(X,Y,dx,dy,wavelength, rtol=1e-5, min_rindex=1.44,
                      nconv=100, complex_eps=True)

sim.Nmax = 10*sim.Ncycle
sim.courant_num = 0.9
w_pml = dx * 100 # set the PML width

# we use symmetry boundary conditions at y=0 to speed things up. We
# need to make sure to set the PML width at the minimum y boundary is set to
# zero. Currently, FDTD cannot compute accurate gradients using symmetry in z
# :(
#sim.w_pml = [w_pml, w_pml, 0, w_pml]
sim.w_pml = [w_pml, w_pml, w_pml, w_pml]
sim.bc = '00'

# get actual simulation dimensions
X = sim.X
Y = sim.Y


####################################################################################
# Setup system materials
####################################################################################
# Materials
n0 = 1.44
n1 = -1000.0+10.0j

# set a background permittivity of 1
eps_background = emopt.grid.Rectangle(X/2, Y/2, 2*X, 2*Y)
eps_background.layer = 2
eps_background.material_value = n0**2

# Create a high index waveguide through the center of the simulation
h_wg = 0.5
#waveguide = emopt.grid.Rectangle(X/2, Y/2, 2*X, h_wg)
waveguide = emopt.grid.Rectangle(X/2+2, Y/2, 1.0, 1.0)
waveguide.layer = 1
waveguide.material_value = n1

waveguide2 = emopt.grid.Rectangle(X/2-2, Y/2, 1.0, 1.0)
waveguide2.layer = 1
waveguide2.material_value = n1

eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
eps.add_primitive(waveguide)
eps.add_primitive(waveguide2)
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
src_domain = emopt.misc.DomainCoordinates(X/2, X/2, Y/2, Y/2, 0, 0, dx, dy, 1.0)
#Jz = np.zeros([M,N], dtype=np.complex128)
#Mx = np.zeros([M,N], dtype=np.complex128)
#My = np.zeros([M,N], dtype=np.complex128)

M = src_domain.Nx
N = src_domain.Ny
print((M,N))

Jz = np.zeros([N,M], dtype=np.complex128)
Mx = np.zeros([N,M], dtype=np.complex128)
My = np.zeros([N,M], dtype=np.complex128)

Jz[0,0] = 1.0

src = [Jz, Mx, My]
sim.set_sources(src, src_domain)

####################################################################################
# Build and simulate
####################################################################################
sim_area = emopt.misc.DomainCoordinates(0, X, 0, Y, 0, 0, dx, dy, 1.0)


#if(NOT_PARALLEL):
#    print('hello')
#    eps = sim.eps.get_values_in(sim_area)
#    import matplotlib.pyplot as plt
#    extent = sim_area.get_bounding_box()[0:4]
#    f = plt.figure()
#    ax = f.add_subplot(111)
#    im = ax.imshow(eps.imag, extent=extent,
#                            #vmin=-np.max(HH)/1.0,
#                            #vmin=0.0,
#                            #vmax=0.0001,
#                            #cmap='jet',interpolation='nearest')
#                            cmap='seismic')
#    f.colorbar(im)
#    plt.savefig('lossy_material_eps.pdf')
#    plt.show()

sim.build()
sim.solve_forward()

# Get the fields we just solved for
# define a plane using a DomainCoordinates with no z-thickness
#Ez.append(sim.get_field_interp('Ez', sim_area))
#Hy.append(sim.get_field_interp('Hy', sim_area))
#Hx.append(sim.get_field_interp('Hx', sim_area))
Hz.append(sim.get_field_interp('Ez', sim_area))
#Ey.append(sim.get_field_interp('Ey', sim_area))
#Ex.append(sim.get_field_interp('Ex', sim_area))
#E2 = (Ez*Ez.conj()).real

# Simulate the field.  Since we are running this using MPI, we only generate
# plots in the master process (otherwise we would end up with a bunch of
# plots...). This is accomplished using the NOT_PARALLEL flag which is defined
# by emopt.misc



HH = np.sqrt((Hz[0]*Hz[0].conj()).real)
#EE = np.sqrt((Ez[0]*Ez[0].conj()).real)
#print(np.allclose(Ez[0], Ez[1]))
#print(np.allclose(Hx[0], Hx[1]))
#print(np.allclose(Hy[0], Hy[1]))
#print(np.linalg.norm(Ez[0]-Ez[1]))
#print(np.linalg.norm(Hx[0]-Hx[1]))
#print(np.linalg.norm(Hy[0]-Hy[1]))

if(NOT_PARALLEL):
    import matplotlib.pyplot as plt

    extent = sim_area.get_bounding_box()[0:4]

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(Hz[0].real, extent=extent,
                            #vmin=-np.max(HH)/1.0,
                            #vmin=0.0,
                            #vmax=0.0001,
                            vmin=-np.max(Hz[0].real)/1.0,
                            vmax=np.max(Hz[0].real)/1.0,
                            #cmap='jet',interpolation='nearest')
                            cmap='seismic')

    # Plot the waveguide boundaries
    #ax.plot(extent[0:2], [Y/2-h_wg/2, Y/2-h_wg/2], 'k-')
    #ax.plot(extent[0:2], [Y/2+h_wg/2, Y/2+h_wg/2], 'k-')

    ax.set_title('E$_z$', fontsize=18)
    ax.set_xlabel('x [um]', fontsize=14)
    ax.set_ylabel('y [um]', fontsize=14)
    f.colorbar(im)
    plt.savefig('lossy_material_fdtd_2d_cyl.pdf')
