"""Demonstrate how to set up a simple simulation in emopt consisting of a
waveguide which is excited by a dipole current located at the center of the
waveguide.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python simple_waveguide.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""

import emopt
from emopt.adjoint_method import AdjointMethodPNF2D
from emopt.misc import NOT_PARALLEL, run_on_master

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Test(AdjointMethodPNF2D):
    def __init__(self, sim, rectangle):
        super(Test, self).__init__(sim, step=1e-8)
        self.rectangle = rectangle
        self.current_fom = 0

    def update_system(self, params):
        w = params[0]
        h = params[1]
        x0 = params[2]
        y0 = params[3]
        self.rectangle.width = w
        self.rectangle.height = h
        self.rectangle.x0 = x0
        self.rectangle.y0 = y0

    #def calc_fom(self, sim, params):
    @run_on_master
    def calc_f(self, sim, params):
        Ez, Hx, Hy = sim.saved_fields[0]

        self.current_fom = -1*(Ez*Ez.conj()).real.squeeze()
        return self.current_fom

    #def calc_dFdx(self, sim, params):
    @run_on_master
    def calc_dfdx(self, sim, params):
        Ez, Hx, Hy = sim.saved_fields[0]
        adjsrc_plane = sim.field_domains[0]

        dfdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dfdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        dfdEz[adjsrc_plane.j, adjsrc_plane.k] = -1*np.conj(Ez)

        #dFdEz, dFdHx, dFdHy = emopt.fomutils.interpolated_dFdx_2D(sim, dfdEz,
        #                                                          dfdHx, dfdHy)

        #return (dFdEz, dFdHx, dFdHy)
        return (dfdEz, dfdHx, dfdHy)

    def calc_grad_p(self, sim, params):
        return np.zeros(params.shape)

def callback(params, sim, am, full_field, fom_history):
    #fom = am.current_fom
    fom = am.calc_fom(sim, params)
    fom_history.append(fom)

    Ez, Hx, Hy = sim.saved_fields[1]
    E2 = (Ez*Ez.conj()).real

    extent = full_field.get_bounding_box()[0:4]

    f = plt.figure()
    ax = f.add_subplot(111)
    #im = ax.imshow(Ez.real, extent=extent,
    #                        #vmin=0.0,
    #                        #vmax=E2.max(),
    #                        vmin=-1*np.max(Ez.real),
    #                        vmax=1*np.max(Ez.real),
    #                        cmap='seismic', origin='lower')
    im = ax.imshow(E2, extent=extent,
                            vmin=0.0,
                            vmax=E2.max()/50.0,
                            cmap='seismic', origin='lower')
    print('psrc', sim.get_source_power())
    print('fom',fom)
    plt.savefig('current_iter.pdf')
    


if __name__ == '__main__':
    ####################################################################################
    #Simulation Region parameters
    ####################################################################################
    X = 10.0
    Y = 7.0
    dx = 0.01
    dy = 0.01
    wlen = 1.55
    sim = emopt.fdfd.FDFD_TE(X, Y, dx, dy, wlen)
    
    # by default, PML size is chosen for you. If you want to specify your own PML
    # sizes you can set them using the sim.w_pml attribute which is an array with 4
    # values [w_xmin, w_xmax, w_ymin, w_ymax] where each value is a PML width for
    # the corresponding simulation boundary. e.g.
    # sim.w_pml = [dx*12, dx*12, dx*12, dx*12]
    
    # Get the actual width and height
    # The true width/height will not necessarily match what we used when
    # initializing the solver. This is the case when the width is not an integer
    # multiple of the grid spacing used.
    X = sim.X
    Y = sim.Y
    M = sim.M
    N = sim.N
    w_pml = sim.w_pml
    
    
    ####################################################################################
    # Setup system materials
    ####################################################################################
    # Materials
    n0 = 1.0
    #$n1 = 3.0
    eps_au = -130.0 + 3.3j
    
    # set a background permittivity of 1
    eps_background = emopt.grid.Rectangle(X/2, Y/2, 2*X, Y)
    eps_background.layer = 2
    eps_background.material_value = n0**2
    
    # Create a high index waveguide through the center of the simulation
    h_wg = 1.0
    waveguide = emopt.grid.Rectangle(X/2, Y/2+Y/4, h_wg, h_wg)
    waveguide.layer = 1
    waveguide.material_value = eps_au
    
    eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
    eps.add_primitive(waveguide)
    eps.add_primitive(eps_background)
    
    mu = emopt.grid.ConstantMaterial2D(1.0)
    
    # set the materials used for simulation
    sim.set_materials(eps, mu)
    
    ####################################################################################
    # setup the sources
    ####################################################################################
    # setup the sources -- just a dipole in the center of the waveguide
    Jz = np.zeros([1,1], dtype=np.complex128)
    Mx = np.zeros([1,1], dtype=np.complex128)
    My = np.zeros([1,1], dtype=np.complex128)
    #Jz[M//2, N//2] = 1.0
    Jz[0,0] = 1.0
    src_plane = emopt.misc.DomainCoordinates(X/4, X/4,
                                             Y/4, Y/4,
                                             0, 0, dx, dy, 1.0)
    
    sim.set_sources((Jz, Mx, My), src_plane)
    
    ####################################################################################
    # Build and simulate
    ####################################################################################
    fom_plane = emopt.misc.DomainCoordinates(3*X/4, 3*X/4,
                                             Y/4, Y/4,
                                             0, 0, dx, dy, 1.0)

    full_field = emopt.misc.DomainCoordinates(w_pml[0], X-w_pml[1],
                                              w_pml[2], Y-w_pml[3],
                                              0, 0, dx, dy, 1.0)
    sim.build()
    sim.field_domains = [fom_plane, full_field]
    

    design_params = np.array([3.0, 1.0, X/2, 3*Y/4])
    am = Test(sim, waveguide)

    am.check_gradient(design_params)

    fom_history = []
    callback_func = lambda p : callback(p, sim, am, full_field, fom_history)
    bounds = [(0.0, 10.0), (0.0, 5.0), (X/4, 3*X/4), (Y/4, 3*X/4)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='L-BFGS-B',
                                    Nmax=50, bounds=bounds,
                                    callback_func=callback_func)
    fom, params = opt.run()
    print(params)
