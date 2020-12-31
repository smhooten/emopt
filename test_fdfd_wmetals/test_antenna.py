"""Demonstrate how to set up a simple simulation in emopt consisting of a
waveguide which is excited by a dipole current located at the center of the
waveguide.

On most *nix-based machines, run the script with:

    $ mpirun -n 8 python simple_waveguide.py

If you wish to increase the number of cores that the example is executed on,
change 8 to the desired number of cores.
"""

import emopt
#from emopt.adjoint_method import AdjointMethodPNF2D
from emopt.adjoint_method import AdjointMethod
from emopt.misc import NOT_PARALLEL, run_on_master
from emopt.fomutils import interpolated_dFdx_2D

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


#class Test(AdjointMethodPNF2D):
class Test(AdjointMethod):
    def __init__(self, sim, left, right, d):
        super(Test, self).__init__(sim, step=1e-8)
        self.left = left
        self.right = right
        self.d = d
        self.current_fom = 0
        self.center = [sim.X / 2, sim.Y / 2]
        self.npts = 50

    def update_system(self, params):
        #length = params[0]
        #width = params[1]
        #self.right.width = length/2
        #self.right.height = width
        ##self.right.x0 = self.center + self.d/2 + length/4
        #self.right.x0 = self.center[0] + self.d/2 + length/4

        #self.left.width = length/2
        #self.left.height = width
        ##self.left.x0 = self.center - (self.d/2 + length/4)
        #self.left.x0 = self.center[0] - (self.d/2 + length/4)

        #r = params[0]
        #bla = params[1]
        #print(r)
        #self.right.set_position(self.center[0] + self.d/2 + r, self.center[1]+bla)
        #self.right.set_radius(r)

        #self.left.set_position(self.center[0] - (self.d/2 + r), self.center[1]+bla)
        #self.left.set_radius(r)
        r = params[0]
        lxpts, lypts = circle_func(self.center[0]-self.d/2-r, self.center[1], r, self.npts)
        self.left.set_points(lxpts, lypts)
        rxpts, rypts = circle_func(self.center[0]+self.d/2+r, self.center[1], r, self.npts)
        self.right.set_points(rxpts, rypts)

    #@run_on_master
    #def calc_f(self, sim, params):
    @run_on_master
    def calc_fom(self, sim, params):
        #Ezl, Hxl, Hyl = sim.saved_fields[0]
        #Ezr, Hxr, Hyr = sim.saved_fields[1]
        #Ezb, Hxb, Hyb = sim.saved_fields[2]
        #Ezt, Hxt, Hyt = sim.saved_fields[3]

        #Pxl =    sim.dx*0.5*np.sum((Ezl*Hyl.conj()).real)
        #Pxr = -1*sim.dx*0.5*np.sum((Ezr*Hyr.conj()).real)
        #Pyb = -1*sim.dy*0.5*np.sum((Ezb*Hxb.conj()).real)
        #Pyt =    sim.dy*0.5*np.sum((Ezt*Hxt.conj()).real)

        ##self.current_fom = -1*(Ez*Ez.conj()).real.squeeze()
        #self.current_fom = -1*(Pxl + Pxr + Pyb + Pyt)*1e8

        self.current_fom = -1*sim.get_source_power()
        return self.current_fom

    #@run_on_master
    #def calc_dfdx(self, sim, params):
    @run_on_master
    def calc_dFdx(self, sim, params):
        #Ezl, Hxl, Hyl = sim.saved_fields[0]
        #Ezr, Hxr, Hyr = sim.saved_fields[1]
        #Ezb, Hxb, Hyb = sim.saved_fields[2]
        #Ezt, Hxt, Hyt = sim.saved_fields[3]

        #plane_l = sim.field_domains[0]
        #plane_r = sim.field_domains[1]
        #plane_b = sim.field_domains[2]
        #plane_t = sim.field_domains[3]

        #dfdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        #dfdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        #dfdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        #dfdEz[plane_l.j, plane_l.k] =    sim.dx*0.25*Hyl.conj()
        #dfdEz[plane_r.j, plane_r.k] = -1*sim.dx*0.25*Hyr.conj()
        #dfdEz[plane_b.j, plane_b.k] = -1*sim.dy*0.25*Hxb.conj()
        #dfdEz[plane_t.j, plane_t.k] =    sim.dy*0.25*Hxt.conj()

        #dfdHx[plane_b.j, plane_b.k] = -1*sim.dy*0.25*Ezb.conj()
        #dfdHx[plane_t.j, plane_t.k] =    sim.dy*0.25*Ezt.conj()

        #dfdHy[plane_l.j, plane_l.k] =    sim.dx*0.25*Ezl.conj()
        #dfdHy[plane_r.j, plane_r.k] = -1*sim.dx*0.25*Ezr.conj()

        #dfdEz *= 1e8 
        #dfdHx *= 1e8 
        #dfdHy *= 1e8 
        Hzc = np.conj(sim.get_field_interp('Hz'))
        Exc = np.conj(sim.get_field_interp('Ex'))
        Eyc = np.conj(sim.get_field_interp('Ey'))
        dx = sim.dx
        dy = sim.dy
        M = sim.M
        N = sim.N

        if(not sim.real_materials):
            eps = sim.eps.get_values(0,N,0,M)
            mu = sim.mu.get_values(0,N,0,M)
        else:
            eps = np.zeros(Hzc.shape, dtype=np.complex128)
            mu = np.zeros(Hzc.shape, dtype=np.complex128)

        # get the planes through which power leaves the system
        w_pml_l = sim._w_pml_left + 2
        w_pml_r = sim._w_pml_right + 2
        w_pml_t = sim._w_pml_top + 2
        w_pml_b = sim._w_pml_bottom + 2

        x_bot = np.arange(w_pml_l, N-w_pml_r)
        y_bot = w_pml_b
        x_top = np.arange(w_pml_l, N-w_pml_r)
        y_top = M-w_pml_t

        x_left = w_pml_l
        y_left = np.arange(w_pml_b, M-w_pml_t)
        x_right = N-w_pml_r
        y_right = np.arange(w_pml_b, M-w_pml_t)

        x_all = np.arange(w_pml_l, N-w_pml_r)
        y_all = np.arange(w_pml_b, M-w_pml_t)
        y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

        dPdHz = np.zeros([M, N], dtype=np.complex128)
        dPdEx = np.zeros([M, N], dtype=np.complex128)
        dPdEy = np.zeros([M, N], dtype=np.complex128)

        dPdHz[y_left, x_left]   += -0.25*dy*Eyc[y_left, x_left]
        dPdHz[y_top, x_top]     += -0.25*dx*Exc[y_top, x_top]
        dPdHz[y_right, x_right] += 0.25*dy*Eyc[y_right, x_right]
        dPdHz[y_bot, x_bot]     += 0.25*dx*Exc[y_bot, x_bot]
        dPdHz[y_all, x_all]     += 0.25*dx*dy*mu[y_all,x_all].imag*Hzc[y_all, x_all]

        dPdEx[y_top, x_top] += -0.25*dx*Hzc[y_top, x_top]
        dPdEx[y_bot, x_bot] += +0.25*dx*Hzc[y_bot, x_bot]
        dPdEx[y_all, x_all] += 0.25*dx*dy*eps[y_all,x_all].imag*Exc[y_all, x_all]

        dPdEy[y_left, x_left]   += -0.25*dy*Hzc[y_left, x_left]
        dPdEy[y_right, x_right] += 0.25*dy*Hzc[y_right, x_right]
        dPdEy[y_all, x_all] += 0.25*dx*dy*eps[y_all,x_all].imag*Eyc[y_all, x_all]


        dFdHz, dFdEx, dFdEy = interpolated_dFdx_2D(sim, dPdHz, dPdEx, dPdEy)

        return (-1*dFdHz, -1*dFdEx, -1*dFdEy)

    def calc_grad_p(self, sim, params):
        return np.zeros(params.shape)

def callback(params, sim, am, full_field, fom_history):
    #fom = am.current_fom
    fom = am.calc_fom(sim, params)
    fom_history.append(fom)

    Ez, Hx, Hy = sim.saved_fields[-1]
    #E2 = (Ez*Ez.conj()).real
    E2 = Hx.real

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
                            #vmin=0.0,
                            #vmin=-E2.max()/50.0,
                            #vmax=E2.max()/50.0,
                            vmin=-E2.max(),
                            vmax=E2.max(),
                            cmap='seismic', origin='lower')
    print('psrc', sim.get_source_power())
    print('fom',fom)
    plt.savefig('current_iter.pdf')
    

def circle_func(x0, y0, r, npts):
    angles = np.linspace(0.0, 2*np.pi, num=npts)
    #xs = np.linspace(-r+x0, r+x0, num=npts)
    #ysp = np.sqrt(r**2-(xs-x0)**2) + y0
    #ysm = -np.sqrt(r**2-(xs-x0)**2) + y0

    #xs = np.concatenate((xs, np.flip(xs)))
    #ys = np.concatenate((ysp, np.flip(ysm)))
    xs = r*np.cos(angles) + x0
    ys = r*np.sin(angles) + y0

    return xs, ys


if __name__ == '__main__':
    ####################################################################################
    #Simulation Region parameters
    ####################################################################################
    X = 3.0
    Y = 3.0
    dx = 0.02
    dy = 0.02
    wlen = 1.55
    sim = emopt.fdfd.FDFD_TM(X, Y, dx, dy, wlen)
    
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
    #eps_au = 3.0**2
    
    # set a background permittivity of 1
    eps_background = emopt.grid.Rectangle(X/2, Y/2, 2*X, Y)
    eps_background.layer = 2
    eps_background.material_value = n0**2
    
    l = 0.75
    d = 0.1
    r = 0.1
    # Create a high index waveguide through the center of the simulation
    #left = emopt.grid.Rectangle(X/2+d/2+l/4, Y/2, l/2-d, 2*r)
    #left = emopt.grid.Circle(X/2-d/2-r, Y/2, r)
    #left.material_value = eps_au
    #left.set_material(eps_au)
    lxpts, lypts = circle_func(X/2-d/2-r, Y/2, r, 50)
    left = emopt.grid.Polygon(xs=lxpts, ys=lypts)
    left.material_value = eps_au
    left.layer = 1

    #right = emopt.grid.Rectangle(X/2-d/2-l/4, Y/2, l/2-d, 2*r)
    #right = emopt.grid.Circle(X/2+d/2+r, Y/2, r)
    #right.material_value = eps_au
    #right.set_material(eps_au)
    #right.layer = 1
    rxpts, rypts = circle_func(X/2+d/2+r, Y/2, r, 50)
    right = emopt.grid.Polygon(xs=rxpts, ys=rypts)
    right.material_value = eps_au
    right.layer = 1

    if NOT_PARALLEL:
        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(lxpts, lypts)
        ax.plot(rxpts, rypts)
        plt.show()
    
    eps = emopt.grid.StructuredMaterial2D(X, Y, dx, dy)
    eps.add_primitive(left)
    eps.add_primitive(right)
    eps.add_primitive(eps_background)
    
    mu = emopt.grid.ConstantMaterial2D(1.0)
    
    # set the materials used for simulation
    sim.set_materials(eps, mu)
    
    ####################################################################################
    # setup the sources
    ####################################################################################
    # setup the sources -- just a dipole in the center of the waveguide
    #Jz = np.zeros([1,1], dtype=np.complex128)
    #Mx = np.zeros([1,1], dtype=np.complex128)
    #My = np.zeros([1,1], dtype=np.complex128)
    ##Jz[M//2, N//2] = 1.0
    #Mx[0,0] = 1.0 # this is actually an electric current source
    Mz = np.zeros([1,1], dtype=np.complex128)
    Jx = np.zeros([1,1], dtype=np.complex128)
    Jy = np.zeros([1,1], dtype=np.complex128)
    #Jz[M//2, N//2] = 1.0
    #Jx[0,0] = 1.0 # this is actually an electric current source
    Mz[0,0] = 1.0 # this is actually an electric current source
    src_plane = emopt.misc.DomainCoordinates(X/2, X/2,
                                             Y/2, Y/2,
                                             0, 0, dx, dy, 1.0)
    
    sim.set_sources((Mz, Jx, Jy), src_plane)
    
    ####################################################################################
    # Build and simulate
    ####################################################################################
    #fom_plane = emopt.misc.DomainCoordinates(3*X/4, 3*X/4,
    #                                         Y/4, Y/4,
    #                                         0, 0, dx, dy, 1.0)
    #l_ind = w_pml[0]+2*dx
    #r_ind = X-w_pml[1]-2*dx
    #b_ind = w_pml[2]+2*dy
    #t_ind = Y-w_pml[3]-2*dy

    #fom_l = emopt.misc.DomainCoordinates(l_ind, l_ind, 
    #                                     b_ind, t_ind, 0, 0, dx, dy, 1.0)
    #fom_r = emopt.misc.DomainCoordinates(r_ind, r_ind, 
    #                                     b_ind, t_ind, 0, 0, dx, dy, 1.0)
    #fom_b = emopt.misc.DomainCoordinates(l_ind, r_ind, 
    #                                     b_ind, b_ind, 0, 0, dx, dy, 1.0)
    #fom_t = emopt.misc.DomainCoordinates(l_ind, r_ind, 
    #                                     t_ind, t_ind, 0, 0, dx, dy, 1.0)

    full_field = emopt.misc.DomainCoordinates(w_pml[0], X-w_pml[1],
                                              w_pml[2], Y-w_pml[3],
                                              0, 0, dx, dy, 1.0)
    sim.build()
    #sim.field_domains = [fom_l, fom_r, fom_b, fom_t, full_field]
    sim.field_domains = [full_field]

    sim.spy_A()
    

    sim.solve_forward()

    sim_area = emopt.misc.DomainCoordinates(1.0, X-1.0, 1.0, Y-1.0, 0, 0, dx, dy, 1.0)
    #Ez = sim.get_field_interp('Ez', sim_area)
    Ez = sim.get_field_interp('Ex', sim_area)

    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt
    
        extent = sim_area.get_bounding_box()[0:4]
    
        E2 = (Ez*Ez.conj()).real
        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(Ez.imag, extent=extent,
                                vmin=-np.max(Ez.imag)/1.0,
                                vmax=np.max(Ez.imag)/1.0,
                                cmap='seismic')

        #im = ax.imshow(E2, extent=extent,
        #                        vmin=0.0,
        #                        vmax=np.max(E2),
        #                        cmap='seismic')
    
        # Plot the waveguide boundaries
        #ax.plot(extent[0:2], [Y/2-h_wg/2, Y/2-h_wg/2], 'k-')
        #ax.plot(extent[0:2], [Y/2+h_wg/2, Y/2+h_wg/2], 'k-')
    
        ax.set_title('E$_z$', fontsize=18)
        ax.set_xlabel('x [um]', fontsize=14)
        ax.set_ylabel('y [um]', fontsize=14)
        f.colorbar(im)
        plt.show()
        #plt.savefig('test.pdf')


    #design_params = np.array([0.4, 0.1])
    design_params = np.array([0.1])
    am = Test(sim, left, right, d)

    am.check_gradient(design_params, fd_step=1e-8)

    fom_history = []
    callback_func = lambda p : callback(p, sim, am, full_field, fom_history)
    #bounds = [(0.0, 3.0), (0.0, 1.0)]
    bounds = [(0.0, 1.0)]
    opt = emopt.optimizer.Optimizer(am, design_params, opt_method='BFGS',
                                    Nmax=50, bounds=bounds,
                                    callback_func=callback_func, tol=1e-15)
    fom, params = opt.run()
    print(params)
