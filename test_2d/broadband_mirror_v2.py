import emopt2
from emopt2.misc import NOT_PARALLEL, run_on_master
from emopt2.adjoint_method import AdjointMethodPNF2D, AdjointMethodMO

import numpy as np
from math import pi
import matplotlib.pyplot as plt

import numpy as np
from math import pi
import copy

#CONSTANTS
c = 299792458.0
hbar = 1.055e-34
h = 2*pi*hbar
q = 1.602e-19
k = 1.3806e-23

class SpectralAveraging(AdjointMethodMO):
    def __init__(self, ams, Ep, step=1e-8):
        super(SpectralAveraging, self).__init__(ams, step=step)
        self.n = len(Ep)
        self.Ep = Ep
        ws = self.energy_weight(Ep)
        self.ws = ws
        self.integral = np.trapz(self.ws, x=Ep)

    def calc_total_fom(self, foms):
        n = self.n
        fom = np.array(foms)
        fom_TE = fom[:n]
        #fom_TM = fom[n:]
        tot = 0.0
        tot += np.trapz(fom_TE*self.ws, x=self.Ep)/self.integral
        #tot += np.trapz(fom_TM*self.ws, x=self.Ep)/self.integral
        #tot /= 2.0
        return tot

    def calc_total_gradient(self, foms, grads):
        n = self.n
        grad = np.array(grads)
        grad_TE = grad[:n,:]
        #grad_TM = grad[n:,:]
        for i in range(n):
            grad_TE[i,:] = grad_TE[i,:]*self.ws[i]
        #    grad_TM[i,:] = grad_TM[i,:]*self.ws[i]
        tot_grad_TE = np.trapz(grad_TE, x=self.Ep, axis=0)
        tot_grad_TE /= self.integral
        #tot_grad_TM = np.trapz(grad_TM, x=self.Ep, axis=0)
        #tot_grad_TM /= self.integral

        tot_grad = np.zeros(grad.shape[1])
        #tot_grad = 0.5*(tot_grad_TE+tot_grad_TM)
        tot_grad = tot_grad_TE
        return tot_grad

    def energy_weight(self, Ep, Ts=1473.0):
        weight = 2*np.power(Ep,2)/(c**2*h**3*(np.exp(Ep/(k*Ts))-1))
        return weight

class BroadbandMirrorAM(AdjointMethodPNF2D):
    def __init__(self, sim, ox_layers, si, Np, si_ts, ox_ts, ref_line, y_off, w_pml):
        super(BroadbandMirrorAM, self).__init__(sim, step=1e-8)
        self.sim = sim
        self.ox_layers = ox_layers
        self.si = si
        self.Np = Np
        self.si_ts = si_ts
        self.ox_ts = ox_ts
        self.y_off = y_off
        self.w_pml = w_pml

        self.ref_line = ref_line
        self.fom_domain = ref_line

        self.current_fom = 0.0

    def update_system(self, params):
        si_ts = self.si_ts
        ox_ts = self.ox_ts

        y_off = self.y_off
        w_pml = self.w_pml

        y_si = 0.0
        y_ox = 0.0

        superstrate_t =  h*c/(4.0*3.5*0.1*q)*1e6
        y_si += superstrate_t
        
        Np = self.Np

        for i in range(Np):
            si_t = si_ts[i] + params[i]
            ox_t = ox_ts[i] + params[i+Np]

            self.ox_layers[i].height = ox_t
            self.ox_layers[i].y0     = w_pml + y_off + y_si + y_ox + ox_t/2

            y_si += si_t
            y_ox += ox_t

        self.si.height = y_si + y_ox
        self.si.y0     = w_pml + y_off + (y_si + y_ox)/2

    @run_on_master
    def calc_f(self, sim, params):
        #if isinstance(sim, emopt2.fdtd_2d.FDFD_TM):
        #    Hz, Ex, Ey = sim.saved_fields[0]
        #    Py = np.sum(0.5*(Ex*np.conj(Hz)).real * sim.dx)
        #    self.current_fom = -1*Py
        print("got here 0")
        if False:
            pass

        elif isinstance(sim, emopt2.fdtd_2d.FDTD_TE):
            print("got here 1")
            Ez, Hx, Hy = sim.saved_fields[0]
            Py = np.sum(-0.5*(Ez*np.conj(Hx)).real * sim.dx)
            self.current_fom = -1*Py
            print("got here 2")

        else: Exception('Wrong simulation type')
        print("got here 3")
        print self.current_fom

        return self.current_fom

    def calc_dfdx(self, sim, params):
        #if isinstance(sim, emopt2.fdtd_2d.FDFD_TM):
        #    dFdHz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        #    dFdEx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        #    dFdEy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        #    # Get the fields which were recorded 
        #    if NOT_PARALLEL:
        #        Hz, Ex, Ey = sim.saved_fields[0]
        #        dFdHz[self.ref_line.j, self.ref_line.k] = -0.25*sim.dx*np.conj(Ex)
        #        dFdEx[self.ref_line.j, self.ref_line.k] = -0.25*sim.dx*np.conj(Hz)

        #    return (dFdHz, dFdEx, dFdEy)
        if False: pass

        elif isinstance(sim, emopt2.fdtd_2d.FDTD_TE):
            dFdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
            dFdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
            dFdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

            # Get the fields which were recorded 
            if NOT_PARALLEL:
                Ez, Hx, Hy = sim.saved_fields[0]
                dFdEz[self.ref_line.j, self.ref_line.k] = 0.25*sim.dx*np.conj(Hx)
                dFdHx[self.ref_line.j, self.ref_line.k] = 0.25*sim.dx*np.conj(Ez)

            return [(dFdEz, dFdHx, dFdHy)]



        else: Exception('Wrong simulation type')


    #def get_update_boxes(self, sim, params):
    #    """Define custom update boxes.

    #    When calculating gradients using the adjoint method, we need to compute
    #    the derivative of the system matrix A (running a simulation means
    #    solving Ax=b). The diagonal elements of this matrix contain the
    #    spatially-distributed permittivity and permeability and are thus
    #    modified when the geometry of the system is modified.  In many cases,
    #    changes to structure only locally modify the permittivity/permeability
    #    in a small region and thus only the correspondingly small number of
    #    elements in A need to be updated to compute the derivative. Limiting
    #    this update allows us to speed up the calculation of the gradient of
    #    the figure of merit.

    #    When modifying the etches of the grating, the only grid elements that
    #    need to be updated are contained in a rectangle that encompasses the
    #    grating coupler. This is equally true for the horizontal grating shift
    #    and the grating etch depth parameters. When updating the BOX thickness,
    #    the grid elements in a larger area must be updated.

    #    Note: This function is optional. By default, the whole grid is updated
    #    in the calculation of the derivative of A w.r.t. each design variable.
    #    """
    #    h_wg = self.h_wg
    #    y_wg = self.y_ts
    #    lenp = len(params)

    #    # define boxes surrounding grating
    #    boxes = [(0, sim.X, y_wg-h_wg, y_wg+h_wg) for i in range(lenp-1)]

    #    # for BOX, update everything (easier)
    #    boxes.append((0, sim.X, 0, sim.Y))
    #    return boxes

    def get_fom_domains(self):
        """We must return the DomainCoordinates object that corresponds to our
        figure of merit. In theory, we could have many of these.
        """
        return [self.fom_domain]

    def calc_grad_p(self, sim, params):
        """Our FOM does not depend explicitly on the design parameters so we
        return zeros."""
        return np.zeros(len(params))



def plot_update(params, fom_list, sim, am):
    """Save a snapshot of the current state of the structure.

    This function is passed to an Optimizer object and is called after each
    iteration of the optimization. It plots the current refractive index
    distribution, the electric field, and the full figure of merit history.
    """
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    print("got here 4")
    fom_list.append(current_fom)
    print(fom_list)

    Ez, Hx, Hy = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])
    print("got here 5")

    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ez.real), np.flipud(eps.real), sim.Xreal,
                            sim.Yreal, foms, fname='current_result.pdf',
                            dark=True)

    data = {}
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['eps'] = eps
    data['params'] = params
    data['foms'] = fom_list

    i = len(fom_list)
    fname = 'data/results'
    emopt.io.save_results(fname, data)

class BroadbandMirror(object):
    def __init__(self, Ep, Np, theta):
        wavelength = h*c/Ep*1e6
        w_pml = 2*wavelength
        X = 2*w_pml+20*wavelength
        Y = 2*w_pml+5.0+2.5*1*Np+2.0

        dx = wavelength/20
        #dx = 0.04
        dy = 0.04
        
        #sim_TE = emopt2.fdtd_2d.FDTD_TE(X, Y, dx, dy, wavelength)
        sim_TE = emopt2.fdtd_2d.FDTD_TE(X,Y,dx,dy,wavelength, rtol=1e-5, min_rindex=1.0,
                      nconv=100)
        #sim_TM = emopt.fdfd.FDFD_TM(X, Y, dx, dy, wavelength)
        #sim_TM.w_pml = [w_pml, w_pml, w_pml, w_pml]

        sim_TE.Nmax = 1000*sim_TE.Ncycle
        #w_pml = dx * 30 # set the PML width
        sim_TE.w_pml = [w_pml, w_pml, w_pml, w_pml]
        sim_TE.Sc = 0.5

        X = sim_TE.X
        Y = sim_TE.Y
        #$M = sim_TE.M
        #$N = sim_TE.N
        w_pml = sim_TE.w_pml[0] 

        eps_si = 3.5**2
        eps_ox = 1.6**2

        M1 = h*c/(0.1*q)*1e6
        M2 = h*c/(0.74*q)*1e6

        a = h*c/(0.1*q)*1e6
        b = 1.0/(Np-1)*(M1-M2)

        y_off = 5.0
        y_si = 0.0
        y_ox = 0.0
        
        ox_layers = []
        si_ts = []
        ox_ts = []

        superstrate_t =  h*c/(4.0*np.sqrt(eps_si)*0.1*q)*1e6

        y_si += superstrate_t

        for i in range(Np):
            si_t = (a-b*i)/(4.0*np.sqrt(eps_si))
            ox_t = (a-b*i)/(4.0*np.sqrt(eps_ox))

            ox_rect = emopt2.grid.Rectangle(X/2.0, w_pml+y_off+y_si+si_t+ox_t/2,
                                           X, ox_t)
            y_si += si_t
            y_ox += ox_t

            ox_rect.layer = 1
            ox_rect.material_value = eps_ox
            ox_layers.append(ox_rect)

            si_ts.append(si_t)
            ox_ts.append(ox_t)

        si_ts = np.array(si_ts)
        ox_ts = np.array(ox_ts)

        background = emopt2.grid.Rectangle(X/2, Y/2, X, Y)
        background.layer = 3
        background.material_value = 1.0

        si = emopt2.grid.Rectangle(X/2, w_pml+y_off+(y_si+y_ox)/2, X, y_si+y_ox)
        si.layer = 2
        si.material_value = eps_si

        eps = emopt2.grid.StructuredMaterial2D(X, Y, dx, dy)

        for r in ox_layers:
            eps.add_primitive(r)

        eps.add_primitive(si)
        eps.add_primitive(background)

        mu = emopt2.grid.ConstantMaterial2D(1.0)

        sim_TE.set_materials(eps, mu)
        #sim_TM.set_materials(eps, mu)

        #eps_TM = copy.deepcopy(eps)
        #mu_TM = copy.deepcopy(mu)

        #sim_TM.set_materials(eps_TM, mu_TM)


        ####################################################################################
        # Setup the sources
        ####################################################################################
        # place the source in the simulation domain

        center = X/2+3.5*wavelength
        src_line = emopt2.misc.DomainCoordinates(w_pml, X-w_pml, w_pml+y_off/2,
                                     w_pml+y_off/2, 0, 0, dx, dy, 1.0)

        if(NOT_PARALLEL):
            print('Generating sources...')

        M = src_line.Nx
        N = src_line.Ny
        
        Jz = np.zeros([N,M], dtype=np.complex128)
        Mx = np.zeros([N,M], dtype=np.complex128)
        My = np.zeros([N,M], dtype=np.complex128)

        Jzz, Mxx, Myy = self.gaussian_src(src_line, center, 4*wavelength, -np.pi/4, sim_TE.wavelength)
        Jz[0,:] = Jzz
        Mx[0,:] = Mxx
        My[0,:] = Myy
        srcs = (Jz, Mx, My)
        sim_TE.set_sources(srcs, src_line)

        #Hz, Jx, Jy = self.gaussian_src(src_line, center, 4*wavelength, -np.pi/4, sim_TM.wavelength)
        #srcs = (Hz, -Jx, -Jy)
        #sim_TM.set_sources(srcs, src_domain=src_line)


        ####################################################################################
        # Setup the mode match domain
        ####################################################################################
        full_field = emopt2.misc.DomainCoordinates(w_pml, X-w_pml, w_pml, Y-w_pml, 0.0, 0.0,
                                                  dx, dy, 1.0)
        ref_line = emopt2.misc.DomainCoordinates(3*w_pml, X-3*w_pml, w_pml+y_off/4, w_pml+y_off/4, 0.0, 0.0,
                                                  dx, dy, 1.0)
        sim_TE.field_domains = [ref_line, full_field]
        #sim_TM.field_domains = [ref_line, full_field]
        #sim_TE.field_domains = [ref_line]
        #sim_TM.field_domains = [ref_line]

        ####################################################################################
        # Build the system
        ####################################################################################
        sim_TE.build()
        #sim_TM.build()

        am_TE = BroadbandMirrorAM(sim_TE, ox_layers, si, Np, si_ts, ox_ts, ref_line, y_off, w_pml)
        #am_TM = BroadbandMirrorAM(sim_TM, ox_layers, si, Np, si_ts, ox_ts, ref_line, y_off, w_pml)

        self.am_TE = am_TE
        #self.am_TM = am_TM
        self.sim_TE = sim_TE
        #self.sim_TM = sim_TM

    @staticmethod
    def gaussian_src(src_line, center, w0, theta, wavelength):
        x = src_line.x-center
        Nx = src_line.Nx
        dx = x[1]-x[0]
    
        phase_factor = lambda i: np.exp(1j*2*np.pi*dx*i/wavelength*np.sin(theta))
        phase_ramp = np.array([phase_factor(i) for i in range(Nx)], dtype=np.complex128)
    
        amplitude_ramp = np.exp(-np.power(x,2.0)/w0**2)
    
        zr = np.pi*w0**2/wavelength
        psi_fn = lambda i: np.sin(theta)*dx*i/zr
    
        psi_ramp = np.array([np.exp(0.0*1j*np.arctan(psi_fn(i))) for i in range(Nx)], dtype=np.complex128)
        
    
        Jz = phase_ramp * amplitude_ramp * psi_ramp
        Mx = -1*phase_ramp * amplitude_ramp * np.cos(theta) * psi_ramp
        My = -1*phase_ramp * amplitude_ramp * np.sin(theta) * psi_ramp
    
        return(Jz, Mx, My)


if __name__ == '__main__':

    ####################################################################################
    # Setup the optimization
    ####################################################################################
    Np = 6
    Eps = q*np.linspace(0.1, 0.74, num=2)
    #thetas = np.pi/180.0*np.arange(1,89)
    thetas = np.array([-45.0])
    thetas *= np.pi/180.0

    mirrors = []
    sims_TE = []
    sims_TM = []
    ams_TE = []
    ams_TM = []
    for Ep in Eps:
        mirror = BroadbandMirror(Ep, Np, thetas)
        mirrors.append(mirror)
        sims_TE.append(mirror.sim_TE)
        #sims_TM.append(mirror.sim_TM)
        ams_TE.append(mirror.am_TE)
        #ams_TM.append(mirror.am_TM)

    design_params = np.zeros((2*Np))
    
    #ams = ams_TE + ams_TM
    ams = ams_TE
    am = SpectralAveraging(ams, Eps)

    #am.check_gradient(design_params, np.arange(0,2*Np,2))

    fom_list = []
    callback = lambda x : plot_update(x, fom_list, sims_TE[1], ams_TE[1])

    # setup and run the optimization!
    opt = emopt2.optimizer.Optimizer(am, design_params, tol=1e-5,
                                    callback_func=callback,
                                    opt_method='BFGS',
                                    Nmax=10)

    # Run the optimization
    final_fom, final_params = opt.run()
