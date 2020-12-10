"""Implements a simple 3D CW-FDTD (continuous wave finite difference frequenct domain)
solver for simulating Maxwell's equations.

The FDTD method provides a simple, efficient, highly parallelizable, and scalable
solution for solving Maxwell's equations. Typical implementations of FDTD, however,
can be tricky to use with highly dispersive materials and are tough to use in
conjunction with the adjoint method for inverse electromagnetic design.

In order to overcome these issues, this implementation uses a ramped-CW source which
allows us to solve for the frequency domain fields at the exact frequency we care
about without doing any form of interpolation. While this eliminates one of FDTD's
advantages (obtaining broadband info with a single simulation), it is easier to
implement and plays nice with the adjoint method which is easily derivable for
frequency-domain solvers. Furthermore, contrary to explicit frequency domain solvers,
CW-FDTD is highly parallelizable which enables it to run quite a bit faster.
Furthermore, compared to frequency domain solvers like FDFD, CW-FDTD consumes
considerably less RAM, making it useful for optimizing much larger devices at much
higher resolutions.

"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import range
from builtins import object
from .simulation import MaxwellSolver
from .defs import FieldComponent
from .misc import DomainCoordinates, RANK, MathDummy, NOT_PARALLEL, COMM, \
info_message, warning_message, N_PROC, run_on_master
from .fdtd_2d_cyl_ctypes import libFDTD
from .modes import ModeTE, ModeTM
import petsc4py
import sys
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
from math import pi
from mpi4py import MPI
import sys

__author__ = "Andrew Michaels"
__license__ = "GPL License, Version 3.0"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

class SourceArray_TE(object):
    """A container for source arrays and its associated parameters.

    Parameters
    ----------
    Jz : numpy.ndarray
        The x-component of the electric current density
    Mx : numpy.ndarray
        The x-component of the magnetic current density
    My : numpy.ndarray
        The x-component of the magnetic current density
    j0 : int
        The LOCAL starting y index of the source arrays
    k0 : int
        The LOCAL starting z index of the source arrays
    J : int
        The y width of the source array
    K : int
        The x width of the source array
    """
    def __init__(self, Jz, Mx, My, j0, k0, J, K):
        self.Jz = Jz
        self.Mx = Mx
        self.My = My

        self.j0 = j0
        self.k0 = k0
        self.J  = J
        self.K  = K

class FDTD_TE(MaxwellSolver):
    """A 3D continuous-wave finite difference time domain (CW-FDTD) solver.

    This class implements a continuous-wave finite difference time domain
    solver which solves for the freqeuncy-domain fields using the finite
    difference time domain method. Unlike the typical pulsed FDTD method, this
    implementation uses a ramped CW source which ensures that the simulation
    error exhibits convergent behavior (which is useful for optimization).
    Furthermore, this CW eliminates the need to perform discrete fourier
    transforms and allows us to compute the fields at the exact
    frequency/wavelength we desire (without relying on any interpolation which
    complicates optimizations).

    Compared to :class:`fdfd.FDFD_3D`, FDTD will generally scale to
    larger/higher resolution problems much better. For such problems, the
    memory consumption can be an order of magnitude or more lower and it
    parallelizes significantly better. For very small problems, however,
    :class:`fdfd.FDFD_3D` may be faster.

    Notes
    -----
    1. Complex materials are not yet supported, however they will be in the
    future!

    2. Because of how power is computed, you should not do any calculations
    within 1 grid cells of the PMLs

    3. Currently, the adjoint simulation can have some difficulty converging
    when used for a power-normalized figure of merit. To get around this, you
    can either increase the relative tolerance, or better yet set a maximum
    number of time steps using Nmax.

    4. Power is currently computed using a transmission box which is placed 1
    grid cell within the PML regions and does not account for material
    absorption.

    Parameters
    ----------
    X : float
        The width of the simulation in the x direction
    Y : float
        The width of the simulation in the y direction
    Z : float
        The width of the simulation in the z direction
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    dz : float
        The grid spacing in the z direction
    wavelength : float
        The wavelength of the source (and fields)
    rtol : float (optional)
        The relative tolerance used to terminate the simulation.
        (default = 1e-6)
    nconv : int (optional)
        The number of field points to check for convergence. (default = 100)
    min_rindex : float (optional)
        The minimum refractive index of the simulation. Setting this parameter
        correctly can speed up your simulations by quite a bit, however you
        must be careful that its value does not exceed the minimum refractive
        index in your simulation. (default = 1.0)
    complex_eps : boolean (optional)
        Tells the solver if the permittivity is complex-valued. Setting this to
        False can speed up the solver when run on fewer cores (default = False)

    Attributes
    ----------
    X : float
        The width of the simulation in the x direction
    Y : float
        The width of the simulation in the y direction
    Z : float
        The width of the simulation in the z direction
    dx : float
        The grid spacing in the x direction
    dy : float
        The grid spacing in the y direction
    dz : float
        The grid spacing in the z direction
    Nx : int
        The number of grid cells along the x direction
    Ny : int
        The number of grid cells along the y direction
    Nz : int
        The number of grid cells along the z direction
    courant_num : float
        The courant number (fraction of maximum time step needed for
        stability). This must be <= 1. By default it is 0.95. Don't change this
        unless you simulations appear to diverge for some reason.
    wavelength : float
        The wavelength of the source (and fields)
    eps : emopt.grid.Material3D
        The permittivity distribution.
    mu : emopt.grid.Material3D
        The permeability distribution.
    rtol : float (optional)
        The relative tolerance used to terminate the simulation.
        (default = 1e-6)
    src_min_value : float
        The starting value of the ramped source. The source is ramped using a
        smooth envelope function which never truely goes to zero. This
        attribute lets us set the "zero" value which will effect the
        convergence time and minimum error. (default = 1e-5)
    src_ramp_time : int
        The number of time steps over which the source is ramped. This will
        affect the convergence time and final error. Note: faster ramps lead to
        higher error (in the current implementation) and do not gurantee that
        the simulation will converge faster. (default = 15*Nlambda)
    Nmax : int
        Max number of time steps
    Nlambda : float
        Number of spatial steps per wavelength (in minimum index material)
    Ncycle : float
        Number of time steps per period of oscillation of the fields.
    bc : str
        The the boundary conditions (PEC, field symmetries, etc)
    w_pml : list of floats
        The list of pml widths (in real coordinates) in the format [xmin, xmax,
        ymin, ymax, zmin, zmax]. Use this to change the PML widths.
    w_pml_xmin : int
        The number of grid cells making up with PML at the minimum x boundary.
    w_pml_xmax : int
        The number of grid cells making up with PML at the maximum x boundary.
    w_pml_ymin : int
        The number of grid cells making up with PML at the minimum y boundary.
    w_pml_ymax : int
        The number of grid cells making up with PML at the maximum y boundary.
    w_pml_zmin : int
        The number of grid cells making up with PML at the minimum z boundary.
    w_pml_zmax : int
        The number of grid cells making up with PML at the maximum z boundary.
    X_real : float
        The width of the simulation excluding PMLs.
    Y_real : float
        The height of the simulation excluding PMLs.
    """

    def __init__(self, X, Y, dx, dy, wavelength, rtol=1e-6, nconv=None,
                 min_rindex=1.0, complex_eps=False):
        super(FDTD_TE, self).__init__(2)

        if(nconv is None):
            nconv = N_PROC*10
        elif(nconv < N_PROC*2):
            warning_message('Number of convergence test points (nconv) is ' \
                            'likely too low. If the simulation does not ' \
                            'converge, increase nconv to > 2 * # processors',
                            'emopt.fdtd')

        self._dx = dx
        self._dy = dy

        Nx = int(np.ceil(X/dx)+1); self._Nx = Nx
        Ny = int(np.ceil(Y/dy)+1); self._Ny = Ny

        self._X = dx * (Nx-1)
        self._Y = dy * (Ny-1)

        self._wavelength = wavelength
        self._R = wavelength/(2*pi)

        ## Courant number < 1
        ## Why are some problems unstable for larger Sc?
        self._Sc = 0.95
        self._min_rindex = min_rindex
        # sqrt(3) needed or sqrt(2)?
        dt = self._Sc * np.min([dx, dy])/self._R / np.sqrt(2) * min_rindex
        self._dt = dt


        # stencil_type=1 => box
        # stencil_width=1 => 1 element ghosted region
        # boundary_type=1 => ghosted simulation boundary (padded everywhere)
        da = PETSc.DA().create(sizes=[Nx, Ny], dof=1, stencil_type=0,
                               stencil_width=1, boundary_type=1)

        self._da = da
        ## Setup the distributed array. Currently, we need 2 for each field
        ## component (1 for the field, and 1 for the averaged material) and 1
        ## for each current density component
        self._vglobal = da.createGlobalVec() # global for data sharing

        pos, lens = da.getCorners()
        k0, j0 = pos
        K, J = lens

        # field arrays
        self._Ez = np.zeros(((K+2)*(J+2),), dtype=np.double)
        self._Hx = np.zeros(((K+2)*(J+2),), dtype=np.double)
        self._Hy = np.zeros(((K+2)*(J+2),), dtype=np.double)

        # material arrays -- global since we dont need to pass values around
        self._eps_z = da.createGlobalVec()
        self._mu_x = da.createGlobalVec()
        self._mu_y = da.createGlobalVec()

        # Frequency-domain field arrays for forward simulation
        # Two sets of fields for two snapshots in time. The frequency-domain
        # fields are stored in the t0 field set
        self._Ez_fwd_t0 = da.createGlobalVec()
        self._Hx_fwd_t0 = da.createGlobalVec()
        self._Hy_fwd_t0 = da.createGlobalVec()

        self._Ez_fwd_t1 = da.createGlobalVec()
        self._Hx_fwd_t1 = da.createGlobalVec()
        self._Hy_fwd_t1 = da.createGlobalVec()

        # Frequency-domain field arrays for ajdoint simulation
        self._Ez_adj_t0 = da.createGlobalVec()
        self._Hx_adj_t0 = da.createGlobalVec()
        self._Hy_adj_t0 = da.createGlobalVec()

        self._Ez_adj_t1 = da.createGlobalVec()
        self._Hx_adj_t1 = da.createGlobalVec()
        self._Hy_adj_t1 = da.createGlobalVec()


        # setup the C library which takes care of the E and H updates
        # Nothing complicated here -- just passing all of the sim parameters
        # and work vectors over to the c library
        self._libfdtd = libFDTD.FDTD_TE_new()
        libFDTD.FDTD_TE_set_wavelength(self._libfdtd, wavelength)
        libFDTD.FDTD_TE_set_physical_dims(self._libfdtd, X, Y, dx, dy)
        libFDTD.FDTD_TE_set_grid_dims(self._libfdtd, Nx, Ny)
        libFDTD.FDTD_TE_set_local_grid(self._libfdtd, k0, j0, K, J)
        libFDTD.FDTD_TE_set_dt(self._libfdtd, dt)
        libFDTD.FDTD_TE_set_field_arrays(self._libfdtd,
                self._Ez,
                self._Hx, self._Hy)
        libFDTD.FDTD_TE_set_mat_arrays(self._libfdtd,
                self._eps_z.getArray(),
                self._mu_x.getArray(), self._mu_y.getArray())

        # set whether or not materials are complex valued
        libFDTD.FDTD_TE_set_complex_eps(self._libfdtd, complex_eps)

        ## Setup default PML properties
        w_pml = 15
        # Thicker PML seems to work better
        self._w_pml = [w_pml*dx, w_pml*dx, \
                      w_pml*dy, w_pml*dy]
        self._w_pml_xmin = w_pml
        self._w_pml_xmax = w_pml
        self._w_pml_ymin = w_pml
        self._w_pml_ymax = w_pml

        #self._pml_sigma = 3.0
        #self._pml_alpha = 0.0
        #self._pml_kappa = 2.0
        #self._pml_pow = 3.0
        self._pml_sigma = 4.0
        self._pml_alpha = 1.0
        self._pml_kappa = 3.0
        self._pml_pow = 3.0

        libFDTD.FDTD_TE_set_pml_widths(self._libfdtd, w_pml, w_pml,
                                                   w_pml, w_pml)
        libFDTD.FDTD_TE_set_pml_properties(self._libfdtd, self._pml_sigma,
                                        self._pml_alpha, self._pml_kappa,
                                        self._pml_pow)
        libFDTD.FDTD_TE_build_pml(self._libfdtd)

        ## Setup the source properties
        Nlambda = wavelength / np.min([dx, dy]) / self._min_rindex
        Ncycle = Nlambda * np.sqrt(2)
        self._Nlambda = Nlambda # spatial steps per wavelength
        self._Ncycle = Ncycle #  time steps per period of oscillation

        self._src_T    = Ncycle * 20.0 * self._dt
        self._src_min  = 1e-4
        self.Nmax = Nlambda*500
        libFDTD.FDTD_TE_set_source_properties(self._libfdtd, self._src_T,
                                           self._src_min)

        self._rtol = rtol
        self.verbose = 2

        ## determine the points used to check for convergence on this process
        # these are roughly evenly spaced points in the local-stored field
        # vectors. We exclude PMLs from these checks. Consider moving this to
        # C++ if this ever gets slow (possible for larger problems?)
        n_pts = int(nconv / N_PROC)
        self._conv_pts = np.arange(0, J*K, int(J*K/n_pts))
        self._nconv = len(self._conv_pts)

        self._sources = []
        self._adj_sources = []

        ## Define a natural vector for field retrieval
        # We could do this much more efficiently, but except for huge problems,
        # it isn't really necessary and this way the code is simpler
        self._vn = da.createNaturalVec()

        # default boundary conditions = PEC
        self._bc = ['0', '0']
        libFDTD.FDTD_TE_set_bc(self._libfdtd, ''.join(self._bc).encode('ascii'))

        # make room for eps and mu
        self._eps = None
        self._mu = None

        # defome a GhostComm object which we will use to share edge values of
        # the fields between processors
        self._gc = GhostComm(k0, j0, K, J, Nx, Ny)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wlen):
        if(wlen <= 0):
            raise ValueError('Wavelength must be >= 0.')
        else:
            self._wavelength = wlen
            self._R = wlen/2/pi
            libFDTD.FDTD_TE_set_wavelength(self._libfdtd, wlen)

            ds = np.min([self._dx, self._dy])
            dt = self._Sc * ds/self._R / np.sqrt(2) * self._min_rindex
            self._dt = dt
            libFDTD.FDTD_TE_set_dt(self._libfdtd, dt)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def Nx(self):
        return self._Nx

    @property
    def Ny(self):
        return self._Ny

    @property
    def eps(self):
        return self._eps

    @property
    def mu(self):
        return self._mu

    @property
    def courant_num(self):
        return self._Sc

    @courant_num.setter
    def courant_num(self, Sc):
        if(Sc > 1.0):
            warning_message('A courant number greater than 1 will lead to ' \
                            'instability.', 'emopt.fdtd')
        self._Sc = Sc
        ds = np.min([self._dx, self._dy])
        dt = self._Sc * ds/self._R / np.sqrt(2) * self._min_rindex
        self._dt = dt
        libFDTD.FDTD_TE_set_dt(self._libfdtd, dt)

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, rtol):
        if(rtol <= 0):
            raise ValueError('Relative tolerance must be greater than zero.')
        self._rtol = rtol

    @property
    def src_min_value(self):
        return self._src_min

    @src_min_value.setter
    def src_min_value(self, new_min):
        if(new_min <= 0):
            raise ValueError('Minimum source value must be greater than zero.')

        self._src_min = new_min
        libFDTD.FDTD_TE_set_source_properties(self._libfdtd, self._src_T,
                                           self._src_min)

    @property
    def Nlambda(self):
        return self._Nlambda

    @property
    def Ncycle(self):
        return self._Ncycle

    @property
    def src_ramp_time(self):
        return self._src_T

    @src_ramp_time.setter
    def src_ramp_time(self, new_T):
        if(new_T <= 0):
            raise ValueError('Source ramp time must be greater than zero.')
        elif(new_T <= self._Nlambda*5):
            warning_message('Source ramp time is likely too short. '\
                            'Simulations may not converge.',
                            'emopt.fdtd')
            self._src_T = new_T
            libFDTD.FDTD_TE_set_source_properties(self._libfdtd, self._src_T,
                                               self._src_min)
        else:
            self._src_T = new_T
            libFDTD.FDTD_TE_set_source_properties(self._libfdtd, self._src_T,
                                               self._src_min)
    @property
    def bc(self):
        return ''.join(self._bc)

    @bc.setter
    def bc(self, newbc):
        if(len(newbc) != 2):
            raise ValueError('Incorrect number of boundary conditions specified!')

        for bc in newbc:
            if(not bc in '0EHP'):
                raise ValueError('Unrecognized boundary condition: %s' % (bc))
            if(bc == 'P'):
                raise NotImplementedError('Periodic boundary conditions not yet ' \
                                          'implemented')

        self._bc = list(newbc)
        libFDTD.FDTD_TE_set_bc(self._libfdtd, ''.join(self._bc).encode('ascii'))

    @property
    def w_pml(self):
        return self._w_pml

    @w_pml.setter
    def w_pml(self, w_pml):
        if(len(w_pml) != 4):
            raise ValueError('Incorrect number of pml widths specified. You ' \
                             'must specify 4 widths (xmin, xmax, ymin, ymax).')

        dx = self._dx; dy = self._dy;

        self._w_pml_xmin = int(w_pml[0]/dx)
        self._w_pml_xmax = int(w_pml[1]/dx)
        self._w_pml_ymin = int(w_pml[2]/dy)
        self._w_pml_ymax = int(w_pml[3]/dy)

        self._w_pml = [self._w_pml_xmin*dx, self._w_pml_xmax*dx, \
                       self._w_pml_ymin*dy, self._w_pml_ymax*dy]

        # rebuild the PMLs
        libFDTD.FDTD_TE_set_pml_widths(self._libfdtd, self._w_pml_xmin,
                                                   self._w_pml_xmax,
                                                   self._w_pml_ymin,
                                                   self._w_pml_ymax)
        libFDTD.FDTD_TE_build_pml(self._libfdtd)

    @property
    def w_pml_xmin(self):
        return self._w_pml_xmin

    @property
    def w_pml_xmax(self):
        return self._w_pml_xmax

    @property
    def w_pml_ymin(self):
        return self._w_pml_ymin

    @property
    def w_pml_ymax(self):
        return self._w_pml_ymax

    @property
    def X_real(self):
        return self._X-self._w_pml[0]-self._w_pml[1]

    @property
    def Y_real(self):
        return self._Y-self._w_pml[2]-self._w_pml[3]

    def set_materials(self, eps, mu):
        self._eps = eps
        self._mu = mu

    def __get_local_domain_overlap(self, domain):
        ## Get the local and global ijk bounds which correspond to the overlap
        # between the supplied domain and the grid chunk owned by this
        # processor
        #
        # Returns four tuples or four None
        #   None if no overlap
        #   (global start indices), (local start indices), 
        #        (domain start indices), (widths of overlap)
        pos, lens = self._da.getCorners()
        k0, j0 = pos
        K, J = lens

        # determine the local portion of the array that is relevant
        # First: don't do anything if this process does not contain the
        # provided source arrays
        if(j0 >= domain.j2):       return None, None, None, None
        elif(j0 + J <= domain.j1): return None, None, None, None
        elif(k0 >= domain.k2):     return None, None, None, None
        elif(k0 + K <= domain.k1): return None, None, None, None

        jmin = 0; kmin = 0
        jmax = 0; kmax = 0

        if(j0 + J > domain.j2): jmax = domain.j2
        else: jmax = j0 + J

        if(k0 + K > domain.k2): kmax = domain.k2
        else: kmax = k0 + K

        if(j0 > domain.j1): jmin = j0
        else: jmin = domain.j1

        if(k0 > domain.k1): kmin = k0
        else: kmin = domain.k1

        jd1 = jmin - domain.j1
        kd1 = kmin - domain.k1

        return (jmin, kmin), \
               (jmin - j0, kmin - k0), \
               (jd1, kd1), \
               (jmax - jmin, kmax - kmin)

    def __set_sources(self, src, domain, adjoint=False):
        ## Set the source arrays. The process is identical for forward and
        # adjoint, so this function manages both using a flag. The process
        # consists of finding which portion of the supplied arrays overlaps
        # with the space owned by this processor, copying that piece, and then
        # submitting the cropped arrays to libFDTD which will convert the
        # complex-valued source to an amplitude and phase.
        Jz, Mx, My = src

        g_inds, l_inds, d_inds, sizes = self.__get_local_domain_overlap(domain)
        if(g_inds == None): return # no overlap between source and this chunk

        jd1 = d_inds[0]; jd2 = d_inds[0] + sizes[0];
        kd1 = d_inds[1]; kd2 = d_inds[1] + sizes[1];

        # get the pieces which are relevant to this processor
        Jzs = np.copy(Jz[jd1:jd2, kd1:kd2]).ravel()
        Mxs = np.copy(Mx[jd1:jd2, kd1:kd2]).ravel()
        Mys = np.copy(My[jd1:jd2, kd1:kd2]).ravel()

        src = SourceArray_TE(Jzs,
                             Mxs, Mys,
                             l_inds[0], l_inds[1],
                             sizes[0], sizes[1])

        print(src.J)
        print(src.K)
        print(src.j0)
        print(src.k0)

        if(adjoint): self._adj_sources.append(src)
        else: self._sources.append(src)

        libFDTD.FDTD_TE_add_source(self._libfdtd,
                                src.Jz,
                                src.Mx, src.My,
                                src.j0, src.k0,
                                src.J, src.K,
                                True)

    def set_sources(self, src, domain, mindex=0):
        """Set a simulation source.

        Simulation sources can be set either using a set of 6 arrays (Jx, Jy,
        Jz, Mx, My, Mz) or a :class:`modes.ModeFullVector` object. In either
        case, a domain must be provided which tells the simulation where to put
        those sources.

        This function operates in an additive manner meaning that multiple
        calls will add multiple sources to the simulation. To replace the
        existing sources, simply call :def:`clear_sources` first.

        Parameters
        ----------
        src : tuple or modes.ModeFullVector
            The source arrays or mode object containing source data
        domain : misc.DomainCoordinates2D
            The domain which specifies where the source is located
        mindex : int (optional)
            The mode source index. This is only relevant if using a
            ModeFullVector object to set the sources. (default = 0)
        """
        if(type(src) == ModeTE):
            Jzs, Mxs, Mys = src.get_source(mindex, self._dx,
                                                   self._dy)

            Jzs = COMM.bcast(Jzs, root=0)
            Mxs = COMM.bcast(Mxs, root=0)
            Mys = COMM.bcast(Mys, root=0)

            src_arrays = (Jzs, Mxs, Mys)
        else:
            src_arrays = src

        self.__set_sources(src_arrays, domain, adjoint=False)
        COMM.Barrier()

    def set_adjoint_sources(self, src):
        """Set the adjoint sources.

        This function works a bit differently than set_sources in order to
        maintain compatibility with the adjoint_method class. It takes a single
        argument which is a list of lists containing multiple sets of source
        arrays and corresponding domains.

        TODO
        ----
        Clean up EMopt interfaces so that this doesn't feel so hacky.

        Paramters
        ---------
        src : list
            List containing two lists: 1 with sets of source arrays and on with
            DomainCoordinates2D which correspond to those source array sets.
        """
        # clear the old sources
        self.clear_adjoint_sources()

        # add the new sources
        for dFdx, domain in zip(src[0], src[1]):
            self.add_adjoint_sources(dFdx, domain)

    def add_adjoint_sources(self, src, domain):
        """Add an adjoint source.

        Add an adjoint source using a set of 6 current source arrays.

        Parameters
        ----------
        src : list or tuple of numpy.ndarray
            The list/tuple of source arrays in the following order: (Jx, Jy, Jz,
            Mx, My, Mz)
        domain : misc.DomainCoordinates2D
            The domain which specifies the location of the source arrays
        """
        self.__set_sources(src, domain, adjoint=True)

    def clear_sources(self):
        """Clear simulation sources."""
        # Cleanup old source arrays -- not strictly necessary but it's nice to
        # guarantee that the space is freed up
        for src in self._sources: del src
        self._sources = []

    def clear_adjoint_sources(self):
        """Clear the adjoint simulation sources."""
        for src in self._adj_sources: del src
        self._adj_sources = []

    def build(self):
        """Assemble the strcutre.

        This involves computing the permittiviy and permeability distribution
        for the simulation.
        """
        if(self.verbose > 0 and NOT_PARALLEL):
            info_message('Building FDTD system...')

        eps = self._eps
        mu = self._mu

        pos, lens = self._da.getCorners()
        k0, j0 = pos
        K, J = lens

        eps.get_values(k0, k0+K, j0, j0+J,
                       sx=0.0, sy=0.0,
                       arr=self._eps_z.getArray())

        mu.get_values(k0, k0+K, j0, j0+J,
                      sx=0.0, sy=0.5,
                      arr=self._mu_x.getArray())

        mu.get_values(k0, k0+K, j0, j0+J,
                       sx=0.5, sy=0.0,
                       arr=self._mu_y.getArray())


    def update(self, bbox=None):
        """Update the permittivity and permeability distribution.

        A small portion of the whole simulation region can be updated by
        supplying a bounding box which encompasses the desired region to be
        updated.

        Notes
        -----
        The bounding box accepted by this class is currently different from the
        FDFD solvers in that it is specified using real-space coordinates not
        indices. In the future, the other solvers will implement this same
        functionality.

        TODO
        ----
        Implement 3 point frequency domain calculation. This should be
        more accurate and produce more consistent results.

        Parameters
        ----------
        bbox : list of floats (optional)
            The region of the simulation which should be updated. If None, then
            the whole region is updated. Format: [xmin, xmax, ymin, ymax, zmin,
            zmax]
        """
        if(bbox == None):
            verbose = self.verbose
            self.verbose = 0
            self.build()
            self.verbose = verbose

        else:
            eps = self._eps
            mu = self._mu

            pos, lens = self._da.getCorners()
            k0, j0 = pos
            K, J = lens

            #bbox = DomainCoordinates2D(bbox[0], bbox[1], bbox[2], bbox[3],
            #                         self._dx, self._dy)
            bbox = DomainCoordinates(bbox[0], bbox[1], bbox[2], bbox[3], 0.0, 0.0,
                                     self._dx, self._dy, 1.0)

            g_inds, l_inds, d_inds, sizes = self.__get_local_domain_overlap(bbox)
            if(g_inds == None): return # no overlap between source and this chunk

            temp = np.zeros([sizes[0], sizes[1]], dtype=np.complex128)

            lj = slice(l_inds[0], l_inds[0]+sizes[0])
            lk = slice(l_inds[1], l_inds[1]+sizes[1])

            # update eps_z
            eps_z = self._eps_z.getArray()
            eps_z = np.reshape(eps_z, [J,K])
            eps.get_values(g_inds[1], g_inds[1]+sizes[1],
                           g_inds[0], g_inds[0]+sizes[0],
                           sx=0.0, sy=0.0,
                           arr=temp)
            eps_z[lj, lk] = temp


            # update mu_x
            mu_x = self._mu_x.getArray()
            mu_x = np.reshape(mu_x, [J,K])
            mu.get_values(g_inds[1], g_inds[1]+sizes[1],
                          g_inds[0], g_inds[0]+sizes[0],
                          sx=0.0, sy=0.5,
                          arr=temp)
            mu_x[lj, lk] = temp

            # update mu_y
            mu_y = self._mu_y.getArray()
            mu_y = np.reshape(mu_y, [J,K])
            mu.get_values(g_inds[1], g_inds[1]+sizes[1],
                          g_inds[0], g_inds[0]+sizes[0],
                          sx=0.5, sy=0.0,
                          arr=temp)
            mu_y[lj, lk] = temp


    def __solve(self):
        ## Solve Maxwell's equations. This process is identical for the forward
        # and adjoint simulation. The only difference is the specific sources
        # and auxillary arrays used for saving the final frequency-domain
        # fields.
        # setup spatial derivative factors
        R = self._wavelength/(2*pi)
        odx = R/self._dx
        ody = R/self._dy
        Nx, Ny = self._Nx, self._Ny

        COMM.Barrier()

        # define time step
        dt = self._dt

        pos, lens = self._da.getCorners()
        k0, j0 = pos
        K, J = lens

        # Reset field values, pmls, etc
        libFDTD.FDTD_TE_reset_pml(self._libfdtd)
        self._Ez.fill(0)
        self._Hx.fill(0); self._Hy.fill(0);

        da = self._da
        Tn = np.int(self._Ncycle*3/4)
        p = 0

        Ez0 = np.zeros(self._nconv)
        Ez1 = np.zeros(self._nconv)
        Ez2 = np.zeros(self._nconv)

        phi0 = np.zeros(self._nconv)
        phi1 = np.zeros(self._nconv)
        phi2 = np.zeros(self._nconv)

        A0 = np.zeros(self._nconv)
        A1 = np.zeros(self._nconv)
        A2 = np.zeros(self._nconv)

        t0 = 0; t1 = 0; t2 = 0

        A_change = 1
        phi_change = 1

        recvbuff = []

        # Note: ultimately we care about the error in the real and imaginary
        # part of the fields. The error in the real/imag parts goes as the
        # square of the phase error.
        amp_rtol = self._rtol
        phi_rtol = np.sqrt(self._rtol)

        n = 0
        while(A_change > amp_rtol or phi_change > phi_rtol or \
              np.isnan(A_change) or np.isinf(A_change) or \
              np.isnan(phi_change) or np.isinf(phi_change)):

            if(n > self.Nmax):
                warning_message('Maximum number of time steps exceeded.',
                                'emopt.fdtd')
                break


            libFDTD.FDTD_TE_update_H(self._libfdtd, n, n*dt)

            self._gc.update_local_vector(self._Hx)
            self._gc.update_local_vector(self._Hy)

            libFDTD.FDTD_TE_update_E(self._libfdtd, n, (n+0.5)*dt)

            self._gc.update_local_vector(self._Ez)

            if(p == Tn-1):
                # Update times of field snapshots
                t0 = t1
                t1 = t2
                t2 = n*dt

                phi0[:] = phi1
                A0[:] = A1
                for q in range(self._nconv):
                    conv_index = self._conv_pts[q]

                    # start with Ex
                    Ez0[q] = Ez1[q]
                    Ez1[q] = Ez2[q]
                    Ez2[q] = np.real(self._Ez[conv_index])

                    phasez = libFDTD.FDTD_TE_calc_phase_3T(t0, t1, t2,
                                                        Ez0[q], Ez1[q], Ez2[q])

                    ampz = libFDTD.FDTD_TE_calc_amplitude_3T(t0, t1, t2, Ez0[q],
                                                          Ez1[q], Ez2[q], phasez)

                    if(ampz < 0):
                        ampz *= -1
                        phasez += pi

                    phi1[q] = phasez
                    A1[q] = ampz


                # broadcast amplitudes and phases so that "net" change can be
                # calculated
                phi1s = COMM.gather(phi1, root=0)
                phi0s = COMM.gather(phi0, root=0)
                A1s   = COMM.gather(A1, root=0)
                A0s   = COMM.gather(A0, root=0)

                if(NOT_PARALLEL):
                    phi0s = np.concatenate(phi0s)
                    phi1s = np.concatenate(phi1s)
                    A0s = np.concatenate(A0s)
                    A1s = np.concatenate(A1s)

                    A_change = np.linalg.norm(A1s-A0s)/np.linalg.norm(A0s)
                    phi_change = np.linalg.norm(phi1s-phi0s)/np.linalg.norm(phi0s)
                else:
                    A_change = 1
                    phi_change = 1

                A_change = COMM.bcast(A_change, root=0)
                phi_change = COMM.bcast(phi_change, root=0)

                if(NOT_PARALLEL and self.verbose > 1 and n > 2*Tn):
                    print('time step: {0: <8d} Delta A: {1: <12.4e} ' \
                          'Delta Phi: {2: <12.4e}'.format(n, A_change, phi_change))

                p = 0
            else:
                p += 1

            n += 1

        libFDTD.FDTD_TE_capture_t0_fields(self._libfdtd)

        # perform a couple more iterations to get a second time point
        n0 = n
        for n in range(Tn):
            libFDTD.FDTD_TE_update_H(self._libfdtd, n+n0, (n+n0)*dt)

            # Note: da.localToLocal seems to have the same performance?
            self._gc.update_local_vector(self._Hx)
            self._gc.update_local_vector(self._Hy)

            libFDTD.FDTD_TE_update_E(self._libfdtd, n+n0, (n+n0+0.5)*dt)

            self._gc.update_local_vector(self._Ez)


        libFDTD.FDTD_TE_capture_t1_fields(self._libfdtd)

        for n in range(Tn):
            libFDTD.FDTD_TE_update_H(self._libfdtd, n+n0+Tn, (n+n0+Tn)*dt)

            # Note: da.localToLocal seems to have the same performance?
            self._gc.update_local_vector(self._Hx)
            self._gc.update_local_vector(self._Hy)

            libFDTD.FDTD_TE_update_E(self._libfdtd, n+n0+Tn, (n+n0+Tn+0.5)*dt)

            self._gc.update_local_vector(self._Ez)

        t0 = n0*dt
        t1 = (n0+Tn)*dt
        t2 = (n0+2*Tn)*dt
        libFDTD.FDTD_TE_calc_complex_fields_3T(self._libfdtd, t0, t1, t2)

    def solve_forward(self):
        """Run a forward simulation.

        A forward simulation is just a solution to Maxwell's equations.
        """
        if(self.verbose > 0 and NOT_PARALLEL):
            info_message('Solving forward simulation.')

        # Reset fourier-domain fields
        self._Ez_fwd_t0.set(0)
        self._Hx_fwd_t0.set(0); self._Hy_fwd_t0.set(0);

        self._Ez_fwd_t1.set(0)
        self._Hx_fwd_t1.set(0); self._Hy_fwd_t1.set(0);

        # make sure we are recording forward fields
        libFDTD.FDTD_TE_set_t0_arrays(self._libfdtd,
                                   self._Ez_fwd_t0.getArray(),
                                   self._Hx_fwd_t0.getArray(),
                                   self._Hy_fwd_t0.getArray())

        libFDTD.FDTD_TE_set_t1_arrays(self._libfdtd,
                                   self._Ez_fwd_t1.getArray(),
                                   self._Hx_fwd_t1.getArray(),
                                   self._Hy_fwd_t1.getArray())

        # set the forward simulation sources
        libFDTD.FDTD_TE_clear_sources(self._libfdtd)
        for src in self._sources:
            libFDTD.FDTD_TE_add_source(self._libfdtd,
                                    src.Jz,
                                    src.Mx, src.My,
                                    src.j0, src.k0,
                                    src.J, src.K,
                                    False)
        self.__solve()


        self.update_saved_fields()

        # calculate source power
        Psrc = self.get_source_power()
        if(NOT_PARALLEL):
            self._source_power = Psrc
        else:
            self._source_power = MathDummy()

    def solve_adjoint(self):
        """Run an adjoint simulation.

        In FDTD, this is a solution to Maxwell's equations but using a
        different set of sources than the forward simulation.
        """
        if(self.verbose > 0 and NOT_PARALLEL):
            info_message('Solving adjoint simulation...')

        # Reset fourier-domain fields
        self._Ez_adj_t0.set(0)
        self._Hx_adj_t0.set(0); self._Hy_adj_t0.set(0);

        self._Ez_adj_t1.set(0)
        self._Hx_adj_t1.set(0); self._Hy_adj_t1.set(0);

        # make sure we are recording adjoint fields
        libFDTD.FDTD_TE_set_t0_arrays(self._libfdtd,
                                    self._Ez_adj_t0.getArray(),
                                    self._Hx_adj_t0.getArray(),
                                    self._Hy_adj_t0.getArray())

        libFDTD.FDTD_TE_set_t1_arrays(self._libfdtd,
                                    self._Ez_adj_t1.getArray(),
                                    self._Hx_adj_t1.getArray(),
                                    self._Hy_adj_t1.getArray())

        # set the adjoint simulation sources
        # The phase of these sources has already been calculated,
        # so we tell the C++ library to skip the phase calculation
        libFDTD.FDTD_TE_clear_sources(self._libfdtd)
        for src in self._adj_sources:
            libFDTD.FDTD_TE_add_source(self._libfdtd,
                                    src.Jz,
                                    src.Mx, src.My,
                                    src.j0, src.k0,
                                    src.J, src.K,
                                    False)

        self.__solve()

    def update_saved_fields(self):
        """Update the fields contained in the regions specified by
        self.field_domains.

        This function is called internally by solve_forward and should not need
        to be called otherwise.
        """
        # clean up old fields
        self._saved_fields = []

        for domain in self._field_domains:
            Ez = self.get_field_interp(FieldComponent.Ez, domain)
            Hx = self.get_field_interp(FieldComponent.Hx, domain)
            Hy = self.get_field_interp(FieldComponent.Hy, domain)

            self._saved_fields.append((Ez, Hx, Hy))

    def __get_field(self, component, domain=None, adjoint=False):
        ##Get the uninterpolated field component in the specified domain.
        # The process is nearly identical for forward/adjoint
        if(domain == None):
            #domain = DomainCoordinates2D(0, self._X, 0, self._Y,
            #                           self._dx, self._dy)
            domain = DomainCoordinates(0, self._X, 0, self._Y, 0, 0,
                                       self._dx, self._dy, 1.0)

        if(component == FieldComponent.Ez):
            if(adjoint): field = self._Ez_adj_t0
            else: field = self._Ez_fwd_t0
        elif(component == FieldComponent.Hx):
            if(adjoint): field = self._Hx_adj_t0
            else: field = self._Hx_fwd_t0
        elif(component == FieldComponent.Hy):
            if(adjoint): field = self._Hy_adj_t0
            else: field = self._Hy_fwd_t0

        # get a "natural" representation of the appropriate field vector,
        # gather it on the rank 0 node and return the appropriate piece
        self._da.globalToNatural(field, self._vn)
        scatter, fout = PETSc.Scatter.toZero(self._vn)
        scatter.scatter(self._vn, fout, False, PETSc.Scatter.Mode.FORWARD)

        if(NOT_PARALLEL):
            fout = np.array(fout, dtype=np.complex128)
            fout = np.reshape(fout, [self._Ny, self._Nx])
            return fout[domain.j, domain.k]
        else:
            return MathDummy()

    def get_field(self, component, domain=None):
        """Get the (raw, uninterpolated) field.

        In most cases, you should use :def:`get_field_interp` instead.

        Parameters
        ----------
        component : str
            The field component to retrieve
        domain : misc.DomainCoordinates2D (optional)
            The domain in which the field is retrieved. If None, retrieve the
            field in the whole 3D domain (which you probably should avoid doing
            for larger problems). (default = None)

        Returns
        -------
        numpy.ndarray
            An array containing the field.
        """
        return self.__get_field(component, domain, adjoint=False)

    def get_adjoint_field(self, component, domain=None):
        """Get the adjoint field.

        Parameters
        ----------
        component : str
            The adjoint field component to retrieve.
        domain : misc.DomainCoordinates2D (optional)
            The domain in which the field is retrieved. If None, retrieve the
            field in the whole 3D domain (which you probably should avoid doing
            for larger problems). (default = None)

        Returns
        -------
        numpy.ndarray
            An array containing the field.
        """
        return self.__get_field(component, domain, adjoint=True)

    def get_field_interp(self, component, domain=None, squeeze=False):
        """Get the desired field component.

        Internally, fields are solved on a staggered grid. In most cases, it is
        desirable to know all of the field components at the same sets of
        positions. This requires that we interpolate the fields onto a single
        grid. In emopt, we interpolate all field components onto the Ez grid.

        Parameters
        ----------
        component : str
            The desired field component.
        domain : misc.DomainCoordinates2D (optional)
            The domain from which the field is retrieved. (default = None)

        Returns
        -------
        numpy.ndarray
            The interpolated field
        """
        # Ez does not need to be interpolated
        if(component == FieldComponent.Ez):
            if(squeeze): return np.squeeze(self.get_field(component, domain))
            else: return self.get_field(component, domain)
        else:
            # if no domain was provided
            if(domain == None):
                #domain_interp = DomainCoordinates2D(0, self._X, 0, self._Y,
                #                                  self._dx, self._dy)
                domain_interp = DomainCoordinates(0, self._X, 0, self._Y, 0, 0,
                                                  self._dx, self._dy, 1.0)
                domain = domain_interp

                k1 = domain_interp.k1; k2 = domain_interp.k2
                j1 = domain_interp.j1; j2 = domain_interp.j2

            # in order to properly handle interpolation at the boundaries, we
            # need to expand the domain
            else:
                k1 = domain.k1; k2 = domain.k2
                j1 = domain.j1; j2 = domain.j2

                if(k1 > 0): k1 -= 1
                if(k2 < self._Nx-1): k2 += 1
                if(j1 > 0): j1 -= 1
                if(j2 < self._Ny-1): j2 += 1

                #domain_interp = DomainCoordinates2D(k1*self._dx, k2*self._dx,
                #                                  j1*self._dy, j2*self._dy,
                #                                  self._dx, self._dy)
                domain_interp = DomainCoordinates(k1*self._dx, k2*self._dx,
                                                  j1*self._dy, j2*self._dy,
                                                  0, 0,
                                                  self._dx, self._dy, 1.0)

                k1 = domain_interp.k1; k2 = domain_interp.k2
                j1 = domain_interp.j1; j2 = domain_interp.j2

            fraw = self.get_field(component, domain_interp)

            if(RANK != 0):
                return MathDummy()

            fraw = np.pad(fraw, 1, 'constant', constant_values=0)

            # after interpolation, we will need to crop the field so that it
            # matches the supplied domain
            crop_field = lambda f : f[1+domain.j1-j1:-1-(j2-domain.j2), \
                                      1+domain.k1-k1:-1-(k2-domain.k2)]

            field = None
            bc = self._bc

            if(component == FieldComponent.Hx):
                # handle special boundary conditions
                if(j1 == 0 and bc[1] == 'E'):
                    fraw[0, :] = -1*fraw[1, :]
                elif(j1 == 0 and bc[1] == 'H'):
                    fraw[0, :] = fraw[1, :]

                Hx = np.copy(fraw)
                Hx[1:, :] += fraw[0:-1, :]
                Hx = Hx/2.0
                field = crop_field(Hx)

            elif(component == FieldComponent.Hy):
                # Handle special boundary conditions
                if(k1 == 0 and bc[0] == 'E'):
                    fraw[:, 0] = -1*fraw[:,1]
                elif(k1 == 0 and bc[0] == 'H'):
                    fraw[:, 0] = fraw[:, 1]

                Hy = np.copy(fraw)
                Hy[:, 1:] += fraw[:, 0:-1]
                Hy = Hy/2.0
                field = crop_field(Hy)

            else:
                pass

            if(squeeze): return np.squeeze(field)
            else: return field

    def get_source_power(self):
        """Get source power.

        The source power is the total electromagnetic power radiated by the
        electric and magnetic current sources.

        Returns
        -------
        float
            The source power.
        """
        Psrc = 0.0

        # define pml boundary domains
        dx = self._dx; dy = self._dy;
        if(self._w_pml[0] > 0): xmin = self._w_pml[0]+dx
        else: xmin = 0.0

        if(self._w_pml[1] > 0): xmax = self._X - self._w_pml[1]-dx
        else: xmax = self._X - self._dx

        if(self._w_pml[2] > 0): ymin = self._w_pml[2]+dy
        else: ymin = 0.0

        if(self._w_pml[3] > 0): ymax = self._Y - self._w_pml[3]-dy
        else: ymax = self._Y - self._dy


        #x1 = DomainCoordinates2D(xmin, xmin, ymin, ymax, dx, dy)
        #x2 = DomainCoordinates2D(xmax, xmax, ymin, ymax, dx, dy)
        #y1 = DomainCoordinates2D(xmin, xmax, ymin, ymin, dx, dy)
        #y2 = DomainCoordinates2D(xmin, xmax, ymax, ymax, dx, dy)
        x1 = DomainCoordinates(xmin, xmin, ymin, ymax, 0, 0, dx, dy, 1.0)
        x2 = DomainCoordinates(xmax, xmax, ymin, ymax, 0, 0, dx, dy, 1.0)
        y1 = DomainCoordinates(xmin, xmax, ymin, ymin, 0, 0, dx, dy, 1.0)
        y2 = DomainCoordinates(xmin, xmax, ymax, ymax, 0, 0, dx, dy, 1.0)

        # calculate power transmitter through xmin boundary
        Ez = self.get_field_interp('Ez', x1)
        Hy = self.get_field_interp('Hy', x1)

        if(NOT_PARALLEL and self._bc[0] != 'E' and self._bc[0] != 'H'):
            Px = -0.5*dy*np.sum(np.real(-1*Ez*np.conj(Hy)))
            #print Px
            Psrc += Px
        del Ez; del Hy;

        # calculate power transmitter through xmax boundary
        Ez = self.get_field_interp('Ez', x2)
        Hy = self.get_field_interp('Hy', x2)

        if(NOT_PARALLEL):
            Px = 0.5*dy*np.sum(np.real(-1*Ez*np.conj(Hy)))
            #print Px
            Psrc += Px
        del Ez; del Hy;

        # calculate power transmitter through ymin boundary
        Ez = self.get_field_interp('Ez', y1)
        Hx = self.get_field_interp('Hx', y1)

        if(NOT_PARALLEL and self._bc[1] != 'E' and self._bc[1] != 'H'):
            Py = 0.5*dx*np.sum(np.real(-1*Ez*np.conj(Hx)))
            #print Py
            Psrc += Py
        del Ez; del Hx;

        # calculate power transmitter through ymax boundary
        Ez = self.get_field_interp('Ez', y2)
        Hx = self.get_field_interp('Hx', y2)

        if(NOT_PARALLEL):
            Py = -0.5*dx*np.sum(np.real(-1*Ez*np.conj(Hx)))
            #print Py
            Psrc += Py
        del Ez; del Hx;

        return Psrc

    def get_A_diag(self):
        """Get a representation of the diagonal of the discretized Maxwell's
        equations assuming they are assembled in a matrix in the frequency
        domain.

        For the purposes of this implementation, this is just a copy of the
        permittivity and permeability distribution. In reality, there should be
        a factor of 1j and -1j for the permittivities and permeabilities,
        respectively, however we handle these prefactors elsewhere.

        Returns
        -------
        tuple of numpy.ndarrays
            A copy of the set of permittivity and permeability distributions used
            internally.
        """
        eps_z = np.copy(self._eps_z.getArray())
        mu_x = np.copy(self._mu_x.getArray())
        mu_y = np.copy(self._mu_y.getArray())

        return (eps_z, mu_x, mu_y)

    def calc_ydAx(self, Adiag0):
        """Calculate the product y^T*dA*x.

        Parameters
        ----------
        Adiag0 : tuple of 6 numpy.ndarray
            The 'initial' diag[A] obtained from self.get_A_diag()

        Returns
        -------
        complex
            The product y^T*dA*x
        """
        eps_z0, mu_x0, mu_y0 = Adiag0

        ydAx = np.zeros(eps_z0.shape, dtype=np.complex128)
        ydAx = ydAx + self._Ez_adj_t0[...] *  1j * (self._eps_z[...]-eps_z0) * self._Ez_fwd_t0[...]
        ydAx = ydAx + self._Hx_adj_t0[...] * -1j * (self._mu_x[...]-mu_x0)   * self._Hx_fwd_t0[...]
        ydAx = ydAx + self._Hy_adj_t0[...] * -1j * (self._mu_y[...]-mu_y0)   * self._Hy_fwd_t0[...]

        return np.sum(ydAx)

    @run_on_master
    def test_src_func(self):
        import matplotlib.pyplot as plt

        time = np.arange(0,3000,1)*self._dt

        ramp = np.zeros(3000)
        for i in range(len(time)):
            ramp[i] = libFDTD.FDTD_TE_src_func_t(self._libfdtd, i, time[i], 0)

        plt.plot(time,ramp)
        plt.show()


class FDTD_TM(FDTD_TE):
    def __init__(self, X, Y, dx, dy, wavelength, rtol=1e-6, nconv=None,
                 min_rindex=1.0, complex_eps=False):
        super(FDTD_TM, self).__init__(X, Y, dx, dy, wavelength, rtol=rtol,
              nconv=nconv, min_rindex=min_rindex, complex_eps=complex_eps)
        #self._bc = ['M','M']

    @property
    def eps(self):
        return self._eps_actual

    @property
    def mu(self):
        return self._mu_actual

    @property
    def bc(self):
        retval = []
        for i in range(2):
            if(self._bc[i]=='E'): retval.append('H')
            elif(self._bc[i]=='H'): retval.append('E')
        return ''.join(retval)

    @bc.setter
    def bc(self, newbc):
        if(len(newbc) != 2):
            raise ValueError('Incorrect number of boundary conditions specified!')

        for bc in newbc:
            if(not bc in '0EHP'):
                raise ValueError('Unrecognized boundary condition: %s' % (bc))
            if(bc == 'P'):
                raise NotImplementedError('Periodic boundary conditions not yet ' \
                                          'implemented')


        newbc = list(newbc)
        for i in range(2):
            if(newbc[i] == 'E'): newbc[i] = 'H'
            elif(newbc[i] == 'H'): newbc[i] = 'E'
            #if(self._bc[i] == '0'): self._bc[i] = 'M'
            #elif(self._bc[i] == 'M'): self._bc[i] = '0'


        #self._bc = list(newbc)
        self._bc = newbc
        libFDTD.FDTD_TE_set_bc(self._libfdtd, ''.join(self._bc).encode('ascii'))

    def set_materials(self, eps, mu):
        super(FDTD_TM, self).set_materials(mu, eps)
        self._eps_actual = eps
        self._mu_actual = mu

    def __get_local_domain_overlap(self, domain):
        super(FDTD_TM, self).__get_local_domain_overlap(domain)

    #def __set_sources(self, src, domain, adjoint=False):
    #    super(FDTD_TM, self).__set_sources(src, domain, adjoint=adjoint)

    def set_sources(self, src, domain, mindex=0):
        """Set a simulation source.

        Simulation sources can be set either using a set of 6 arrays (Jx, Jy,
        Jz, Mx, My, Mz) or a :class:`modes.ModeFullVector` object. In either
        case, a domain must be provided which tells the simulation where to put
        those sources.

        This function operates in an additive manner meaning that multiple
        calls will add multiple sources to the simulation. To replace the
        existing sources, simply call :def:`clear_sources` first.

        Parameters
        ----------
        src : tuple or modes.ModeFullVector
            The source arrays or mode object containing source data
        domain : misc.DomainCoordinates2D
            The domain which specifies where the source is located
        mindex : int (optional)
            The mode source index. This is only relevant if using a
            ModeFullVector object to set the sources. (default = 0)
        """
        if(type(src) == ModeTM):
            Mzs, Jxs, Jys = src.get_source(mindex, self._dx,
                                                   self._dy)

            Mz = COMM.bcast(Mzs, root=0)
            Jx = COMM.bcast(Jxs, root=0)
            Jy = COMM.bcast(Jys, root=0)

            src_arrays = (Mz, -1*Jx, -1*Jy)
        else:
            Mz = src[0]
            Jx = src[1]
            Jy = src[2]
            src_arrays = (Mz, -1*Jx, -1*Jy)

        #self.__set_sources(src_arrays, domain, adjoint=False)
        super(FDTD_TM, self).set_sources(src_arrays, domain, mindex)
        COMM.Barrier()

    def set_adjoint_sources(self, src):
        super(FDTD_TM, self).set_adjoint_sources((src[0], -1*src[1], -1*src[2]))

    def get_field(self, component, domain=None):
        """Get the (raw, uninterpolated) field.

        In most cases, you should use :def:`get_field_interp` instead.

        Parameters
        ----------
        component : str
            The field component to retrieve
        domain : misc.DomainCoordinates2D (optional)
            The domain in which the field is retrieved. If None, retrieve the
            field in the whole 3D domain (which you probably should avoid doing
            for larger problems). (default = None)

        Returns
        -------
        numpy.ndarray
            An array containing the field.
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = component

        field = super(FDTD_TM, self).get_field(te_comp, domain)

        if(component == 'Hz'):
            return -1 * field
        else:
            return field

    def get_adjoint_field(self, component, domain=None):
        """Get the adjoint field.

        Parameters
        ----------
        component : str
            The adjoint field component to retrieve.
        domain : misc.DomainCoordinates2D (optional)
            The domain in which the field is retrieved. If None, retrieve the
            field in the whole 3D domain (which you probably should avoid doing
            for larger problems). (default = None)

        Returns
        -------
        numpy.ndarray
            An array containing the field.
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = component

        field = super(FDTD_TM, self).get_adjoint_field(te_comp, domain)

        if(component == 'Hz'):
            return -1 * field
        else:
            return field

    def get_field_interp(self, component, domain=None, squeeze=False):
        """Get the desired field component.

        Internally, fields are solved on a staggered grid. In most cases, it is
        desirable to know all of the field components at the same sets of
        positions. This requires that we interpolate the fields onto a single
        grid. In emopt, we interpolate all field components onto the Ez grid.

        Parameters
        ----------
        component : str
            The desired field component.
        domain : misc.DomainCoordinates2D (optional)
            The domain from which the field is retrieved. (default = None)

        Returns
        -------
        numpy.ndarray
            The interpolated field
        """
        te_comp = ''
        if(component == 'Hz'): te_comp = 'Ez'
        elif(component == 'Ex'): te_comp = 'Hx'
        elif(component == 'Ey'): te_comp = 'Hy'
        else: te_comp = component

        field = super(FDTD_TM, self).get_field_interp(te_comp, domain, squeeze)

        if(component == 'Hz'):
            return -1 * field
        else:
            return field

    def get_source_power(self):
        """Get source power.

        The source power is the total electromagnetic power radiated by the
        electric and magnetic current sources.

        Returns
        -------
        float
            The source power.
        """
        Psrc = 0.0

        # define pml boundary domains
        dx = self._dx; dy = self._dy;
        if(self._w_pml[0] > 0): xmin = self._w_pml[0]+dx
        else: xmin = 0.0

        if(self._w_pml[1] > 0): xmax = self._X - self._w_pml[1]-dx
        else: xmax = self._X - self._dx

        if(self._w_pml[2] > 0): ymin = self._w_pml[2]+dy
        else: ymin = 0.0

        if(self._w_pml[3] > 0): ymax = self._Y - self._w_pml[3]-dy
        else: ymax = self._Y - self._dy


        x1 = DomainCoordinates(xmin, xmin, ymin, ymax, 0, 0, dx, dy, 1.0)
        x2 = DomainCoordinates(xmax, xmax, ymin, ymax, 0, 0, dx, dy, 1.0)
        y1 = DomainCoordinates(xmin, xmax, ymin, ymin, 0, 0, dx, dy, 1.0)
        y2 = DomainCoordinates(xmin, xmax, ymax, ymax, 0, 0, dx, dy, 1.0)

        # calculate power transmitter through xmin boundary
        Hz = self.get_field_interp('Hz', x1)
        Ey = self.get_field_interp('Ey', x1)

        if(NOT_PARALLEL and self._bc[0] != 'E' and self._bc[0] != 'H'):
            Px = 0.5*dy*np.sum(np.real(-1*Ey*np.conj(Hz)))
            Psrc += Px
        del Hz; del Ey;

        # calculate power transmitter through xmax boundary
        Hz = self.get_field_interp('Hz', x2)
        Ey = self.get_field_interp('Ey', x2)

        if(NOT_PARALLEL):
            Px = -0.5*dy*np.sum(np.real(-1*Ey*np.conj(Hz)))
            Psrc += Px
        del Hz; del Ey;

        # calculate power transmitter through ymin boundary
        Hz = self.get_field_interp('Hz', y1)
        Ex = self.get_field_interp('Ex', y1)

        if(NOT_PARALLEL and self._bc[1] != 'E' and self._bc[1] != 'H'):
            Py = -0.5*dx*np.sum(np.real(-1*Ex*np.conj(Hz)))
            Psrc += Py
        del Hz; del Ex;

        # calculate power transmitter through ymax boundary
        Hz = self.get_field_interp('Hz', y2)
        Ex = self.get_field_interp('Ex', y2)

        if(NOT_PARALLEL):
            Py = 0.5*dx*np.sum(np.real(-1*Ex*np.conj(Hz)))
            Psrc += Py
        del Hz; del Ex;

        return Psrc

    def get_A_diag(self):
        """Get the diagonal entries of the system matrix A.

        Parameters
        ----------
        vdiag : petsc4py.PETSc.Vec
            Vector with dimensions Mx1 where M is equal to the number of
            diagonal entries in A.

        Returns
        -------
        **(Master node only)** the diagonal entries of A.
        """
        # We need to override this function since the TE matrix diagonals do
        # not match the TM matrix diagonals (even when swapping eps and mu).
        # This is because the signs on epsilon and mu in Maxwell's equations
        # are flipped when moving from TE to TM.  In most cases, it is easiest
        # to handle this change by swapping Ez with -Hz, Mx with -Jx, and My
        # with -Jy in the TE equations, which can be achieved by simply
        # overriding the corresponding setter and getter functions.  In
        # reality, a better way to handle reusing the TE equations is to swap E
        # and H, J and M, eps with -mu, and mu with -eps.  This way of doing
        # things, however, is harder to achieve programmatically if we want to
        # reuse as much of the TE code as possible.  When using the FDFD object
        # with an AdjointMethod, it turns out that simply swapping field and
        # source components is insufficient and knowledge of the A's diagonals
        # is needed, hence this overriden function.
        mu_z, eps_x, eps_y = super(FDTD_TM, self).get_A_diag()
        
        return (-1*mu_z, -1*eps_x, -1*eps_y)

class GhostComm(object):

    def __init__(self, k0, j0, K, J, Nx, Ny):
        """Create a GhostComm object.

        The GhostComm object mediates the communication of ghost (edge) values
        in shared vectors which store values on a divided rectangular grid.

        Parameters
        ----------
        k0 : int
            The starting x grid index of a block of the grid.
        j0 : int
            The starting y grid index of a block of the grid.
        i0 : int
            The starting z grid index of a block of the grid.
        K : int
            The number of grid points along x of the block.
        J : int
            The number of grid points along y of the block.
        I : int
            The number of grid points along z of the block.

        """
        # Inputs describe the block in the grid which has ghost values
        self._j0 = j0
        self._k0 = k0
        self._J = J
        self._K = K
        self._Nx = Nx
        self._Ny = Ny

        # define the number of boundary points and ghost points
        # for simplicity, the edge values of  the boundary elements are
        # duplicated. This should minimally impact performance.
        nbound = K*2 + J*2
        nghost = nbound
        self._nbound = nbound
        self._nghsot = nghost

        # Create the vector that will store edge and ghost values
        gvec = PETSc.Vec().createMPI((nbound, None))
        self._gvec = gvec

        # determine the global indices for ghost values. To do this, each
        # process needs to know (1) the starting index of each block in the
        # global ghost vector, (2) the position of each block in the physical
        # grid, and (3) the size of each block.
        start, end = gvec.getOwnershipRange()
        pos_data = (start, j0, k0, J, K)

        # collect and then share all of the position data
        pos_data = COMM.gather(pos_data, root=0)
        pos_data = COMM.bcast(pos_data, root=0)

        ighosts = []

        # Find the global indices of the different boundaries
        # xmin boundary
        k = k0-1
        if(k < 0): k = Nx-1
        for pd in pos_data:
            gindex, j0n, k0n, Jn, Kn = pd
            if(k == k0n + Kn-1 and j0 == j0n):
                ighosts += list(range(gindex+Jn, gindex+2*Jn))

        # xmax boundary
        k = k0+K
        if(k > Nx-1): k = 0
        for pd in pos_data:
            gindex, j0n, k0n, Jn, Kn = pd
            if(k ==  k0n and j0 == j0n):
                ighosts += list(range(gindex, gindex + Jn))

        # ymin boundary
        j = j0-1
        if(j < 0): j = Ny-1
        for pd in pos_data:
            gindex, j0n, k0n, Jn, Kn = pd
            if(j == j0n + Jn - 1 and k0 == k0n):
                ighosts += list(range(gindex+2*Jn+Kn, gindex+2*Jn+2*Kn))

        # ymax boundary
        j = j0+J
        if(j > Ny-1): j = 0
        for pd in pos_data:
            gindex, j0n, k0n, Jn, Kn = pd
            if(j == j0n and k0 == k0n):
                ighosts += list(range(gindex+2*Jn, gindex+2*Jn+Kn))


        # Set the ghost indices = finish constructing ghost vector
        self._ighosts = np.array(ighosts, dtype=np.int32)
        ghosts = PETSc.IS().createGeneral(self._ighosts)
        self._gvec.setMPIGhost(ghosts)

    def update_local_vector(self, vl):
        """Update the ghost values of a local vector.

        Parameters
        ----------
        vl : np.ndarray
            The local vector
        """
        J = self._J; K = self._K
        j0 = self._j0; k0 = self._k0
        nbound = self._nbound

        garr = self._gvec.getArray()

        libFDTD.FDTD_TE_copy_to_ghost_comm(vl, garr, J, K)

        # do the ghost update
        self._gvec.ghostUpdate()

        # copy the distributed ghost values to the correct positions
        nstart = nbound
        with self._gvec.localForm() as gloc:
            gloc_arr = gloc.getArray()

            libFDTD.FDTD_TE_copy_from_ghost_comm(vl, garr, J, K)


