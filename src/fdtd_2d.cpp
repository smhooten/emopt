#include "fdtd_2d.hpp"
#include <math.h>
#include <algorithm>

fdtd_2d::FDTD_TE::FDTD_TE() 
{
    // make sure all of our PML arrays start NULL
    _pml_Ezx0 = NULL; _pml_Ezx1 = NULL; _pml_Ezy0 = NULL; _pml_Ezy1 = NULL;
    _pml_Hxy0 = NULL; _pml_Hxy1 = NULL; 
    _pml_Hyx0 = NULL; _pml_Hyx1 = NULL; 
    
    _kappa_H_x = NULL; _kappa_H_y = NULL;
    _kappa_E_x = NULL; _kappa_E_y = NULL;

    _bHx = NULL; _bHy = NULL; 
    _bEx = NULL; _bEy = NULL; 

    _cHx = NULL; _cHy = NULL; 
    _cEx = NULL; _cEy = NULL; 

    _w_pml_x0 = 0; _w_pml_x1 = 0;
    _w_pml_y0 = 0; _w_pml_y1 = 0;

    _complex_eps = false;
}

fdtd_2d::FDTD_TE::~FDTD_TE()
{
    // Clean up PML arrays
    delete[] _pml_Ezx0; delete[] _pml_Ezx1; delete[] _pml_Ezy0; delete[] _pml_Ezy1;
    delete[] _pml_Hxy0; delete[] _pml_Hxy1;
    delete[] _pml_Hyx0; delete[] _pml_Hyx1;

    delete [] _kappa_H_x;
    delete [] _kappa_H_y;

    delete [] _kappa_E_x;
    delete [] _kappa_E_y;

    delete [] _bHx;
    delete [] _bHy;

    delete [] _bEx;
    delete [] _bEy;

    delete [] _cHx;
    delete [] _cHy;

    delete [] _cEx;
    delete [] _cEy;
}

void fdtd_2d::FDTD_TE::set_physical_dims(double X, double Y,
                                         double dx, double dy)
{
    _X = X; _Y = Y;
    _dx = dx; _dy = dy;
}

void fdtd_2d::FDTD_TE::set_grid_dims(int Nx, int Ny)
{
    _Nx = Nx;
    _Ny = Ny;
}


void fdtd_2d::FDTD_TE::set_local_grid(int k0, int j0, int K, int J)
{
     _j0 = j0; _k0 = k0;
     _J = J; _K = K;

}


void fdtd_2d::FDTD_TE::set_wavelength(double wavelength)
{
    _wavelength = wavelength;
    _R = _wavelength/(2*M_PI);
}


void fdtd_2d::FDTD_TE::set_dt(double dt)
{
    _dt = dt;
    _odt = 1.0/_dt;
}

void fdtd_2d::FDTD_TE::set_complex_eps(bool complex_eps)
{
    _complex_eps = complex_eps;
}

void fdtd_2d::FDTD_TE::set_field_arrays(double *Ez,
                                        double *Hx, double *Hy)
{
    _Ez = Ez;
    _Hx = Hx; _Hy = Hy;
}

void fdtd_2d::FDTD_TE::set_mat_arrays(complex128 *eps_z,
                                complex128 *mu_x, complex128 *mu_y)
{
    _eps_z = eps_z;
    _mu_x = mu_x; _mu_y = mu_y;
}

void fdtd_2d::FDTD_TE::update_H(int n, double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           b, C, kappa,
           src_t,
           dt_by_mux, dt_by_muy,

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,

    int ind_jk, ind_jp1k, ind_jkp1, ind_global,
        ind_pml, ind_src, j0s, k0s, Js, Ks,
        ind_pml_param;

    double dEzdx, dEzdy;
    complex128 *Mx, *My;

    // Setup the fields on the simulation boundary based on the boundary conditions
    if(_bc[0] != 'P' && _k0 + _K == _Nx){
        for(int j = 0; j < _J; j++) {
            ind_jk = (_J+2)*(_K+2) + (j+1)*(_K+2) + _K + 1;

            _Ez[ind_jk] = 0.0;
        }
    }

    if(_bc[1] != 'P' && _j0 + _J == _Ny){
        for(int k = 0; k < _K; k++) {
            ind_jk = (_J+2)*(_K+2) + (_J+1)*(_K+2) + k + 1;

            _Ez[ind_jk] = 0.0;
        }
    }

    for(int j = 0; j < _J; j++) {
        for(int k = 0; k < _K; k++) {
            ind_jk = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
            ind_jp1k = (_J+2)*(_K+2) + (j+2)*(_K+2) + k + 1;
            ind_jkp1 = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 2;

            ind_global = _J*_K + j*_K + k;

            // compute prefactors
            dt_by_mux = _dt/_mu_x[ind_global].real;
            dt_by_muy = _dt/_mu_y[ind_global].real;
    
            // Update Hx
            dEzdy = ody*(_Ez[ind_jp1k] - _Ez[ind_jk]);

            _Hx[ind_jk] = _Hx[ind_jk] +  dt_by_mux * 
                           (dEydz);

            // update Hy
            dEzdx = odx * (_Ez[ind_jkp1] - _Ez[ind_jk]);

            _Hy[ind_jk] = _Hy[ind_jk] + dt_by_muy *
                           (dEzdx);

            // Do PML updates
            if(k + _k0 < pml_xmin) {
                // get index in PML array
                ind_pml = _J*(pml_xmin - _k0) + j*(pml_xmin - _k0) + k;

                // get PML coefficients
                ind_pml_param = pml_xmin - k - _k0 - 1;
                kappa = _kappa_H_x[ind_pml_param];
                b = _bHx[ind_pml_param];
                C = _cHx[ind_pml_param];

                // Update PML convolution
                _pml_Ezx0[ind_pml] = C * dEzdx + b*_pml_Ezx0[ind_pml];

                _Hy[ind_jk] = _Hy[ind_jk] + dt_by_muy * (_pml_Ezx0[ind_pml]-dEzdx+dEzdx/kappa);
            
            }
            else if(k + _k0 >= pml_xmax) {
                ind_pml = _J*(_k0 + _K - pml_xmax) + j*(_k0 + _K - pml_xmax) + k + _k0 - pml_xmax;

                // get pml coefficients
                ind_pml_param = k+_k0 - pml_xmax + _w_pml_x0;
                kappa = _kappa_H_x[ind_pml_param];
                b = _bHx[ind_pml_param];
                C = _cHx[ind_pml_param];

                _pml_Ezx1[ind_pml] = C * dEzdx + b*_pml_Ezx1[ind_pml];

                _Hy[ind_jk] = _Hy[ind_jk] + dt_by_muy * (_pml_Ezx1[ind_pml]-dEzdx+dEzdx/kappa);
            }

            if(j + _j0 < pml_ymin) {
                ind_pml = (pml_ymin - _j0)*_K +j*_K + k;

                // compute coefficients
                ind_pml_param = pml_ymin - j - _j0 - 1;
                kappa = _kappa_H_y[ind_pml_param];
                b = _bHy[ind_pml_param];
                C = _cHy[ind_pml_param];

                _pml_Ezy0[ind_pml] = C * dEzdy + b*_pml_Ezy0[ind_pml];

                _Hx[ind_jk] = _Hx[ind_jk] - dt_by_mux * (_pml_Ezy0[ind_pml]-dEzdy+dEzdy/kappa);
            }
            else if(j + _j0 >= pml_ymax) {
                ind_pml = (_j0 + _J - pml_ymax)*_K +(_j0 + j - pml_ymax)*_K + k;

                // compute coefficients
                ind_pml_param = j+_j0 - pml_ymax + _w_pml_y0;
                kappa = _kappa_H_y[ind_pml_param];
                b = _bHy[ind_pml_param];
                C = _cHy[ind_pml_param];

                _pml_Ezy1[ind_pml] = C * dEzdy + b*_pml_Ezy1[ind_pml];

                _Hx[ind_jk] = _Hx[ind_jk] - dt_by_mux * (_pml_Ezy1[ind_pml]-dEzdy+dEzdy/kappa);
            }
        }
    }
    

    // Update sources
    for(auto const& src : _sources) {
        j0s = src.j0; Js = src.J;
        k0s = src.k0; Ks = src.K;

        // update Mx
        Mx = src.Mx;

        for(int j = 0; j < Js; j++) {
            for(int k = 0; k < Ks; k++) {
                ind_jk = (_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                ind_global = _J*_K + (j+j0s)*_K + k+k0s;
                ind_src = Js*Ks + j*Ks + k;
                
                src_t = src_func_t(n, t, Mx[ind_src].imag);
                _Hx[ind_jk] = _Hx[ind_jk] + src_t * Mx[ind_src].real * _dt / _mu_x[ind_global].real;                   
            }
        }

        // update My
        My = src.My;

        for(int j = 0; j < Js; j++) {
            for(int k = 0; k < Ks; k++) {
                ind_jk = (_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                ind_global = _J*_K + (j+j0s)*_K + k+k0s;
                ind_src = Js*Ks + j*Ks + k;
                
                src_t = src_func_t(n, t, My[ind_src].imag);
                _Hy[ind_jk] = _Hy[ind_jk] + src_t * My[ind_src].real * _dt / _mu_y[ind_global].real;                   
            }
        }
        
    }
}

void fdtd_2d::FDTD_TE::update_E(int n, double t)
{
    double odx = _R/_dx,
           ody = _R/_dy,
           b, C, kappa,
           src_t,
           a_z, b_z;

#ifdef COMPLEX_EPS
    double epszr_by_dt,
           epszi_by_2;
    complex128 epsz;
#endif

    int pml_xmin = _w_pml_x0, pml_xmax = _Nx-_w_pml_x1,
        pml_ymin = _w_pml_y0, pml_ymax = _Ny-_w_pml_y1,

    int ind_jk, ind_jm1k, ind_jkm1, ind_global,
        ind_pml, ind_src, j0s, k0s, Js, Ks,
        ind_pml_param;

    int ind_jkp1, ind_jp1k;  // used for setting boundary values

    double dHxdy, dHxdz, dHydx, dHydz;
    complex128 *Jz;

    // Setup the fields on the simulation boundary based on the boundary conditions
    if(_k0 == 0){
        if(_bc[0] == '0') {
            for(int j = 0; j < _J; j++) {
                ind_jk = (_J+2)*(_K+2) + (j+1)*(_K+2);

                _Hy[ind_jk] = 0.0;
            }
        }
        else if(_bc[0] == 'E') {
            for(int j = 0; j < _J; j++) {
                ind_jk = (_J+2)*(_K+2) + (j+1)*(_K+2);
                ind_jkp1 = (_J+2)*(_K+2) + (j+1)*(_K+2)+1;

                _Hy[ind_jk] = -1*_Hy[ind_jkp1];
            }
        }
        else if(_bc[0] == 'H') {
            for(int j = 0; j < _J; j++) {
                ind_jk = (_J+2)*(_K+2) + (j+1)*(_K+2);
                ind_jkp1 = (_J+2)*(_K+2) + (j+1)*(_K+2)+1;

                _Hy[ind_jk] = _Hy[ind_jkp1];
            }
        }
    }

    if(_j0 == 0){
        if(_bc[1] == '0') {
            for(int k = 0; k < _K; k++) {
                ind_jk = (_J+2)*(_K+2) + 0*(_K+2) + k + 1;

                _Hx[ind_jk] = 0.0;
            }
            
        }
        else if(_bc[1] == 'E') {
            for(int k = 0; k < _K; k++) {
                ind_jk = (_J+2)*(_K+2) + 0*(_K+2) + k + 1;
                ind_jp1k = (_J+2)*(_K+2) + 1*(_K+2) + k + 1;

                _Hx[ind_jk] = -1*_Hx[ind_jp1k];
            }
        }
        else if(_bc[1] == 'H') {
            for(int k = 0; k < _K; k++) {
                ind_jk = (_J+2)*(_K+2) + 0*(_K+2) + k + 1;
                ind_jp1k = (_J+2)*(_K+2) + 1*(_K+2) + k + 1;

                _Hx[ind_jk] = _Hx[ind_jp1k];
            }
        }
    }


    for(int j = 0; j < _J; j++) {
        for(int k = 0; k < _K; k++) {
            ind_jk = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
            ind_jm1k = (_J+2)*(_K+2) + (j)*(_K+2) + k + 1;
            ind_jkm1 = (_J+2)*(_K+2) + (j+1)*(_K+2) + k;

            ind_global = j*_K + k;

#ifdef COMPLEX_EPS
            // calculate permittivity quantities
            if(!_complex_eps) {
                a_z = 1.0;
                b_z = _dt/_eps_z[ind_global].real;
            }
            else {
                epsz = _eps_z[ind_global];

                epszr_by_dt = epsz.real*_odt;

                epszi_by_2 = epsz.imag*0.5;

                // The update equations have the following form:
                // E_{i,t+1} = a_i * E_{i,t} + curl(H)_{i,t} / b_x
                // where i is the spatial component, t is the time step,
                // and a_i and b_i are conefficients which depend on
                // the relative permittivities and time step
                a_z = (epszr_by_dt - epszi_by_2) / (epszr_by_dt + epszi_by_2);

                b_z = 1.0/(epszr_by_dt + epszi_by_2);
            }
#else
            a_z = 1.0;
            b_z = _dt/_eps_z[ind_global].real;
#endif

            // Update Ez
            dHydx = odx * (_Hy[ind_jk] - _Hy[ind_jkm1]);
            dHxdy = ody * (_Hx[ind_jk] - _Hx[ind_jm1k]);

            _Ez[ind_jk] = a_z * _Ez[ind_jk] + (dHydx - dHxdy) * b_z;

            
            // Do PML updates
            if(k + _k0 < pml_xmin) {
                ind_pml = _J*(pml_xmin-_k0) + j*(pml_xmin-_k0) + k;
                
                // get PML coefficients
                ind_pml_param = pml_xmin - k - _k0 - 1;
                kappa = _kappa_E_x[ind_pml_param];
                b = _bEx[ind_pml_param];
                C = _cEx[ind_pml_param];

                _pml_Hyx0[ind_pml] = C * dHydx + b*_pml_Hyx0[ind_pml];

                _Ez[ind_jk] = _Ez[ind_jk] + (_pml_Hyx0[ind_pml]-dHydx+dHydx/kappa) * b_z;

            }
            else if(k + _k0 >= pml_xmax) {
                ind_pml = j*(_k0 + _K - pml_xmax) + k + _k0 - pml_xmax;
                
                // get coefficients
                ind_pml_param = k+_k0 - pml_xmax + _w_pml_x0;
                kappa = _kappa_E_x[ind_pml_param];
                b = _bEx[ind_pml_param];
                C = _cEx[ind_pml_param];

                _pml_Hyx1[ind_pml] = C * dHydx + b*_pml_Hyx1[ind_pml];

                _Ez[ind_jk] = _Ez[ind_jk] + (_pml_Hyx1[ind_pml]-dHydx+dHydx/kappa) * b_z;
            }

            if(j + _j0 < pml_ymin) {
                ind_pml = (pml_ymin - _j0)*_K +j*_K + k;

                // get coefficients
                ind_pml_param = pml_ymin - j - _j0 - 1;
                kappa = _kappa_E_y[ind_pml_param];
                b = _bEy[ind_pml_param];
                C = _cEy[ind_pml_param];

                _pml_Hxy0[ind_pml] = C * dHxdy + b*_pml_Hxy0[ind_pml];

                _Ez[ind_jk] = _Ez[ind_jk] - (_pml_Hxy0[ind_pml]-dHxdy+dHxdy/kappa) * b_z;
            }
            else if(j + _j0 >= pml_ymax) {
                ind_pml = (_j0 + _J - pml_ymax)*_K +(_j0 + j - pml_ymax)*_K + k;

                // get coefficients
                ind_pml_param = j+_j0 - pml_ymax + _w_pml_y0;
                kappa = _kappa_E_y[ind_pml_param];
                b = _bEy[ind_pml_param];
                C = _cEy[ind_pml_param];    

                _pml_Hxy1[ind_pml] = C * dHxdy + b*_pml_Hxy1[ind_pml];

                _Ez[ind_jk] = _Ez[ind_jk] - (_pml_Hxy1[ind_pml]-dHxdy+dHxdy/kappa) * b_z;
            }

        }
    }

    // Update sources
    for(auto const& src : _sources) {
        j0s = src.j0; Js = src.J;
        k0s = src.k0; Ks = src.K;

        // update Jz
        Jz = src.Jz;

        for(int j = 0; j < Js; j++) {
            for(int k = 0; k < Ks; k++) {
                ind_jk = (_J+2)*(_K+2) + (j+j0s+1)*(_K+2) + k + k0s + 1;
                ind_global = _J*_K + (j+j0s)*_K + k+k0s;
                ind_src = j*Ks + k;
                
#ifdef COMPLEX_EPS
                if(!_complex_eps)
                    b_z = _dt/_eps_z[ind_global].real;
                else {
                    epsz = _eps_z[ind_global];
                    b_z = 1.0/(epsz.real * _odt + epsz.imag*0.5);
                }
#else
                b_z = _dt/_eps_z[ind_global].real;
#endif
                src_t = src_func_t(n, t, Jz[ind_src].imag);
                _Ez[ind_jk] = _Ez[ind_jk] - src_t * Jz[ind_src].real * b_z;                   
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// PML Management
///////////////////////////////////////////////////////////////////////////


void fdtd_2d::FDTD_TE::set_pml_widths(int xmin, int xmax, int ymin, int ymax)
{
    _w_pml_x0 = xmin; _w_pml_x1 = xmax;
    _w_pml_y0 = ymin; _w_pml_y1 = ymax;
}

void fdtd_2d::FDTD_TE::set_pml_properties(double sigma, double alpha, double kappa, double pow)
{
    _sigma = sigma;
    _alpha = alpha;
    _kappa = kappa;
    _pow   = pow;

    compute_pml_params();
}

void fdtd_2d::FDTD_TE::build_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _J * (xmin - _k0);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Ezx0; _pml_Ezx0 = NULL;
        _pml_Ezx0 = new double[N];

        delete [] _pml_Hyx0; _pml_Hyx0 = NULL;
        _pml_Hyx0 = new double[N];
    }

    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _J * (_k0  + _K - xmax);

        // Clean up old arrays and allocate new ones
        delete [] _pml_Ezx1; _pml_Ezx1 = NULL;
        _pml_Ezx1 = new double[N];

        delete [] _pml_Hyx1; _pml_Hyx1 = NULL;
        _pml_Hyx1 = new double[N];
    }

    // touches ymin boundary
    if(_j0 < ymin) {
        N = _K * (ymin - _j0);

        delete [] _pml_Ezy0; _pml_Ezy0 = NULL;
        _pml_Ezy0 = new double[N];

        delete [] _pml_Hxy0; _pml_Hxy0 = NULL;
        _pml_Hxy0 = new double[N];
    }

    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _K * (_j0 + _J - ymax);

        delete [] _pml_Ezy1; _pml_Ezy1 = NULL;
        _pml_Ezy1 = new double[N];

        delete [] _pml_Hxy1; _pml_Hxy1 = NULL;
        _pml_Hxy1 = new double[N];
    }


    // (re)compute the spatially-dependent PML parameters
    compute_pml_params();
}

void fdtd_2d::FDTD_TE::reset_pml()
{
    int N,
        xmin = _w_pml_x0, xmax = _Nx-_w_pml_x1,
        ymin = _w_pml_y0, ymax = _Ny-_w_pml_y1,

    // touches xmin boudary
    if(_k0 < xmin) {
        N = _J * (xmin - _k0);
        std::fill(_pml_Ezx0, _pml_Ezx0 + N, 0);
        std::fill(_pml_Hyx0, _pml_Hyx0 + N, 0);
    }

    // touches xmax boundary
    if(_k0 +_K > xmax) {
        N = _J * (_k0  + _K - xmax);

        std::fill(_pml_Ezx1, _pml_Ezx1 + N, 0);
        std::fill(_pml_Hyx1, _pml_Hyx1 + N, 0);
    }

    // touches ymin boundary
    if(_j0 < ymin) {
        N = _K * (ymin - _j0);

        std::fill(_pml_Ezy0, _pml_Ezy0 + N, 0);
        std::fill(_pml_Hxy0, _pml_Hxy0 + N, 0);
    }

    // touches ymax boundary
    if(_j0 + _J > ymax) {
        N = _K * (_j0 + _J - ymax);

        std::fill(_pml_Ezy1, _pml_Ezy1 + N, 0);
        std::fill(_pml_Hxy1, _pml_Hxy1 + N, 0);
    }


}

void fdtd_2d::FDTD_TE::compute_pml_params()
{
    double pml_dist, pml_factor, sigma, alpha, kappa, b, c;

    // clean up the previous arrays and allocate new ones
    delete [] _kappa_H_x; _kappa_H_x = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _kappa_H_y; _kappa_H_y = new double[_w_pml_y0 + _w_pml_y1];

    delete [] _kappa_E_x; _kappa_E_x = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _kappa_E_y; _kappa_E_y = new double[_w_pml_y0 + _w_pml_y1];

    delete [] _bHx; _bHx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _bHy; _bHy = new double[_w_pml_y0 + _w_pml_y1];

    delete [] _bEx; _bEx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _bEy; _bEy = new double[_w_pml_y0 + _w_pml_y1];

    delete [] _cHx; _cHx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _cHy; _cHy = new double[_w_pml_y0 + _w_pml_y1];

    delete [] _cEx; _cEx = new double[_w_pml_x0 + _w_pml_x1];
    delete [] _cEy; _cEy = new double[_w_pml_y0 + _w_pml_y1];

    // calculate the PML parameters. These parameters are all functions of
    // the distance from the ONSET of the PML edge (which begins in the simulation
    // domain interior.
    // Note: PML parameters are ordered such that distance from PML onset
    // always increases with index.
    
    // setup xmin PML parameters
    for(int k = 0; k < _w_pml_x0; k++) {
        pml_dist = double(k - 0.5)/_w_pml_x0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        // compute H coefficients
        sigma = _sigma * pml_factor;
        alpha = _alpha * (1-pml_factor);
        kappa = (_kappa-1.0) * pml_factor+1.0;
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_x[k] = kappa;
        _bHx[k] = b;
        _cHx[k] = c;

        pml_dist = double(k)/_w_pml_x0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        // compute E coefficients
        sigma = _sigma * pml_factor;
        alpha = _alpha * (1-pml_factor);
        kappa = (_kappa-1) * pml_factor+1;
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_x[k] = kappa;
        _bEx[k] = b;
        _cEx[k] = c;

    }
    for(int k = 0; k < _w_pml_x1; k++) {
        // compute H coefficients
        pml_dist = double(k + 0.5)/_w_pml_x1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_x[_w_pml_x0 + k] = kappa;
        _bHx[_w_pml_x0 + k] = b;
        _cHx[_w_pml_x0 + k] = c;

        //compute E coefficients
        pml_dist = double(k)/_w_pml_x1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_x[_w_pml_x0 + k] = kappa;
        _bEx[_w_pml_x0 + k] = b;
        _cEx[_w_pml_x0 + k] = c;
    }
    for(int j = 0; j < _w_pml_y0; j++) {
        // calc H coefficients
        pml_dist = double(j - 0.5)/_w_pml_y0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);
        if(pml_factor < 0) pml_factor = 0;

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_y[j] = kappa;
        _bHy[j] = b;
        _cHy[j] = c;

        // calc E coefficients
        pml_dist = double(j)/_w_pml_y0; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_E_y[j] = kappa;
        _bEy[j] = b;
        _cEy[j] = c;
    
    }
    for(int j = 0; j < _w_pml_y1; j++) {
         // calc H coeffs
         pml_dist = double(j + 0.5)/_w_pml_y1; // distance from pml edge
         pml_factor = pml_ramp(pml_dist);

         sigma = _sigma * pml_factor;
         kappa = (_kappa-1) * pml_factor+1;
         alpha = _alpha * (1-pml_factor);
         b = exp(-_dt*(sigma/kappa + alpha));
         if(b == 1) c = 0;
         else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha);

        _kappa_H_y[_w_pml_y0 + j] = kappa;
        _bHy[_w_pml_y0 + j] = b;
        _cHy[_w_pml_y0 + j] = c;

        // compute E coefficients
        pml_dist = double(j)/_w_pml_y1; // distance from pml edge
        pml_factor = pml_ramp(pml_dist);

        sigma = _sigma * pml_factor;
        kappa = (_kappa-1) * pml_factor+1;
        alpha = _alpha * (1-pml_factor);
        b = exp(-_dt*(sigma/kappa + alpha));
        if(b == 1) c = 0;
        else c = (b - 1)*sigma / (sigma*kappa + kappa*kappa*alpha); 

        _kappa_E_y[_w_pml_y0 + j] = kappa;
        _bEy[_w_pml_y0 + j] = b;
        _cEy[_w_pml_y0 + j] = c;
    }

}

double fdtd_2d::FDTD_TE::pml_ramp(double pml_dist)
{
    return std::pow(pml_dist, _pow);
}

///////////////////////////////////////////////////////////////////////////
// Amp/Phase Calculation management Management
///////////////////////////////////////////////////////////////////////////
void fdtd_2d::FDTD_TE::set_t0_arrays(complex128 *Ez_t0,
                                complex128 *Hx_t0, complex128 *Hy_t0)
{
    _Ez_t0 = Ez_t0;
    _Hx_t0 = Hx_t0; _Hy_t0 = Hy_t0;
}

void fdtd_2d::FDTD_TE::set_t1_arrays(complex128 *Ez_t1,
                                complex128 *Hx_t1, complex128 *Hy_t1)
{
    _Ez_t1 = Ez_t1;
    _Hx_t1 = Hx_t1; _Hy_t1 = Hy_t1;
}

void fdtd_2d::FDTD_TE::capture_t0_fields()
{
    int ind_local, ind_global;

    for(int j = 0; j < _J; j++) {
        for(int k = 0; k < _K; k++) {
            ind_local = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
            ind_global = _J*_K + j*_K + k;

            // Copy the fields at the current time to the auxillary arrays
            _Ez_t0[ind_global] = _Ez[ind_local];

            _Hx_t0[ind_global] = _Hx[ind_local];
            _Hy_t0[ind_global] = _Hy[ind_local];
        }
    }

}

void fdtd_2d::FDTD_TE::capture_t1_fields()
{
    int ind_local, ind_global;

    for(int j = 0; j < _J; j++) {
        for(int k = 0; k < _K; k++) {
            ind_local = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
            ind_global = _J*_K + j*_K + k;

            // Copy the fields at the current time to the auxillary arrays
            _Ez_t1[ind_global] = _Ez[ind_local];

            _Hx_t1[ind_global] = _Hx[ind_local];
            _Hy_t1[ind_global] = _Hy[ind_local];
        }
    }

}

void fdtd_2d::FDTD_TE::calc_complex_fields(double t0, double t1)
{
    double f0, f1, phi, A, t0H, t1H;
    int ind_local, ind_global;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;

    for(int j = 0; j < _J; j++) {
        for(int k = 0; k < _K; k++) {
            ind_local = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
            ind_global = _J*_K + j*_K + k;
            
            // Compute amplitude and phase for Ez
            // Note: we are careful to assume exp(-i*w*t) time dependence
            // Ez
            f0 = _Ez_t0[ind_global].real;
            f1 = _Ez[ind_local];
            phi = calc_phase(t0, t1, f0, f1);
            A = calc_amplitude(t0, t1, f0, f1, phi);
            if(A < 0) {
                A *= -1;
                phi += M_PI;
            }
            _Ez_t0[ind_global].real = A*cos(phi);
            _Ez_t0[ind_global].imag = -A*sin(phi); 

            // Hx
            f0 = _Hx_t0[ind_global].real;
            f1 = _Hx[ind_local];
            phi = calc_phase(t0H, t1H, f0, f1);
            A = calc_amplitude(t0H, t1H, f0, f1, phi);
            if(A < 0) {
                A *= -1;
                phi += M_PI;
            }
            _Hx_t0[ind_global].real = A*cos(phi);
            _Hx_t0[ind_global].imag = -A*sin(phi); 

            // Hy
            f0 = _Hy_t0[ind_global].real;
            f1 = _Hy[ind_local];
            phi = calc_phase(t0H, t1H, f0, f1);
            A = calc_amplitude(t0H, t1H, f0, f1, phi);
            if(A < 0) {
                A *= -1;
                phi += M_PI;
            }
            _Hy_t0[ind_global].real = A*cos(phi);
            _Hy_t0[ind_global].imag = -A*sin(phi); 
        }
    }

}


void fdtd_2d::FDTD_TE::calc_complex_fields(double t0, double t1, double t2)
{
    double f0, f1, f2, phi, A, t0H, t1H, t2H;
    int ind_local, ind_global;

    t0H = t0 - 0.5*_dt;
    t1H = t1 - 0.5*_dt;
    t2H = t2 - 0.5*_dt;

    for(int j = 0; j < _J; j++) {
        for(int k = 0; k < _K; k++) {
            ind_local = (_J+2)*(_K+2) + (j+1)*(_K+2) + k + 1;
            ind_global = _J*_K + j*_K + k;

            // Compute amplitude and phase for Ez
            // Note: we are careful to assume exp(-i*w*t) time dependence
            // Ez
            f0 = _Ez_t0[ind_global].real;
            f1 = _Ez_t1[ind_global].real;
            f2 = _Ez[ind_local];
            phi = calc_phase(t0, t1, t2, f0, f1, f2);
            A = calc_amplitude(t0, t1, t2, f0, f1, f2, phi);
            if(A < 0) {
                A *= -1;
                phi += M_PI;
            }
            _Ez_t0[ind_global].real = A*cos(phi);
            _Ez_t0[ind_global].imag = -A*sin(phi); 

            // Hx
            f0 = _Hx_t0[ind_global].real;
            f1 = _Hx_t1[ind_global].real;
            f2 = _Hx[ind_local];
            phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
            A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
            if(A < 0) {
                A *= -1;
                phi += M_PI;
            }
            _Hx_t0[ind_global].real = A*cos(phi);
            _Hx_t0[ind_global].imag = -A*sin(phi); 

            // Hy
            f0 = _Hy_t0[ind_global].real;
            f1 = _Hy_t1[ind_global].real;
            f2 = _Hy[ind_local];
            phi = calc_phase(t0H, t1H, t2H, f0, f1, f2);
            A = calc_amplitude(t0H, t1H, t2H, f0, f1, f2, phi);
            if(A < 0) {
                A *= -1;
                phi += M_PI;
            }
            _Hy_t0[ind_global].real = A*cos(phi);
            _Hy_t0[ind_global].imag = -A*sin(phi); 

        }
    }
}

inline double fdtd_2d::calc_phase(double t0, double t1, double f0, double f1)
{
    if(f0 == 0.0 and f1 == 0) {
        return 0.0;
    }
    else {
        return atan((f1*sin(t0)-f0*sin(t1))/(f0*cos(t1)-f1*cos(t0)));
    }
}

inline double fdtd_2d::calc_amplitude(double t0, double t1, double f0, double f1, double phase)
{
    if(f0*f0 > f1*f1) {
        return f1 / (sin(t1)*cos(phase) + cos(t1)*sin(phase));
    }
    else {
        return f0 / (sin(t0)*cos(phase) + cos(t0)*sin(phase));
    }
}

inline double fdtd_2d::calc_phase(double t0, double t1, double t2, double f0, double f1, double f2)
{
    double f10 = f1 - f0,
           f21 = f2 - f1;

    if(f10 == 0 && f21 == 0) {
        return 0.0;
    }
    else {
        return atan2(f10*(sin(t2)-sin(t1)) - f21*(sin(t1)-sin(t0)), 
                     f21*(cos(t1)-cos(t0)) - f10*(cos(t2)-cos(t1)));
    }
}

inline double fdtd_2d::calc_amplitude(double t0, double t1, double t2, double f0, double f1, double f2, double phase)
{
    double f21 = f2 - f1,
           f10 = f1 - f0;

    if(f21 == 0 && f10 == 0) {
        return 0.0;
    }
    else if(f21*f21 >= f10*f10) {
        return f21 / (cos(phase)*(sin(t2)-sin(t1)) + sin(phase)*(cos(t2)-cos(t1)));
    }
    else {
        return f10 / (cos(phase)*(sin(t1)-sin(t0)) + sin(phase)*(cos(t1)-cos(t0)));
    }
}

///////////////////////////////////////////////////////////////////////////
// Source management
///////////////////////////////////////////////////////////////////////////
void fdtd_2d::FDTD_TE::add_source(complex128 *Jz,
                            complex128 *Mx, complex128 *My,
                            int j0, int k0, int J, int K,
                            bool calc_phase)
{
    int ind=0;
    double real, imag;
    SourceArray_TE src = {Jz, Mx, My, j0, k0, J, K};

    // these source arrays may *actually* be compelx-valued. In the time
    // domain, complex values correspond to temporal phase shifts. We need
    // to convert the complex value to an amplitude and phase. Fortunately,
    // we can use the memory that is already allocated for these values.
    // Specifically, we use src_array.real = amplitude and
    // src_array.imag = phase
    //
    // Important note: EMopt assumes the time dependence is exp(-i*omega*t).
    // In order to account for this minus sign, we need to invert the sign
    // of the calculated phase.
    if(calc_phase) {

    for(int j = 0; j < J; j++) {
        for(int k = 0; k < K; k++) {
            ind = J*K + j*K + k;


            // Jz
            real = Jz[ind].real;
            imag = Jz[ind].imag;

            Jz[ind].real = sqrt(real*real + imag*imag);
            if(imag == 0 && real == 0) Jz[ind].imag = 0.0;
            else Jz[ind].imag = -1*atan2(imag, real);

            // Mx
            real = Mx[ind].real;
            imag = Mx[ind].imag;

            Mx[ind].real = sqrt(real*real + imag*imag);
            if(imag == 0 && real == 0) Mx[ind].imag = 0.0;
            else Mx[ind].imag = -1*atan2(imag, real);

            // My
            real = My[ind].real;
            imag = My[ind].imag;

            My[ind].real = sqrt(real*real + imag*imag);
            if(imag == 0 && real == 0) My[ind].imag = 0.0;
            else My[ind].imag = -1*atan2(imag, real);

        }
    }
    }

    _sources.push_back(src);
}

void fdtd_2d::FDTD_TE::clear_sources()
{
    _sources.clear();
}

void fdtd_2d::FDTD_TE::set_source_properties(double src_T, double src_min)
{
    _src_T = src_T;
    _src_min = src_min;
    _src_k = src_T*src_T / log((1+src_min)/src_min);
    //_src_k = 6.0 / src_T; // rate of src turn on
    //_src_n0 = 1.0 / _src_k * log((1.0-src_min)/src_min); // src delay
}

inline double fdtd_2d::FDTD_TE::src_func_t(int n, double t, double phase)
{
    //return sin(t + phase) / (1.0 + exp(-_src_k*(n-_src_n0)));
    if(t <= _src_T)
        return sin(t + phase)*((1+_src_min) * exp(-(t-_src_T)*(t-_src_T) / _src_k) - _src_min);
    else
        return sin(t + phase);
}

///////////////////////////////////////////////////////////////////////////
// Boundary Conditions
///////////////////////////////////////////////////////////////////////////
void fdtd_2d::FDTD_TE::set_bc(char* newbc)
{
    for(int i = 0; i < 2; i++){
        _bc[i] = newbc[i];
    }
}

///////////////////////////////////////////////////////////////////////////
// ctypes interface
///////////////////////////////////////////////////////////////////////////

fdtd_2d::FDTD_TE* FDTD_TE_new()
{
    return new fdtd_2d::FDTD_TE();
}

void FDTD_TE_set_wavelength(fdtd_2d::FDTD_TE* fdtd_TE, double wavelength)
{
    fdtd_TE->set_wavelength(wavelength);
}

void FDTD_TE_set_physical_dims(fdtd_2d::FDTD_TE* fdtd_TE, 
                            double X, double Y,
                            double dx, double dy)
{
    fdtd_TE->set_physical_dims(X, Y, dx, dy);
}

void FDTD_TE_set_grid_dims(fdtd_2d::FDTD_TE* fdtd_TE, int Nx, int Ny)
{
    fdtd_TE->set_grid_dims(Nx, Ny);
}

void FDTD_TE_set_local_grid(fdtd_2d::FDTD_TE* fdtd_TE, 
                         int k0, int j0
                         int K, int J)
{
    fdtd_TE->set_local_grid(k0, j0, K, J);
}


void FDTD_TE_set_dt(fdtd_2d::FDTD_TE* fdtd_TE, double dt)
{
    fdtd_TE->set_dt(dt);
}

void FDTD_TE_set_complex_eps(fdtd_2d::FDTD_TE* fdtd_TE, bool complex_eps)
{
    fdtd_TE->set_complex_eps(complex_eps);
}

void FDTD_TE_set_field_arrays(fdtd_2d::FDTD_TE* fdtd_TE,
                           double *Ez,
                           double *Hx, double *Hy)
{
    fdtd_TE->set_field_arrays(Ez, Hx, Hy);
}

void FDTD_TE_set_mat_arrays(fdtd_2d::FDTD_TE* fdtd_TE,
                         complex128 *eps_z,
                         complex128 *mu_x, complex128 *mu_y)
{
    fdtd_TE->set_mat_arrays(eps_z, mu_x, mu_y);
}

void FDTD_TE_update_H(fdtd_2d::FDTD_TE* fdtd_TE, int n, double t)
{
    fdtd_TE->update_H(n, t);
}

void FDTD_TE_update_E(fdtd_2d::FDTD_TE* fdtd_TE, int n, double t)
{
    fdtd_TE->update_E(n, t);
}

void FDTD_TE_set_pml_widths(fdtd_2d::FDTD_TE* fdtd_TE, int xmin, int xmax,
                                           int ymin, int ymax)
{
    fdtd_TE->set_pml_widths(xmin, xmax, ymin, ymax);
}

void FDTD_TE_set_pml_properties(fdtd_2d::FDTD_TE* fdtd_TE, double sigma, double alpha,
                                               double kappa, double pow)
{
    fdtd_TE->set_pml_properties(sigma, alpha, kappa, pow);
}

void FDTD_TE_build_pml(fdtd_2d::FDTD_TE* fdtd_TE)
{
    fdtd_TE->build_pml();
}

void FDTD_TE_reset_pml(fdtd_2d::FDTD_TE* fdtd_TE)
{
    fdtd_TE->reset_pml();
}

void FDTD_TE_set_t0_arrays(fdtd_2d::FDTD_TE* fdtd_TE,
                         complex128 *Ez_t0,
                         complex128 *Hx_t0, complex128 *Hy_t0)
{
    fdtd_TE->set_t0_arrays(Ez_t0, Hx_t0, Hy_t0);
}

void FDTD_TE_set_t1_arrays(fdtd_2d::FDTD_TE* fdtd_TE,
                         complex128 *Ez_t1,
                         complex128 *Hx_t1, complex128 *Hy_t1)
{
    fdtd_TE->set_t1_arrays(Ez_t1, Hx_t1, Hy_t1);
}

double FDTD_TE_calc_phase_2T(double t0, double t1, double f0, double f1)
{
    return fdtd_2d::calc_phase(t0, t1, f0, f1);
}

double FDTD_TE_calc_amplitude_2T(double t0, double t1, double f0, double f1, double phase)
{
    return fdtd_2d::calc_amplitude(t0, t1, f0, f1, phase);
}

double FDTD_TE_calc_phase_3T(double t0, double t1, double t2, double f0, double f1, double f2)
{
    return fdtd_2d::calc_phase(t0, t1, t2, f0, f1, f2);
}

double FDTD_TE_calc_amplitude_3T(double t0, double t1, double t2, double f0, double f1, double f2, double phase)
{
    return fdtd_2d::calc_amplitude(t0, t1, t2, f0, f1, f2, phase);
}

void FDTD_TE_capture_t0_fields(fdtd_2d::FDTD_TE* fdtd_TE)
{
    fdtd_TE->capture_t0_fields();
}

void FDTD_TE_capture_t1_fields(fdtd_2d::FDTD_TE* fdtd_TE)
{
    fdtd_TE->capture_t1_fields();
}


void FDTD_TE_calc_complex_fields_2T(fdtd_2d::FDTD_TE* fdtd_TE, double t0, double t1)
{
    fdtd_TE->calc_complex_fields(t0, t1);
}

void FDTD_TE_calc_complex_fields_3T(fdtd_2d::FDTD_TE* fdtd_TE, double t0, double t1, double t2)
{
    fdtd_TE->calc_complex_fields(t0, t1, t2);
}

void FDTD_TE_add_source(fdtd_2d::FDTD_TE* fdtd_TE,
                     complex128 *Jz,
                     complex128 *Mx, complex128 *My,
                     int j0, int k0, int J, int K, bool calc_phase)
{
    fdtd_TE->add_source(Jz, Mx, My, j0, k0, J, K, calc_phase);
}

void FDTD_TE_clear_sources(fdtd_2d::FDTD_TE* fdtd_TE)
{
    fdtd_TE->clear_sources();
}

void FDTD_TE_set_source_properties(fdtd_2d::FDTD_TE* fdtd_TE, double src_T, double src_min)
{
    fdtd_TE->set_source_properties(src_T, src_min);
}

double FDTD_TE_src_func_t(fdtd_2d::FDTD_TE* fdtd_TE, int n, double t, double phase)
{
    return fdtd_TE->src_func_t(n, t, phase);
}

void FDTD_TE_set_bc(fdtd_2d::FDTD_TE* fdtd_TE, char* newbc)
{
    fdtd_TE->set_bc(newbc);
}

// Ghost communication helper functions
void FDTD_TE_copy_to_ghost_comm(double* src, complex128* ghost, int J, int K)
{
    unsigned int nstart = 0,
                 ind_jk, ind_ghost;

    // copy xmin
    for(int j = 0; j < J; j++) {
        ind_jk = (J+2)*(K+2) + (j+1)*(K+2) + 1;
        ind_ghost = nstart + j; 

        ghost[ind_ghost] = src[ind_jk];
    }

    // copy xmax
    nstart = J;
    for(int j = 0; j < J; j++) {
        ind_jk = (J+2)*(K+2) + (j+1)*(K+2) + K;
        ind_ghost = nstart + j;

        ghost[ind_ghost] = src[ind_jk];
    }

    // copy ymin
    nstart = 2*J;
    for(int k = 0; k < K; k++) {
        ind_jk = (J+2)*(K+2) + 1*(K+2) + k + 1;
        ind_ghost = nstart + k;

        ghost[ind_ghost] = src[ind_jk];
    }

    // copy ymax
    nstart = 2*J + K;
    for(int k = 0; k < K; k++) {
        ind_jk = (J+2)*(K+2) + J*(K+2) + k + 1;
        ind_ghost = nstart + k;

        ghost[ind_ghost] = src[ind_jk];
    }

}

void FDTD_TE_copy_from_ghost_comm(double* dest, complex128* ghost, int J, int K)
{
    unsigned int nstart = 2*J + 2*K.
                 ind_jk, ind_ghost;

    // copy xmin
    for(int j = 0; j < J; j++) {
        ind_jk = (J+2)*(K+2) + (j+1)*(K+2) + 0;
        ind_ghost = nstart + j;

        dest[ind_jk] = ghost[ind_ghost].real;
    }

    // copy xmax
    nstart = 2*J + 2*K + J;
    for(int j = 0; j < J; j++) {
        ind_jk = (J+2)*(K+2) + (j+1)*(K+2) + K+1;
        ind_ghost = nstart + j;

        dest[ind_jk] = ghost[ind_ghost].real;
    }

    // copy ymin
    nstart = 2*J + 2*K + 2*J;
    for(int k = 0; k < K; k++) {
        ind_jk = (J+2)*(K+2) + 0*(K+2) + k + 1;
        ind_ghost = nstart + k;

        dest[ind_jk] = ghost[ind_ghost].real;
    }

    // copy ymax
    nstart = 2*J + 2*K + 2*J + K;
    for(int k = 0; k < K; k++) {
        ind_jk = (J+2)*(K+2) + (J+1)*(K+2) + k + 1;
        ind_ghost = nstart + k;

        dest[ind_jk] = ghost[ind_ghost].real;
    }

}
