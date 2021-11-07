"""
Optimal transport algorithms for image warping

To do;
    -comments
    -logging warning not separable

"""

__all__ = ['SolveParabolicMAE','SeparableDensities']

import logging
import math

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s - %(levelname)s : %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np 
from numpy.matlib import repmat
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve


class ImgWarpOptimalTransport():
    """Base class for optimal transport for image warping"""

    def plot_density_ratio(self):
        """Calculate and plot ratio between given input density and output 
        density for calculated ot map"""

        Mx = self.ot_map[0]
        My = self.ot_map[1]
        
        ny_in, nx_in = self.density_in.shape
        ny_out, nx_out = self.density_out.shape
        
        x_out = np.linspace(self.xmin, self.xmax, nx_out)*nx_out/nx_in
        y_out = np.linspace(self.ymin, self.ymax, ny_out)*ny_out/nx_in
        dx_out = x_out[1] - x_out[0]
        dy_out = y_out[1] - y_out[0]

        density_out_interp = RegularGridInterpolator(
                            (y_out, x_out), self.density_out, method = 'linear', 
                            bounds_error = False, fill_value = None)

        dMx_dy, dMx_dx = np.gradient(Mx, dy_out, dx_out, edge_order = 2)
        dMy_dy, dMy_dx = np.gradient(My, dy_out, dx_out, edge_order = 2)    
        det_jacobi = np.abs(dMx_dx*dMy_dy - dMx_dy*dMy_dx)
        
        distance = density_out_interp((My, Mx))*det_jacobi / self.density_in
        
        fig, axes = plt.subplots(1, 1, dpi=300)
        img = axes.imshow(distance)
        axes.set_title(r'Output($\mathbf{u}$)det($\nabla\mathbf{u}$) / Input($\mathbf{x}$)')
        
        fig.colorbar(img,ax=axes)
        
        return

    def plot_map(self, step: int = 1):
        """
        Plot ot map as meshgrid

        Parameters
        ----------
        step : int, optional
            skip lines in meshgrid. The default is 2.

        Returns
        -------
        None.

        """
        
        Mx = self.ot_map[0]
        My = self.ot_map[1]
        
        fig, axes = plt.subplots(1, 1, dpi=300)
        
        Mx_t = np.transpose(Mx)
        My_t = np.transpose(My)

        axes.plot(Mx[::step,::step], My[::step,::step], color="b",linewidth=1)
        axes.plot(Mx_t[::step,::step], My_t[::step,::step], color="b",linewidth=1)
        axes.set_xlim(np.min(Mx),np.max(Mx))
        axes.set_ylim(np.min(My),np.max(My))
        axes.invert_yaxis()
        axes.set_title('OT map')
        axes.set_aspect('equal')

        return
    
    def plot_densities(self):
        """Plot input and output density"""

        density_in = self.density_in
        density_out = self.density_out
        
        fig, axes = plt.subplots(1,2,dpi=300)

        axes[0].imshow(density_in)
        img2 = axes[1].imshow(density_out)
        
        axes[0].set_title("Input density")
        axes[1].set_title("Output density")
        
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", "5%", pad="3%",)
        fig.colorbar(img2,cax=cax)
        
        fig.tight_layout()
        
        return

    def _normalize(self, density):
        """Normalize density"""
        norm = np.trapz(np.trapz(density))
        return density/norm
    
    def _offset(self, density, offset = 1):
        """Remove zero values from density distribution """
        if density.min() == 0:
            density += offset
        return density


class SolveParabolicMAE(ImgWarpOptimalTransport):
    '''Calculate the L2 optimal transport map (Mx, My) for a given input and output 
    density on square domains by solving a parabolic Monge Ampère
    equation (see reference below)
    
    Parameters
    ----------
    density_in: float, 2-d array
        Input density for calculating optimal transport map.
        The dimensions of the density are measured relative to the extension
        along the x-axis (from -0.5 to 0.5). The extension along the y-axis is 
        determined by ny_in/nx_in*(-0.5, 0.5) with 
        ny_in, nx_in = density_in.shape
    density_out: float, 2-d array
        Output density for calculating the optimal transport map. The 
        dimensions are defined relative to the extension of the input density 
        along the x-direction by nx_out/nx_in*(-0.5, 0.5) and 
        ny_out/nx_in*(-0.5, 0.5) with ny_in, nx_in = density_in.shape and
        ny_out, nx_out = density_out.shape 
    absTol: float
        tolerance value which defines the precision according to 
        log(density_out(My, Mx)*det_hesse/density_in) for which the MAE is 
        solved
    relTol: float
        relative tolerance
    step_size: float
        defines stepsize of each time step to solve time depentend Monge-Ampère
        equation
    density_interp: {nearest, 'linear'}
        defines interpolation of output density with RegularGridInterpolator
    smooth_map: bool
        defines if mapping components (Mx,My) are interpolated after tolerance
        is reached. For complex density the map might show oscillations 
        which can be smoothened by the interpolation
    smooth_kind : {'convolve', 'interp'}, optional
        defines kind of smoothing of mapping. The default is "convolve".
    smooth_interp : str, optional
        Option for smooth_kind == 'interp'.
        Defines interpolation kind with interp2d. The default is 'cubic'.
            
    Returns
    -------
    Mx, My
         2d-arrays which describe the mapping between input and output 
         density
        
    References
    -------
        [1] M.M. Sulman, J.F. Williams, and R.D. Russel, “An efficient approach 
            for the numerical solution of Monge-Ampère equation,” Appl. Numer. 
            Math. 61(3), 298-307 (2011).
    '''
    xmax, ymax = 0.5, 0.5
    xmin, ymin = -0.5, -0.5
    
    def __init__(self, density_in, density_out,
                 absTol : float = 20.0, relTol: float = 1e-4, 
                 step_size: float = 5e-6, density_interp : str = 'linear', 
                 smooth_map : bool = True, 
                 smooth_kind : str = 'interp',
                 smooth_interp: str = 'cubic'):
        
        if np.min(density_in) < 0:
            raise ValueError("Minimal value of input density should be"
                             + "larger to zero.")
        if np.min(density_out) < 0:
            raise ValueError("Minimal value of output density should be"
                             + "larger to zero.")
        if density_out.shape != density_in.shape:
            raise ValueError(" Shape of array of input and output density must" 
                             + " be identical.")
        if density_out.shape[0] != density_out.shape[1]:
            raise ValueError("Dimensions of array of densities must be" 
                             + "identical.")
        self.density_in = self.__preprocess(density_in)
        self.density_out = self.__preprocess(density_out)
        
        if type(absTol) is not float and type(absTol) is not int:
            raise ValueError("'tolerance' must be a float or int.")
        if type(relTol) is not float and type(relTol) is not int:
            raise ValueError("'tolerance' must be a float or int.")
            
        self.absTol = absTol
        self.relTol = relTol
        
        if type(step_size) is not float and type(step_size) is not int:
            raise ValueError("'step_size' must be a float or int.") 
        self.step_size = step_size
        
        if (smooth_map is not True and smooth_map is not False):
            raise ValueError("'smoothen_map' must be boolean")
        self.smooth_map = smooth_map
        
        if smooth_kind not in ['convolve', 'interp']:
            raise ValueError("'smooth_kind' must be 'convolve' or 'interp'.")
        self.smooth_kind = smooth_kind
        
        self.smooth_interp = smooth_interp
        self.density_interp = density_interp
        
        self.ot_map = self._get_ot_map()
                 
    def _get_ot_map(self):
        """Solve parabolic Monge-Ampère equation"""
        ny, nx = self.density_out.shape
        
        x = np.linspace(self.xmin, self.xmax, nx)
        y = np.linspace(self.ymin, self.ymax, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        X, Y = np.meshgrid(x, y)
        
        self.density_out_interp = RegularGridInterpolator(
                            (y, x), self.density_out, 
                            method = self.density_interp, 
                            bounds_error = False, fill_value = None)

        # initial iterate of potential
        Psi = 1/2*(X**2 + Y**2)
        # initialize while loop
        norm_F_n = self.absTol + 1
        norm_rel = 1
        # solve time dependent parabolic MAE until tolerance value is reached
        while norm_F_n >= self.absTol and norm_rel > self.relTol:
            norm_F_0 = norm_F_n
            
            My, Mx = np.gradient(Psi, dy, dx, edge_order = 2)
            Mx, My = self.__apply_bc(Mx, My)
            
            dMx_dy, dMx_dx = np.gradient(Mx, dy, dx, edge_order = 2)
            dMy_dy, dMy_dx = np.gradient(My, dy, dx, edge_order = 2)    
            det_hesse = np.abs(dMx_dx*dMy_dy - dMx_dy*dMy_dx)
            
            F_n = np.log(self.density_out_interp((My, Mx))*det_hesse/self.density_in)
            
            # update potential u_n --> u_(n+1)
            Psi = Psi + self.step_size*F_n
            
            norm_F_n = np.linalg.norm(F_n, 2)
            norm_rel = np.abs(norm_F_0 - norm_F_n)/self.absTol
            
            logger.info(" Norm F_n / absTol: {}; relTol: {}".format(
                norm_F_n/self.absTol, norm_rel))
        
        if math.isnan(norm_F_n) or norm_F_n == np.inf:
            logger.warn('Solver diverged. Use smaller step size and/or'
                        + ' increase tolerances.')
        
        My, Mx = np.gradient(Psi, dy, dx, edge_order = 2)
        Mx, My = self.__apply_bc(Mx, My)
        
        # map interpolation (to avoid oscillatory behaviour)
        if self.smooth_map:
            Mx, My = self._smoothen_map(Mx, My, self.smooth_kind, 
                                       self.smooth_interp)
            

        return (Mx, My)
        
    def __apply_bc(self, Mx, My):
        """Apply boundary condition"""
        Mx[:, 0] = self.xmin
        Mx[:, -1] = self.xmax
        My[0, :] = self.ymin
        My[-1, :] = self.ymax    
        
        return Mx, My

    def __preprocess(self, density):
        """Normalize and add offset to densities"""
        density = self._offset(density)
        density = self._normalize(density)
        return density

    def _smoothen_map(self, Mx=None, My=None, smooth_kind : str = "interp", 
                     smooth_interp : str = 'cubic'):
        """
        Smooth oscillations of mapping components

        Parameters
        ----------
        Mx, My: ndarray, optional
            2d-arrays which describe the mapping between input and output 
            density. If not provided, the mapping components self.ot_map will
            be utilized.
        smooth_kind : {'convolve', 'interp'}, optional
            defines kind of smoothing of mapping. The default is "convolve".
        smooth_interp : str, optional
            Option for smooth_kind == 'interp'.
            Defines interpolation kind with interp2d. The default is 'cubic'.
        Returns
        -------
        Mx_new, My_new: ndarray
            Smoothened 2d-arrays which describe the mapping between input and 
            output density

        """
        
        if Mx is None or My is None:
            Mx = self.ot_map[0]
            My = self.ot_map[1]
        
        if smooth_kind == 'convolve': 
            weights = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                               dtype=np.float)
            weights = weights / np.sum(weights[:])
            Mx_new = convolve(Mx, weights, mode='nearest')
            My_new = convolve(My, weights, mode='nearest')
        
        elif smooth_kind == 'interp':
            ny,nx = Mx.shape
            
            x = np.linspace(self.xmin, self.xmax, nx)
            y = np.linspace(self.ymin, self.ymax, ny)
            
            nx, ny = len(x), len(y)
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            
            xmax, ymax = np.max(x), np.max(y)
            xmin, ymin = np.min(x), np.min(y)

            x_q = np.linspace(xmin + dx/2, xmax - dx/2, nx - 1)
            y_q = np.linspace(ymin + dy/2, ymax - dy/2, ny - 1)
              
            X_q, X_q = np.meshgrid(x_q, y_q)
    
            Mx_interp = interpolate.interp2d(y, x, Mx, 
                                             kind = smooth_interp)
            My_interp = interpolate.interp2d(y, x, My, 
                                             kind = smooth_interp)
            
            Mx_q = Mx_interp(y_q, x_q)
            My_q = My_interp(y_q, x_q)
    
            Mx_q_interp = interpolate.interp2d(y_q, x_q, Mx_q, 
                                               kind = smooth_interp)
            My_q_interp = interpolate.interp2d(y_q, x_q, My_q, 
                                               kind = smooth_interp)
    
            Mx_new = Mx_q_interp(y, x)
            My_new = My_q_interp(y, x)
            
        return self.__apply_bc(Mx_new, My_new)



class SeparableDensities(ImgWarpOptimalTransport):
    '''This function calcualte the L2 optimal transport map for two separable
    densities
        in_dist(x,y) = f_x(x)*f_y(y)
        out_dist(x,y) = g_x(x)*g_y(y)
    by two one dimensional integrations of the energy conservation equation 
    along the x- and y-direction
    
    Parameters
    ----------
    density_in: float, 2-d array
        Input density for calculating optimal transport map
    density_out: float, 2-d array
        Output density for calculating optimal transport map
    density_interp: str
        specifies kind of interpolation with interp1d
    Returns
    -------
    Mx, My:
         2d-arrays which describe the mapping between input and output 
         density
        
    '''
    
    xmax, ymax = 0.5, 0.5
    xmin, ymin = -0.5, -0.5
    
    def __init__(self, density_in, density_out, 
                 density_interp : str = 'linear'):
        
        if np.min(density_in) < 0:
            raise ValueError("Minimal value of input density should be"
                             + "larger or equal to zero.")
        if np.min(density_out) < 0:
            raise ValueError("Minimal value of output density should be"
                             + "larger or equal to zero.")
        
        self._check_separable(density_out)
        self._check_separable(density_in)
        
        self.density_in = self._normalize(density_in)
        self.density_out = self._normalize(density_out)
        
        self.density_interp = density_interp
        self.ot_map = self._get_ot_map()
    
    def _check_separable(self, density):
        """Check if density is separable"""
        density_x = repmat(density[:,0], density.shape[1], 1)
        density_y = repmat(density[0,:], density.shape[0], 1)
        density_x = density_x.transpose()
        
        density_new = density_x*density_y
        density_new = density_new / density_new[0,0]*density[0,0]
        
        check_eq = np.array_equal(
                   np.round(np.abs(density_new - density),14),
                   np.zeros(density.shape))
        if not check_eq:
            logger.warn("Density not separable. OT map might be wrong.")
        return
    
    def _get_ot_map(self):
        """Calculate OT map from separable densities """
        
        ny_in, nx_in = self.density_in.shape
        ny_out, nx_out = self.density_out.shape
        
        x_in = np.linspace(self.xmin, self.xmax, nx_in)
        y_in = np.linspace(self.ymin, self.ymax, ny_in)*ny_in / nx_in
        
        x_out = np.linspace(self.xmin, self.xmax, nx_out)*nx_out / nx_in
        y_out = np.linspace(self.ymin, self.ymax, ny_out)*ny_out / nx_in
        
        dx_in =  x_in[1] - x_in[0]
        dy_in =  y_in[1] - y_in[0]
        
        dx_out =  x_out[1] - x_out[0]
        dy_out =  y_out[1] - y_out[0]
        
        # Normalized densities in x and y-direction
        density_in_x = self.density_in[0, :]/np.trapz(self.density_in[0, :])
        density_in_y = self.density_in[:, 0]/np.trapz(self.density_in[:, 0])
        
        density_out_x = self.density_out[0, :]/np.trapz(self.density_out[0, :])
        density_out_y = self.density_out[:, 0]/np.trapz(self.density_out[:, 0])
        
        # Integration of densities along x- and y-directions
        density_in_x_cumtr = cumtrapz(density_in_x, initial = 0) * dx_in
        density_in_y_cumtr = cumtrapz(density_in_y, initial = 0) * dy_in
        
        density_out_x_cumtr = cumtrapz(density_out_x, initial = 0) * dx_out
        density_out_y_cumtr = cumtrapz(density_out_y, initial = 0) * dy_out
            
        # calculate mapping
        Mx_1D_interp = interp1d(density_out_x_cumtr, x_out, 
                                fill_value="extrapolate",
                                kind= self.density_interp) 
        My_1D_interp = interp1d(density_out_y_cumtr, y_out, 
                                fill_value="extrapolate",
                                kind= self.density_interp)  
    
        Mx_1D =  Mx_1D_interp(density_in_x_cumtr)
        My_1D =  My_1D_interp(density_in_y_cumtr)
        
        Mx = repmat(Mx_1D, ny_in, 1)
        My = np.transpose(repmat(My_1D, nx_in, 1))
    
        return(Mx, My)

