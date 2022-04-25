import autolens as al
import numpy as np
import grid_util
import pixelized_mass
import pixelized_source
import potential_correction_util as pcu
import scipy.linalg as linalg
from scipy.spatial import Delaunay
from potential_correction_util import LinearNDInterpolatorExt
from matplotlib import pyplot as plt
import copy
from plot import pixelized_source as ps_plot


class IterativePotentialCorrect(object):
    def __init__(self, masked_imaging, shape_2d_dpsi=None, shape_2d_src=(50,50)):
        """
        shape_2d_dpsi: the shape of potential correction grid, if not set, this will be set to the lens image shape
        shape_2d_src: the number of grid used for source reconstruction (defined on image-plane)
        """
        self.masked_imaging = masked_imaging #include grid, mask, image, noise, psf etc

        self.image_data = self.masked_imaging.image.native #native image resolution, not the oversanpling one
        self.image_noise = self.masked_imaging.noise_map.native
        self.psf_kernel =  self.masked_imaging.psf.native
        image_mask = self.masked_imaging.mask 
        dpix_data = self.masked_imaging.pixel_scales[0]

        if shape_2d_dpsi is None:
            shape_2d_dpsi = self.image_data.shape
        self.grid_obj = grid_util.SparseDpsiGrid(image_mask, dpix_data, shape_2d_dpsi=shape_2d_dpsi) #Note, mask_data has not been cleaned

        self.shape_2d_src = shape_2d_src


    def initialize_iteration(
        self, 
        psi_2d_start=None, 
        niter=100, 
        lam_s_start=None, 
        lam_dpsi_start=1e9,
        psi_anchor_points=None,
    ):
        """
        psi_2d_0: the lens potential map of the initial start mass model, typicall given by a macro model like elliptical power law model.
        niter: the upper limit of the number of the potential correctio iterations
        lam_s_0: the initial regularization strength of pixelized sources. 
        lam_dpsi_0: the initial regularization strength of potential correction (dpsi)
        psi_anchor_points: the anchor points of lens potential. we require the lens potential values at those anchor point
        remain unchanged during potential corrention, to avoid various degeneracy problems. (see sec.2.3 in our document);
        dpsi_anchor_points has the following form: [(y1,x1), (y2,x2), (y3,x3)]
        """
        self._niter = niter
        self._lam_s_start = lam_s_start
        self._lam_dpsi_start = lam_dpsi_start
        self._psi_anchor_points = psi_anchor_points
        self._psi_2d_start = psi_2d_start
        self._psi_2d_start[self.masked_imaging.mask] = 0.0 #set the lens potential of masked pixels to 0

        # self.cum_dpsi_2d_tmp = np.zeros_like(self._psi_2d_start, dtype='float') #the cumulative 2d poential correction in native image resolution.
        # self.cum_dpsi_2d = np.zeros_like(self._psi_2d_start, dtype='float') #the cumulative 2d poential correction in native image resolution.

        #do iteration-0, the macro model
        self.count_iter = 0 #count the iteration number
        #initialized some necessary info for potential correction
        #1--------regularization of source and lens potential
        self.lam_s_this_iter = self._lam_s_start #source regularization strength of current iteration
        self.lam_dpsi_this_iter = self._lam_dpsi_start #potential correction regularization strength of current iteration
        #2--------the lensing potential of currect iteration
        self.pix_mass_this_iter = self.pixelized_mass_from(self._psi_2d_start) #init pixelized mass object
        #3---------pix src obj is mainly used for evalulating lens mapping matrix given a lens mass model
        self.pix_src_obj = pixelized_source.PixelizedSource(
            self.masked_imaging, 
            pixelization_shape_2d=self.shape_2d_src,
        ) 

        #to begin the potential correction algorithm, we need a initial guess of source light
        #do the source inversion for the initial mass model
        self.pix_src_obj.source_inversion(
            self.pix_mass_this_iter, 
            lam_s=self.lam_s_this_iter,
        )
        #Note: self.s_points_this_iter are given in autolens [(y1,x1),(y2,x2),...] order
        self._s_values_start = self.pix_src_obj.src_recontruct[:] #the intensity values of current best-fit pixelized source model
        self._s_points_start = np.copy(self.pix_src_obj.relocated_pixelization_grid) #the location of pixelized source grids (on source-plane).
        self.s_values_this_iter = np.copy(self._s_values_start)
        self.s_points_this_iter = np.copy(self._s_points_start)

        #Init other auxiliary info
        self._psi_anchor_values = self.pix_mass_this_iter.eval_psi_at(self._psi_anchor_points)
        self.pix_src_obj.inverse_covariance_matrix()
        self.inv_cov_matrix =  np.copy(self.pix_src_obj.inv_cov_mat) #inverse covariance matrix
        self._ns = len(self.s_values_this_iter) #number source grids
        self._np = len(self.grid_obj.xgrid_dpsi_1d) #number dpsi grids
        self._d_1d = self.image_data[~self.grid_obj.mask_data] #1d unmasked image data
        self._n_1d = self.image_noise[~self.grid_obj.mask_data] #1d unmasked noise
        self.B_matrix = np.copy(self.pix_src_obj.psf_blur_matrix) #psf bluring matrix, see eq.7 in our document
        self.Cf_matrix = np.copy(
            self.grid_obj.map_matrix
        ) #see the $C_f$ matrix in our document (eq.7), which interpolate data defined on coarser dpsi grid to native image grid
        self.Dpsi_matrix = pcu.dpsi_gradient_operator_from(
            self.grid_obj.Hx_dpsi, 
            self.grid_obj.Hy_dpsi
        ) #the potential correction gradient operator, see the eq.8 in our document
        self.dpsi_grid_points = np.vstack([self.grid_obj.ygrid_dpsi_1d, self.grid_obj.xgrid_dpsi_1d]).T #points of sparse potential correction grid
        
        #calculate the merit of initial macro model. see eq.16 in our document 
        # merit_0 = np.sum(self.pix_src_obj.norm_residual_map**2) + \
        #     np.matmul(
        #         self.pix_src_obj.src_recontruct.T, 
        #         np.matmul(
        #             self.pix_src_obj.regularization_matrix, 
        #             self.pix_src_obj.src_recontruct
        #         )
        #     )
        self.merit_0 = np.inf #float(merit_0)
        self.merit_this_iter = self.merit_0

        #visualize iteration-0
        self.visualize_iteration(iter_num=self.count_iter)

        #assign info of this iteration to the previous one
        self.update_iterations()


    def pixelized_mass_from(self, psi_2d):
        pix_mass_obj = pixelized_mass.PixelizedMass(
            xgrid=self.grid_obj.xgrid_data, 
            ygrid=self.grid_obj.ygrid_data, 
            psi_map=psi_2d, 
            mask=self.grid_obj.mask_data, 
            Hx=self.grid_obj.Hx_data, 
            Hy=self.grid_obj.Hy_data,
            Hxx=self.grid_obj.Hxx_data, 
            Hyy=self.grid_obj.Hyy_data,
        ) 
        return pix_mass_obj


    def update_lam_s(self):
        """
        update the regularization strength of source with iterations
        """
        self.lam_s_this_iter = self.lam_s_prev_iter


    def update_lam_dpsi(self):
        """
        update the regularization strength of potential correction with iterations
        """
        self.lam_dpsi_this_iter = self.lam_dpsi_prev_iter * 1.0 #* 0.1
        pass 


    def update_iterations(self):
        self.count_iter += 1
        #this iteration becomes previous iteration
        self.lam_s_prev_iter = self.lam_s_this_iter
        self.lam_dpsi_prev_iter = self.lam_dpsi_this_iter
        self.pix_mass_prev_iter = copy.copy(self.pix_mass_this_iter)
        self.s_values_prev_iter = np.copy(self.s_values_this_iter)
        self.s_points_prev_iter = np.copy(self.s_points_this_iter)
        self.merit_prev_iter = self.merit_this_iter

        #erase information of this iteration 
        self.lam_s_this_iter = None
        self.lam_dpsi_this_iter = None
        self.pix_mass_this_iter = None
        self.s_values_this_iter = None
        self.s_points_this_iter = None
        self.merit_this_iter = None


    def return_L_matrix(self):
        #need to update lens mapping beforehand
        return np.copy(self.pix_src_obj.mapping_matrix)


    def return_Ds_matrix(self, pix_mass_obj, source_points, source_values):
        self.alpha_dpsi_yx = pix_mass_obj.eval_alpha_yx_at(self.dpsi_grid_points) #use previously found pix_mass_object to ray-tracing
        self.alpha_dpsi_yx = np.asarray(self.alpha_dpsi_yx).T
        self.src_plane_dpsi_yx = self.dpsi_grid_points - self.alpha_dpsi_yx #the location of dpsi grid on the source-plane
        source_gradient = pcu.source_gradient_from(
            source_points, #previously found best-fit src pixlization grids
            source_values, #previously found best-fit src reconstruction
            self.src_plane_dpsi_yx, 
            cross_size=1e-3,
        )
        return pcu.source_gradient_matrix_from(source_gradient)  

    
    def return_RTR_matrix(self):
        #need to update lens mapping beforehand
        #see eq.21 in our document, the regularization matrix for both source and lens potential corrections.
        RTR_matrix = np.zeros((self._ns+self._np, self._ns+self._np), dtype='float')

        self.pix_src_obj.build_reg_matrix(lam_s=self.lam_s_this_iter) #this statement depend on the lens mass model (via the `mapper`)
        RTR_matrix[0:self._ns, 0:self._ns] = np.copy(self.pix_src_obj.regularization_matrix)

        HTH_dpsi = np.matmul(self.grid_obj.Hx_dpsi_4th.T, self.grid_obj.Hx_dpsi_4th) + \
            np.matmul(self.grid_obj.Hy_dpsi_4th.T, self.grid_obj.Hy_dpsi_4th)
        RTR_matrix[self._ns:, self._ns:] = self.lam_dpsi_this_iter**2 * HTH_dpsi

        return RTR_matrix    


    def Mc_RTR_matrices_from(self, pix_mass_obj, source_points, source_values):
        self.pix_src_obj.build_lens_mapping(pix_mass_obj) #update the lens mapping matrix with pixelized mass object

        self.L_matrix = self.return_L_matrix()
        self.Ds_matrix = self.return_Ds_matrix(pix_mass_obj, source_points, source_values)

        self.intensity_deficit_matrix = -1.0*np.matmul(
            self.Cf_matrix,
            np.matmul(
                self.Ds_matrix,
                self.Dpsi_matrix,
            )
        )

        self.Lc_matrix = np.hstack([self.L_matrix, self.intensity_deficit_matrix]) #see eq.14 in our document
        self.Mc_matrix = np.matmul(self.B_matrix, self.Lc_matrix)

        self.RTR_matrix = self.return_RTR_matrix()


    def return_data_vector(self):
        #need to update Mc_matrix beforehand
        #see the right hand side of eq.20 in our document
        data_vector = np.matmul(
            np.matmul(self.Mc_matrix.T, self.inv_cov_matrix),
            self._d_1d,
        )
        return data_vector

    
    def run_this_iteration(self):
        #update regularization parameters for this iteration
        self.update_lam_s()
        self.update_lam_dpsi()

        self.Mc_RTR_matrices_from(self.pix_mass_prev_iter, self.s_points_prev_iter, self.s_values_prev_iter)
        data_vector = self.return_data_vector()

        #solve the next source and potential corrections
        temp_term = np.matmul(
            np.matmul(self.Mc_matrix.T, self.inv_cov_matrix),
            self.Mc_matrix,
        )
        self.r_vector = linalg.solve(temp_term+self.RTR_matrix, data_vector)

        #extract source
        self.s_values_this_iter = self.r_vector[0:self._ns]
        self.s_points_this_iter = np.copy(self.pix_src_obj.relocated_pixelization_grid)
        #extract potential correction
        dpsi_2d = np.zeros_like(self._psi_2d_start, dtype='float')
        dpsi_2d[~self.grid_obj.mask_data] = np.matmul(
            self.Cf_matrix, 
            self.r_vector[self._ns:]
        )
        #update lens potential with potential correction at this iteration
        psi_2d_this_iter = self.pix_mass_prev_iter.psi_map + dpsi_2d #the new 2d lens potential map
        #rescale the current lens potential, to avoid various degeneracy problems. (see sec.2.3 in our document);
        psi_2d_this_iter = self.rescale_lens_potential(psi_2d_this_iter)
        #get pixelized mass object of this iteration
        self.pix_mass_this_iter = self.pixelized_mass_from(psi_2d_this_iter)

        #do visualization
        self.visualize_iteration(iter_num=self.count_iter)

        #check convergence
        #TODO, better to be s_{i} and psi_{i+1}?
        self.merit_this_iter = self.return_this_iter_merit()
        if self.has_converged():
            return True
            
        # if not converge, keep updating 
        self.update_iterations()
        return False 

    
    def rescale_lens_potential(self, psi_2d_in):
        if not hasattr(self, 'tri_psi_interp'):
            self.tri_psi_interp = Delaunay(
                list(zip(self.grid_obj.xgrid_data_1d, self.grid_obj.ygrid_data_1d))
            )
        psi_interpolator = LinearNDInterpolatorExt(self.tri_psi_interp, psi_2d_in[~self.grid_obj.mask_data])
        
        psi_anchor_values_new = psi_interpolator(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0])
        psi_2d_out = pcu.rescale_psi_map(
            self._psi_anchor_values, 
            self._psi_anchor_points, 
            psi_anchor_values_new, 
            psi_2d_in, 
            self.grid_obj.xgrid_data, 
            self.grid_obj.ygrid_data,
        )
        psi_2d_out[self.grid_obj.mask_data] = 0.0 #always set lens potential values at masked region to 0.0

        return psi_2d_out


    def has_converged(self):
        relative_change = (self.merit_prev_iter - self.merit_this_iter)/self.merit_this_iter
        print('next VS current merit:', self.merit_prev_iter, self.merit_this_iter, relative_change)

        # if relative_change < 1e-5:
        #     return True
        # else:
        #     return False 
        return False


    def return_this_iter_merit(self):
        self.mapped_reconstructed_image = np.matmul(self.Mc_matrix, self.r_vector)
        self.residual_map =  self.mapped_reconstructed_image - self._d_1d
        self.norm_residual_map = self.residual_map / self._n_1d
        self.chi_squared = np.sum(self.norm_residual_map**2)

        self.reg_terms = np.matmul(
            self.r_vector.T, 
            np.matmul(
                self.RTR_matrix, 
                self.r_vector
            )
        ) #include the contribution from both source and potential corrections

        return self.chi_squared + self.reg_terms


    def run_iter_solve(self):
        for ii in range(1, self._niter):
            condition = self.run_this_iteration()
            if condition:
                print('111','code converge')  
                break
            else:
                print('111',ii, self.count_iter)  


        
    def visualize_iteration(self, basedir='./result', iter_num=0):
        plt.figure(figsize=(15, 10))
        percent = [0,100]
        cbpar = {}
        cbpar['fraction'] = 0.046
        cbpar['pad'] = 0.04
        myargs = {'origin':'upper'}
        cmap = copy.copy(plt.get_cmap('jet'))
        cmap.set_bad(color='white')
        myargs['cmap'] = cmap
        myargs['extent'] = copy.copy(self.grid_obj.image_bound)
        markersize = 10

        rgrid = np.sqrt(self.grid_obj.xgrid_data**2 + self.grid_obj.ygrid_data**2)
        limit = np.max(rgrid[~self.grid_obj.mask_data])

        #--------data image
        plt.subplot(231)
        vmin = np.percentile(self.image_data,percent[0]) 
        vmax = np.percentile(self.image_data,percent[1]) 
        masked_image_data = np.ma.masked_array(self.image_data, mask=self.grid_obj.mask_data)
        plt.imshow(masked_image_data,vmax=vmax,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(f'Data, Niter={iter_num}')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        #model reconstruction given current mass model
        mapped_reconstructed_image_2d = np.zeros_like(self.image_data)
        self.pix_src_obj.source_inversion(self.pix_mass_this_iter, lam_s=self.lam_s_this_iter)
        mapped_reconstructed_image_2d[~self.grid_obj.mask_data] = np.copy(self.pix_src_obj.mapped_reconstructed_image)
        plt.subplot(232)
        vmin = np.percentile(mapped_reconstructed_image_2d,percent[0]) 
        vmax = np.percentile(mapped_reconstructed_image_2d,percent[1])
        mapped_reconstructed_image_2d = np.ma.masked_array(
            mapped_reconstructed_image_2d, 
            mask=self.grid_obj.mask_data
        ) 
        plt.imshow(mapped_reconstructed_image_2d,vmin=vmin,vmax=vmax,**myargs)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title('Model')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        #normalized residual
        norm_residual_map_2d = np.zeros_like(self.image_data)
        norm_residual_map_2d[~self.grid_obj.mask_data] = np.copy(self.pix_src_obj.norm_residual_map)
        plt.subplot(233)
        vmin = np.percentile(norm_residual_map_2d,percent[0]) 
        vmax = np.percentile(norm_residual_map_2d,percent[1])
        norm_residual_map_2d = np.ma.masked_array(
            norm_residual_map_2d, 
            mask=self.grid_obj.mask_data
        )  
        plt.imshow(norm_residual_map_2d,vmin=vmin,vmax=vmax,**myargs)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title('Norm-residual')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        #source image given current mass model
        plt.subplot(234)
        this_ax = plt.gca()
        ps_plot.visualize_source(
            self.pix_src_obj.relocated_pixelization_grid,
            self.pix_src_obj.src_recontruct,
            ax=this_ax,
        )
        this_ax.set_title('Source')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        cumulative_psi_correct =  self.pix_mass_this_iter.psi_map - self._psi_2d_start
        masked_cumulative_psi_correct = np.ma.masked_array(
            cumulative_psi_correct, 
            mask=self.grid_obj.mask_data
        )
        plt.subplot(235)
        vmin = np.percentile(cumulative_psi_correct,percent[0]) 
        vmax = np.percentile(cumulative_psi_correct,percent[1]) 
        plt.imshow(masked_cumulative_psi_correct,vmin=vmin,vmax=vmax,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(r'potential corrections')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        cumulative_kappa_correct = np.zeros_like(cumulative_psi_correct)
        cumulative_kappa_correct_1d = np.matmul(
            self.grid_obj.hamiltonian_data,
            cumulative_psi_correct[~self.grid_obj.mask_data]
        )
        cumulative_kappa_correct[~self.grid_obj.mask_data] = cumulative_kappa_correct_1d
        masked_cumulative_kappa_correct = np.ma.masked_array(
            cumulative_kappa_correct, 
            mask=self.grid_obj.mask_data
        )
        plt.subplot(236)
        vmin = None #np.percentile(self.dkappa_accum,percent[0]) 
        vmax = None #np.percentile(self.dkappa_accum,percent[1]) 
        plt.imshow(masked_cumulative_kappa_correct,vmin=vmin,vmax=vmax,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(r'kappa corrections')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        plt.tight_layout()
        plt.savefig(f'{basedir}/{iter_num}.jpg', bbox_inches='tight')
        plt.close()