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


    def init_iteration(
        self, 
        psi_2d_0=None, 
        niter=100, 
        lam_s_0=None, 
        lam_dpsi_0=1e9,
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
        self.lam_s_0 = lam_s_0
        self.lam_dpsi_0 = lam_dpsi_0
        self._psi_anchor_points = psi_anchor_points

        self.iter_count = 0 #count the iteration number
        self.lam_s_current = self.lam_s_0 #source regularization strength of current iteration
        self.lam_dpsi_current = self.lam_dpsi_0 #potential correction regularization strength of current iteration

        self._psi_2d_0 = psi_2d_0
        self._psi_2d_0[self.masked_imaging.mask] = 0.0 #set the lens potential of masked pixels to 0
        self.psi_correction_2d = np.zeros_like(self._psi_2d_0, dtype='float') #the 2d poential correction in native image resolution.
        self.psi_2d_current = self._psi_2d_0 + self.psi_correction_2d #current best-fit lens mass model

        self.pix_mass_current = self.construct_pixelized_mass(self.psi_2d_current)
        #pix src obj is mainly used for evalulating lens mapping matrix given currect lens mass model, also initialize the source light model 
        self.pix_src_obj = pixelized_source.PixelizedSource(
            self.masked_imaging, 
            pixelization_shape_2d=self.shape_2d_src,
        ) 

        self._psi_anchor_values = self.pix_mass_current.eval_psi_at(self._psi_anchor_points)
        #current best-fit source reconstruction
        self.pix_src_obj.source_inversion(
            self.pix_mass_current, 
            lam_s=self.lam_s_current,
        )
        #Note: self.s_points_current are given in autolens [(y1,x1),(y2,x2),...] order
        self.s_values_current = self.pix_src_obj.src_recontruct[:] #the intensity values of current best-fit pixelized source model
        self.s_points_current = np.copy(self.pix_src_obj.relocated_pixelization_grid) #the location of pixelized source grids (on source-plane).

        self.pix_src_obj.inverse_covariance_matrix()
        self.inv_cov_matrix =  np.copy(self.pix_src_obj.inv_cov_mat) #inverse covariance matrix
        self._ns = len(self.s_values_current) #number source grids
        self._np = len(self.grid_obj.xgrid_dpsi_1d) #number dpsi grids
        self._d_1d = self.image_data[~self.grid_obj.mask_data] #1d unmasked image data
        self._n_1d = self.image_noise[~self.grid_obj.mask_data] #1d unmasked noise

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
        self.merit_current = self.merit_0


    def construct_pixelized_mass(self, psi_2d):
        pix_mass_obj = pixelized_mass.PixelizedMass(
            xgrid=self.grid_obj.xgrid_data, 
            ygrid=self.grid_obj.ygrid_data, 
            psi_map=psi_2d, 
            mask=self.grid_obj.mask_data, 
            Hx=self.grid_obj.Hx_data, 
            Hy=self.grid_obj.Hy_data,
        ) 
        return pix_mass_obj


    def update_lam_s(self):
        """
        update the regularization strength of source with iterations
        """
        pass


    def update_lam_dpsi(self):
        """
        update the regularization strength of potential correction with iterations
        """
        # self.lam_dpsi_current = self.lam_dpsi_current * 0.1**(self.iter_count)
        pass 


    def construct_Mc_matrix(self):
        #the lens map matrixces term only depend on current best-fit mass model
        #we only use pix_src object to help us to get L_matrix
        # we don't use it for source inversion; source reconstruction is extracted from self.r_vector
        #i.e, simultaneously solve source and potential corrections!
        self.pix_src_obj.build_lens_mapping(self.pix_mass_current) 
        L_matrix = np.copy(self.pix_src_obj.mapping_matrix) #see eq.12-14 in our documents

        #evaluate source gradient matrix
        dpsi_grid_vec = np.vstack([self.grid_obj.ygrid_dpsi_1d, self.grid_obj.xgrid_dpsi_1d]).T 
        alpha_dpsi_yx = self.pix_mass_current.eval_alpha_yx_at(dpsi_grid_vec) #TODO, use previously found pix_mass_object to ray-tracing?
        alpha_dpsi_yx = np.asarray(alpha_dpsi_yx).T
        src_plane_dpsi_yx = dpsi_grid_vec - alpha_dpsi_yx #the location of dpsi grid on the source-plane
        source_gradient = pcu.source_gradient_from(
            self.s_points_current, ##current best-fit src pixlization grids
            self.s_values_current, #current best-fit src reconstruction
            src_plane_dpsi_yx, 
            cross_size=1e-3,
        )
        Ds_matrix = pcu.source_gradient_matrix_from(source_gradient)

        #evaluate the potential correction gradient operator
        Dpsi_matrix = pcu.dpsi_gradient_operator_from(self.grid_obj.Hx_dpsi, self.grid_obj.Hy_dpsi) 

        #conformation matrix, which interpolate data defined on coarser dpsi grid to native image grid
        Cf_matrix = np.copy(self.grid_obj.map_matrix)

        #psf bluring matrix
        B_matrix = self.pix_src_obj.psf_blur_matrix

        intensity_deficit_matrix = -1.0*np.matmul(
            Cf_matrix,
            np.matmul(
                Ds_matrix,
                Dpsi_matrix,
            )
        )

        Lc_matrix = np.hstack([L_matrix, intensity_deficit_matrix]) #see eq.14 in our document
        
        self.Mc_matrix = np.matmul(B_matrix, Lc_matrix)


    def construct_RTR_matrix(self):
        self.RTR_matrix = np.zeros((self._ns+self._np, self._ns+self._np), dtype='float')

        self.pix_src_obj.build_reg_matrix(lam_s=self.lam_s_current)
        self.RTR_matrix[0:self._ns, 0:self._ns] = np.copy(self.pix_src_obj.regularization_matrix)

        HTH_dpsi = np.matmul(self.grid_obj.Hx_dpsi_4th.T, self.grid_obj.Hx_dpsi_4th) + \
            np.matmul(self.grid_obj.Hy_dpsi_4th.T, self.grid_obj.Hy_dpsi_4th)
        self.RTR_matrix[self._ns:, self._ns:] = self.lam_dpsi_current**2 * HTH_dpsi


    def construct_data_vector(self):
        self.data_vector = np.matmul(
            np.matmul(self.Mc_matrix.T, self.inv_cov_matrix),
            self._d_1d,
        )

    
    def solve_for_next_iteration(self):
        #prepare matrices for potential correction
        self.construct_Mc_matrix()
        self.construct_data_vector()

        #regularization matrices for source and potential corrections
        # self.update_lam_dpsi()
        # self.update_lam_s()
        self.construct_RTR_matrix()

        #solve the next source and potential corrections
        temp_term = np.matmul(
            np.matmul(self.Mc_matrix.T, self.inv_cov_matrix),
            self.Mc_matrix,
        )
        self.r_vector = linalg.solve(temp_term+self.RTR_matrix, self.data_vector)
        self.iter_count += 1

        #update lens potential for next iterations/checking convergence
        self.psi_correction_2d[~self.grid_obj.mask_data] = self.psi_correction_2d[~self.grid_obj.mask_data] + \
            np.matmul(self.grid_obj.map_matrix, self.r_vector[self._ns:])
        self.psi_2d_current = self._psi_2d_0 + self.psi_correction_2d
        print('--------', np.max(self.r_vector[self._ns:]))
        #rescale the current lens potential, to avoid various degeneracy problems. (see sec.2.3 in our document);
        self.psi_2d_current = self.rescale_lens_potential(self.psi_2d_current)

        #check convergence
        #TODO, better to be s_{i} and psi_{i+1}?
        if self.has_converged():
            return True

        # if not converge, keep updating source
        self.s_values_current = self.r_vector[0:self._ns]
        self.s_points_current = np.copy(self.pix_src_obj.relocated_pixelization_grid)
    
        # update current pixelized mass model
        self.pix_mass_current = self.construct_pixelized_mass(self.psi_2d_current)
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
        merit_next = self.return_current_merit()

        print('next VS current merit:', merit_next, self.merit_current)
        relative_change = (self.merit_current - merit_next)/merit_next
        self.merit_current = merit_next

        # if relative_change < 1e-5:
        #     return True
        # else:
        #     return False 
        return False


    def return_current_merit(self):
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
            condition = self.solve_for_next_iteration()
            self.visualize_iteration(niter=self.iter_count)
            if condition:
                print('111','code converge')  
                break
            else:
                print('111',ii, self.iter_count)  


        
    def visualize_iteration(self, basedir='./result', niter=0):
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
        plt.title(f'Data, Niter={niter}')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        mapped_reconstructed_image_2d = np.zeros_like(self.image_data)
        self.pix_src_obj.source_inversion(self.pix_mass_current, lam_s=self.lam_s_current)
        mapped_reconstructed_image_2d[~self.grid_obj.mask_data] = self.pix_src_obj.mapped_reconstructed_image
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

        norm_residual_map_2d = np.zeros_like(self.image_data)
        norm_residual_map_2d[~self.grid_obj.mask_data] = self.pix_src_obj.norm_residual_map
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

        plt.subplot(234)
        this_ax = plt.gca()
        ps_plot.visualize_source(self.s_points_current, self.s_values_current, ax=this_ax)
        plt.title('Source')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        cumulative_psi_correct =  self.psi_2d_current - self._psi_2d_0
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
        plt.savefig(f'{basedir}/{niter}.jpg', bbox_inches='tight')
        plt.close()