import autolens as al
import numpy as np
import grid_util
import pixelized_mass
import potential_correction_util as pcu

class IterativePotentialCorrect(object):
    def __init__(self, masked_imaging, shape_2d_dpsi=None, shape_2d_src=(50,50)):
        """
        shape_2d_dpsi: the shape of potential correction grid, if not set, this will be set to the lens image shape
        shape_2d_src: the number of grid used for source reconstruction (defined on image-plane)
        """
        self.masked_imaging = masked_imaging #include grid, mask, image, noise, psf etc

        self.image_data = self.masked_imaging.image.native
        self.image_noise = self.masked_imaging.noise_map.native
        self.psf_kernel =  self.masked_imaging.psf.native
        image_mask = self.masked_imaging.mask
        dpix_data = self.masked_imaging.pixel_scales[0]

        if shape_2d_dpsi is None:
            shape_2d_dpsi = self.image_data.shape
        self.grid_obj = grid_util.SparseDpsiGrid(image_mask, dpix_data, shape_2d_dpsi=shape_2d_dpsi)  


    def init_pixelized_source(self, pixelization_shape_2d):
        sub_size = 2
        pixelization = al.pix.DelaunayMagnification(shape=pixelization_shape_2d)

