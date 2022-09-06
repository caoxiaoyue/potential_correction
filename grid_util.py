import numpy as np
import autolens as al 
from matplotlib import pyplot as plt
import pickle

def clean_mask_old(in_mask):
    '''
    in_mask: the input 2d mask, a 2d bool numpy array
    out_mask: the output 2d mask after clipping

    the cliping scheme following the method in https://arxiv.org/abs/0804.2827 (see figure-11), remove the so-called "exposed pixels".
    "exposed pixels" has no ajacent pixels so that the gradient can not be calculated via the finite difference.
    '''
    out_mask = np.ones_like(in_mask).astype('bool')
    n1, n2 = in_mask.shape
    for i in range(1,n1-1): #Not range(n1), because I don't want to deal the index error related to the bound
        for j in range(1,n2-1):
            if_exposed = False
            if not in_mask[i,j]:
                if in_mask[i-1,j] and in_mask[i+1,j]:
                    if_exposed = True
                if in_mask[i,j-1] and in_mask[i,j+1]:
                    if_exposed = True
                if not if_exposed:
                    out_mask[i,j] = False
    return out_mask


def clean_mask(in_mask):
    '''
    in_mask: the input 2d mask, a 2d bool numpy array
    out_mask: the output 2d mask after clipping

    the cliping scheme following the method in https://arxiv.org/abs/0804.2827 (see figure-11), remove the so-called "exposed pixels".
    "exposed pixels" has no ajacent pixels so that the gradient can not be calculated via the finite difference.
    '''
    out_mask = np.ones_like(in_mask).astype('bool')
    n1, n2 = in_mask.shape
    for i in range(2,n1-2): #Not range(n1), because I don't want to deal the index error related to the bound
        for j in range(2,n2-2):
            if not in_mask[i,j]:
                if_exposed_0 = True
                if_exposed_1 = True

                if ~in_mask[i-1,j] and ~in_mask[i-2,j]:
                    if_exposed_0 = False
                if ~in_mask[i+1,j] and ~in_mask[i+2,j]:
                    if_exposed_0 = False
                if ~in_mask[i,j-1] and ~in_mask[i,j-2]:
                    if_exposed_1 = False
                if ~in_mask[i,j+1] and ~in_mask[i,j+2]:
                    if_exposed_1 = False

                if not (if_exposed_0 or if_exposed_1):
                    out_mask[i,j] = False

    return out_mask


def linear_weight_from_box(box_x, box_y, position=None):
    """
    The function find the linear interpolation (extrapolation) at `position`,
    given the box with corrdinates box_x and box_y
    box_x: An 4 elements list/tuple/array; save the x-coordinate of box, in the order of [top-left,top-right, bottom-left, bottom-right]
    box_y: An 4 elements list/tuple/array; similar to box_x, save the y-coordinates
    position: the location of which we estimate the linear interpolation weight; a tuple with (y,x) coordinaes, such as (1.0, 0.0),
    the location at x=0,y=1

    return an array with shape [4,], which save the linear interpolation weight in
    [top-left,top-right, bottom-left, bottom-right] order.
    """
    y, x = position
    box_size = box_x[1] - box_x[0]
    wx = (x - box_x[0])/box_size  #x direction weight 
    wy = (y - box_y[2])/box_size   #y direction weight 

    weight_top_left = (1-wx)*wy
    weight_top_right = wx*wy
    weight_bottom_left = (1-wx)*(1-wy)
    weight_bottom_right = wx*(1-wy)

    return np.array([weight_top_left, weight_top_right, weight_bottom_left, weight_bottom_right])


def pixel_type_from_mask(mask):
    """
    This function return the pixel types of each unmasked pixel.
    The pixel type value can be:
    0: has a neighbour pixel on both left and right (or top and bottom), allows a gradient calculation with 2nd order accuracy.
    +2: has two neighbour pixels on the right (or bottom), allows a gradient calculation with 2nd order accuracy.
    -2: has two neighbour pixels on the left (or top), allows a gradient calculation with 2nd order accuracy.
    +1: has only a neighbour pixel on the right (or bottom), allows a gradient calculation with 1st order accuracy.
    -1: has only a neighbour pixel on the left (or top), allows a gradient calculation with 1st order accuracy. 
    mask: an input mask, 2d bool numpy array. The pixels with value of `True` are masked
    pixel_type: shape, [n_unmasked_pixels, 2]
    for example,
    if pixel_type[5,0]==0, this means the y-direction pixel type of 6th unmasked pixel is 0.
    if pixel_type[5,1]==+2, this means the x-direction pixel type of 6th unmasked pixel is +2.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    n_unmasked_pixels = len(i_indices_unmasked)
    pixel_type = np.zeros((n_unmasked_pixels, 2), dtype='int')

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]

        #------check y-direction
        if unmask[i-1,j] and unmask[i+1,j]:
            pixel_type[count, 0] = 0
        elif unmask[i+1,j] and unmask[i+2,j]:
            pixel_type[count, 0] = +2
        elif unmask[i-1,j] and unmask[i-2,j]:
            pixel_type[count, 0] = -2
        elif unmask[i+1,j]:
            pixel_type[count, 0] = +1
        elif unmask[i-1,j]:
            pixel_type[count, 0] = -1
        else:
            bug_info = """
            No matched pixel type for the y-direction numerical finite differential. 
            Please check whether your mask has been `cleaned` so that the `exposed` pixels are removed.
            For the definition of `exposed` pixels, see suyu09 https://arxiv.org/abs/0804.2827 
            """
            raise Exception(bug_info)

        #------check x-direction
        if unmask[i,j-1] and unmask[i,j+1]:
            pixel_type[count, 1] = 0
        elif unmask[i,j+1] and unmask[i,j+2]:
            pixel_type[count, 1] = +2
        elif unmask[i,j-1] and unmask[i,j-2]:
            pixel_type[count, 1] = -2
        elif unmask[i,j+1]:
            pixel_type[count, 1] = +1
        elif unmask[i,j-1]:
            pixel_type[count, 1] = -1
        else:
            bug_info = """
            No matched pixel type for the x-direction numerical finite differential. 
            Please check whether your mask has been `cleaned` so that the `exposed` pixels are removed.
            For the definition of `exposed` pixels, see suyu09 https://arxiv.org/abs/0804.2827 
            """
            raise Exception(bug_info)

    return pixel_type


def diff_1st_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the gradient operator matrix Hx and Hy.
    The gradient operator matrix (Hx and Hy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the x/y gradient of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    pixel_types = pixel_type_from_mask(mask) #shape: [n_unmasked_pixels, 2]

    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked)

    Hx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        this_type = pixel_types[count,:]
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]

        #y-direction gradient
        if this_type[0] == 0:
            indices_tmp = np.ravel_multi_index([(i-1, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked[0]] = -1.0/(2.0*step_y)
            Hy[count,indices_of_indices_1d_unmasked[1]] = +1.0/(2.0*step_y)
        elif this_type[0] == +2: 
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked[0]] = -3.0/(2.0*step_y)
            Hy[count,indices_of_indices_1d_unmasked[1]] = +4.0/(2.0*step_y)
            Hy[count,indices_of_indices_1d_unmasked[2]] = -1.0/(2.0*step_y)
        elif this_type[0] == -2: 
            indices_tmp = np.ravel_multi_index([(i-2, i-1, i), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked[0]] = +1.0/(2.0*step_y)
            Hy[count,indices_of_indices_1d_unmasked[1]] = -4.0/(2.0*step_y)
            Hy[count,indices_of_indices_1d_unmasked[2]] = +3.0/(2.0*step_y)
        elif this_type[0] == +1: 
            indices_tmp = np.ravel_multi_index([(i, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked[0]] = -1.0/step_y
            Hy[count,indices_of_indices_1d_unmasked[1]] = +1.0/step_y
        elif this_type[0] == -1: 
            indices_tmp = np.ravel_multi_index([(i-1, i), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked[0]] = -1.0/step_y
            Hy[count,indices_of_indices_1d_unmasked[1]] = +1.0/step_y        

        #x-direction gradient
        if this_type[1] == 0:
            indices_tmp = np.ravel_multi_index([(i, i), (j-1, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked[0]] = -1.0/(2.0*step_x)
            Hx[count,indices_of_indices_1d_unmasked[1]] = +1.0/(2.0*step_x)
        elif this_type[1] == +2: 
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked[0]] = -3.0/(2.0*step_x)
            Hx[count,indices_of_indices_1d_unmasked[1]] = +4.0/(2.0*step_x)
            Hx[count,indices_of_indices_1d_unmasked[2]] = -1.0/(2.0*step_x)
        elif this_type[1] == -2: 
            indices_tmp = np.ravel_multi_index([(i ,i ,i), (j-2, j-1, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked[0]] = +1.0/(2.0*step_x)
            Hx[count,indices_of_indices_1d_unmasked[1]] = -4.0/(2.0*step_x)
            Hx[count,indices_of_indices_1d_unmasked[2]] = +3.0/(2.0*step_x)
        elif this_type[1] == +1: 
            indices_tmp = np.ravel_multi_index([(i, i), (j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked[0]] = -1.0/step_x
            Hx[count,indices_of_indices_1d_unmasked[1]] = +1.0/step_x
        elif this_type[1] == -1: 
            indices_tmp = np.ravel_multi_index([(i, i), (j-1, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked[0]] = -1.0/step_x
            Hx[count,indices_of_indices_1d_unmasked[1]] = +1.0/step_x  

    return Hy, Hx


def diff_2nd_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hxx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hyy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 2nd diff first
        if unmask[i-1,j] and unmask[i+1,j]: #2nd central diff
            indices_tmp = np.ravel_multi_index([(i-1, i, i+1), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        elif unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        elif unmask[i-1,j] and unmask[i-2,j]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i-1, i-2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2     
        #if 2nd diff fails, just do nothing, so that the 2nd diff along y-directon is 0
        else:
            pass 
        #------check x-direction  
        #try 2nd diff;
        if unmask[i, j-1] and unmask[i, j+1]: #2nd central diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j-1, j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        elif unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        elif unmask[i, j-1] and unmask[i, j-2]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j-1, j-2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2  
        #if 2nd diff fails, just do nothing, so that the 2nd diff along x-directon is 0
        else:
            pass

    return Hyy, Hxx


def diff_2nd_reg_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hxx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hyy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 2nd diff first 
        if unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2      
        #if 2nd diff fails, try 1st
        elif unmask[i+1,j]: #1st forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_y   
        else:
            Hyy[count,count] = 1.0 

        #------check x-direction  
        #try 2nd diff;
        if unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        #if 2nd diff fails, try 1st
        elif unmask[i, j+1]: #1st forward diff
            indices_tmp = np.ravel_multi_index([(i, i), (j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_x   
        else:
            Hxx[count,count] = 1.0 

    return Hyy, Hxx


def diff_2nd_reg_nopad_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hxx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hyy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 2nd diff first 
        if unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2      
        else:
            pass

        #------check x-direction  
        #try 2nd diff;
        if unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        else:
            pass

    return Hyy, Hxx


def diff_2nd_reg_nopad_operator_from_mask_koopman(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hxx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hyy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 2nd diff first 
        if unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2 
        elif unmask[i-1,j] and unmask[i-2,j]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i-1, i-2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2       
        else:
            print('no-padding pixel, y-direction: ',i,j)
            pass

        #------check x-direction  
        #try 2nd diff;
        if unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2
        elif unmask[i, j-1] and unmask[i, j-2]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j-1, j-2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2     
        else:
            print('no-padding pixel, x-direction: ',i,j)
            pass

    return Hyy, Hxx


def diff_4th_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 4th differential operator matrix Hx_4th and Hy_4th.
    """ 
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 4th diff first
        if unmask[i-2,j] and unmask[i-1,j] and unmask[i+1,j] and unmask[i+2,j]: #4th central diff
            indices_tmp = np.ravel_multi_index([(i-2, i-1, i, i+1, i+2), (j, j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_y**4
        elif unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j] and unmask[i+4,j]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3, i+4), (j, j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_y**4   
        elif unmask[i-1,j] and unmask[i-2,j] and unmask[i-3,j] and unmask[i-4,j]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i-1, i-2, i-3, i-4), (j, j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_y**4
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3), (j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_y**3   
        elif unmask[i-1,j] and unmask[i-2,j] and unmask[i-3,j]: #3rd backward diff
            indices_tmp = np.ravel_multi_index([(i, i-1, i-2, i-3), (j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -3.0, 3.0, -1.0])/step_y**3        
        #if 3rd diff fails, try 2nd diff;
        elif unmask[i-1,j] and unmask[i+1,j]: #2nd central diff
            indices_tmp = np.ravel_multi_index([(i-1, i, i+1), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        elif unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        elif unmask[i-1,j] and unmask[i-2,j]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i-1, i-2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2  
        #if 2nd diff fails, try 1st diff; we don't need to try 1st central if 2nd central fails
        elif unmask[i+1,j]: #1st forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_y   
        elif unmask[i-1,j]: #1st backward diff
            indices_tmp = np.ravel_multi_index([(i, i-1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -1.0])/step_y 
        #if 1st fails, set the zero order drawback
        else:
            Hy[count,count] = 1.0 

        #------check x-direction
        #try 4th diff first
        if unmask[i, j-2] and unmask[i, j-1] and unmask[i, j+1] and unmask[i, j+2]: #4th central diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i, i), (j-2, j-1, j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_x**4
        elif unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3] and unmask[i, j+4]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i, i), (j, j+1, j+2, j+3, j+4)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_x**4   
        elif unmask[i, j-1] and unmask[i, j-2] and unmask[i, j-3] and unmask[i, j-4]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i, i), (j, j-1, j-2, j-3, j-4)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_x**4
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i), (j, j+1, j+2, j+3)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_x**3   
        elif unmask[i, j-1] and unmask[i, j-2] and unmask[i, j-3]: #3rd backward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i), (j, j-1, j-2, j-3)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -3.0, 3.0, -1.0])/step_x**3        
        #if 3rd diff fails, try 2nd diff;
        elif unmask[i, j-1] and unmask[i, j+1]: #2nd central diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j-1, j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        elif unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        elif unmask[i, j-1] and unmask[i, j-2]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j-1, j-2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2  
        #if 2nd diff fails, try 1st diff; we don't need to try 1st central if 2nd central fails
        elif unmask[i, j+1]: #1st forward diff
            indices_tmp = np.ravel_multi_index([(i, i), (j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_x   
        elif unmask[i, j-1]: #1st backward diff
            indices_tmp = np.ravel_multi_index([(i, i), (j, j-1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -1.0])/step_x  
        #if 1st fails, set the zero order drawback
        else:
            Hx[count,count] = 1.0

    return Hy, Hx


def diff_4th_reg_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 4th differential operator matrix Hx_4th and Hy_4th.
    different from the `diff_4th_operator_from_mask`, we only use the `forward differential` in this fucntion
    """ 
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 4th diff first
        if unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j] and unmask[i+4,j]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3, i+4), (j, j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_y**4   
        #if 4th diff fails, try 3rd diff; 
        elif unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3), (j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_y**3          
        #if 3rd diff fails, try 2nd diff;   
        elif unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        #if 2nd diff fails, try 1st diff; 
        elif unmask[i+1,j]: #1st forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_y   
        #if 1st fails, set the zero order drawback
        else:
            Hy[count,count] = 1.0 

        #------check x-direction
        #try 4th diff first
        if unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3] and unmask[i, j+4]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i, i), (j, j+1, j+2, j+3, j+4)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_x**4   
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i), (j, j+1, j+2, j+3)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_x**3       
        #if 3rd diff fails, try 2nd diff;
        elif unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        #if 2nd diff fails, try 1st diff; we don't need to try 1st central if 2nd central fails
        elif unmask[i, j+1]: #1st forward diff
            indices_tmp = np.ravel_multi_index([(i, i), (j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_x   
        #if 1st fails, set the zero order drawback
        else:
            Hx[count,count] = 1.0

    return Hy, Hx


def diff_4th_reg_nopad_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 4th differential operator matrix Hx_4th and Hy_4th.
    different from the `diff_4th_operator_from_mask`, we only use the `forward differential` in this fucntion
    """ 
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 4th diff first
        if unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j] and unmask[i+4,j]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3, i+4), (j, j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_y**4   
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3), (j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_y**3          
        #if 3rd diff fails, try 2nd diff;   
        elif unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        #if 2nd diff fails, do noting, no padding to lower order!
        else:
            pass

        #------check x-direction
        #try 4th diff first
        if unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3] and unmask[i, j+4]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i, i), (j, j+1, j+2, j+3, j+4)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_x**4   
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i), (j, j+1, j+2, j+3)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_x**3       
        #if 3rd diff fails, try 2nd diff;
        elif unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        #if 2nd diff fails, do noting, no padding to lower order!
        else:
            pass

    return Hy, Hx

    
class SparseDpsiGrid(object):
    def __init__(self, mask, dpix_data, shape_2d_dpsi=(30,30)):
        """
        This class represent the potential correction (Dpsi) grid,
        usually sparser than the native ccd image grid (or data grid).

        Parameters
        ----------
        mask: a bool array represents the data mask, which typically marks an annular-like region.
        dpix_data: the pixel size in arcsec for the native ccd image data.
        dpsi_shape_2d: the shape of the sparser potential correction grid before the mask
        """
        self.mask_data = mask
        self.dpix_data = dpix_data
        self.shape_2d_dpsi = shape_2d_dpsi

        grid_data = al.Grid2D.uniform(shape_native=mask.shape, pixel_scales=dpix_data, sub_size=1)
        self.xgrid_data = np.array(grid_data.native[:,:,1])
        self.ygrid_data = np.array(grid_data.native[:,:,0])

        xmin, xmax = self.xgrid_data.min()-0.5*dpix_data, self.xgrid_data.max()+0.5*dpix_data
        ymin, ymax = self.ygrid_data.min()-0.5*dpix_data, self.ygrid_data.max()+0.5*dpix_data
        self.image_bound = [xmin, xmax, ymin, ymax]

        self.dpix_dpsi = float((xmax-xmin)/shape_2d_dpsi[0])
        grid_dpsi = al.Grid2D.uniform(shape_native=shape_2d_dpsi, pixel_scales=self.dpix_dpsi, sub_size=1)
        self.xgrid_dpsi = np.array(grid_dpsi.native[:,:,1])
        self.ygrid_dpsi = np.array(grid_dpsi.native[:,:,0])

        self.grid_1d_from_mask()
        self.get_sparse_box_center()
        self.pair_data_dpsi_pixel()
        self.get_dpsi2data_mapping()
        self.get_gradient_operator_data()
        self.get_gradient_operator_dpsi()
        self.get_diff_2nd_operator_dpsi()
        self.get_hamiltonian_operator_data()
        self.get_hamiltonian_operator_dpsi()
        self.get_diff_4th_reg_operator_dpsi()
        self.get_diff_2nd_reg_operator_dpsi()


    def mask_dpsi_from_data(self):
        self.mask_dpsi = np.ones(self.shape_2d_dpsi).astype('bool')

        for i in range(self.shape_2d_dpsi[0]):
            for j in range(self.shape_2d_dpsi[1]):
                dist = (self.xgrid_data-self.xgrid_dpsi[i,j])**2 + (self.ygrid_data-self.ygrid_dpsi[i,j])**2
                dist = np.sqrt(dist)
                if_array_eq = np.isclose(dist, dist.min(), rtol=1e-05, atol=1e-08, equal_nan=False)
                min_indices = np.where(if_array_eq)
                if np.any((~self.mask_data)[min_indices]):
                    self.mask_dpsi[i,j] = False


    def grid_1d_from_mask(self):
        """
        Get the 1d data/dpsi grid via the mask (self.xgrid_data_1d and self.xgrid_dpsi_1d)
        Also save the corresponding 1d-indices `indices_1d_data` and `indices_1d_dpsi`
        for example,
        self.data_xgrid_1d = self.data_xgrid.flatten()[self.indices_1d_data]
        """
        self.indices_1d_data = np.where((~self.mask_data).flatten())[0]
        self.xgrid_data_1d = self.xgrid_data.flatten()[self.indices_1d_data]
        self.ygrid_data_1d = self.ygrid_data.flatten()[self.indices_1d_data]

        self.mask_dpsi_from_data()
        self.mask_dpsi = clean_mask(self.mask_dpsi) #clean the dpsi_mask, also remove the exposed pixels
        self.indices_1d_dpsi = np.where((~self.mask_dpsi).flatten())[0]
        self.xgrid_dpsi_1d = self.xgrid_dpsi.flatten()[self.indices_1d_dpsi]
        self.ygrid_dpsi_1d = self.ygrid_dpsi.flatten()[self.indices_1d_dpsi]           


    def show_grid(self, output_file='grid.png'):
        plt.figure(figsize=(5,5))
        plt.plot(self.xgrid_data.flatten(), self.ygrid_data.flatten(), '*', color='black')
        plt.plot(self.xgrid_dpsi.flatten(), self.ygrid_dpsi.flatten(), '*', color='red')
        plt.plot(self.xgrid_data_1d, self.ygrid_data_1d, 'o', color='black')
        plt.plot(self.xgrid_dpsi_1d, self.ygrid_dpsi_1d, 'o', color='red')
        plt.plot(self.sparse_box_xcenter.flatten(), self.sparse_box_ycenter.flatten(), '+', color='blue')
        plt.plot(self.sparse_box_xcenter_1d, self.sparse_box_ycenter_1d, '+', color='red')
        plt.savefig(output_file)


    def get_sparse_box_center(self):
        n1, n2 = self.shape_2d_dpsi
        n1-=1
        n2-=1
        sparse_box_center = al.Grid2D.uniform(shape_native=(n2,n2), pixel_scales=self.dpix_dpsi, sub_size=1)
        self.sparse_box_xcenter = np.array(sparse_box_center.native[:,:,1]) #2d sparse box center x-grid
        self.sparse_box_ycenter = np.array(sparse_box_center.native[:,:,0]) #2d sparse box center y-grid

        self.mask_sparse_box = np.ones((n1, n2)).astype('bool')
        for i in range(n1):
            for j in range(n2):
                if (~self.mask_dpsi[i,j]) and (~self.mask_dpsi[i+1,j]) and (~self.mask_dpsi[i,j+1]) and ((~self.mask_dpsi[i+1,j+1])):
                    self.mask_sparse_box[i,j] = False

        self.indices_1d_sparse_box = np.where((~self.mask_sparse_box).flatten())[0]
        self.sparse_box_xcenter_1d = self.sparse_box_xcenter.flatten()[self.indices_1d_sparse_box]
        self.sparse_box_ycenter_1d = self.sparse_box_ycenter.flatten()[self.indices_1d_sparse_box] 


    def pair_data_dpsi_pixel(self):
        """
        pair the data grid to dpsi grid.
	    self.data_dpsi_pair_info: shape [n_unmasked_data_pixels, 2, 4], save the information how to interpolate `image` defined on 
        the coarser `dpsi grid` to finner `data grid`. 
        For exmaple:
	    self.data_dpsi_pair_info[0, 0, :]: the 1d indices of the paried (nearest) dpsi box for the first unmaksed data pixels.
	    self.data_dpsi_pair_info[0, 1, :]: the interpolation weight for each corner point of the box for the first unmaksed data pixels.
        """
        self.data_dpsi_pair_info = np.zeros((len(self.indices_1d_data), 2, 4))

        for count, item in enumerate(self.indices_1d_data):
            this_x_data = self.xgrid_data.flatten()[item]
            this_y_data = self.ygrid_data.flatten()[item]
            dist = np.sqrt((self.sparse_box_xcenter_1d-this_x_data)**2 + (self.sparse_box_ycenter_1d-this_y_data)**2)
            id_nearest_sparse_box = self.indices_1d_sparse_box[np.argmin(dist)] #this data pixel pairs with sparse box with 1d-index of `id_nearest_sparse_box`

            i,j = np.unravel_index(id_nearest_sparse_box, shape=self.mask_sparse_box.shape) #2d indices of nearest sparse box center
            #sparse_box_corners_x: [top-left,top-right, bottom-left, bottom-right] corner x-positions 
            sparse_box_corners_x = [self.xgrid_dpsi[i,j], self.xgrid_dpsi[i,j+1], self.xgrid_dpsi[i+1,j], self.xgrid_dpsi[i+1,j+1]] 
            sparse_box_corners_y = [self.ygrid_dpsi[i,j], self.ygrid_dpsi[i,j+1], self.ygrid_dpsi[i+1,j], self.ygrid_dpsi[i+1,j+1]]
            weight_sparse_box_corners = linear_weight_from_box(sparse_box_corners_x, sparse_box_corners_y, position=(this_y_data, this_x_data))

            indices_tmp = [j+self.shape_2d_dpsi[0]*i, j+1+self.shape_2d_dpsi[0]*i, j+self.shape_2d_dpsi[0]*(i+1), j+1+self.shape_2d_dpsi[0]*(i+1)]
            indices_of_indices_1d_dpsi = [np.where(self.indices_1d_dpsi == item)[0][0] for item in indices_tmp]
            indices_of_indices_1d_dpsi = np.array(indices_of_indices_1d_dpsi, dtype='int64')

            self.data_dpsi_pair_info[count, 0, :] = indices_of_indices_1d_dpsi[:]
            self.data_dpsi_pair_info[count, 1, :] = weight_sparse_box_corners[:]            
            

    def get_dpsi2data_mapping(self):
        """
        This function mapping a unmasked vector defined on coarser dpsi grid (shape: [n_unmasked_dpsi_pixels,]), 
        to a new unmasked vector defined on finner data grid (shape: [n_unmasked_data_pixels,]).

        return a matrix, with a shape of [n_unmasked_data_pixels, n_unmasked_dpsi_pixels]
        """
        self.map_matrix = np.zeros((len(self.indices_1d_data), len(self.indices_1d_dpsi)))

        for id_data in range(len(self.indices_1d_data)):
            box_indices = (self.data_dpsi_pair_info[id_data, 0, :]).astype('int64')
            box_weights = (self.data_dpsi_pair_info[id_data, 1, :])
            self.map_matrix[id_data, box_indices] = box_weights[:]


    def get_gradient_operator_data(self):
        self.Hy_data, self.Hx_data = diff_1st_operator_from_mask(self.mask_data, self.dpix_data)


    def get_gradient_operator_dpsi(self):
        self.Hy_dpsi, self.Hx_dpsi = diff_1st_operator_from_mask(self.mask_dpsi, self.dpix_dpsi)


    def get_diff_4th_reg_operator_dpsi(self):
        self.Hy_dpsi_4th_reg, self.Hx_dpsi_4th_reg = diff_4th_reg_nopad_operator_from_mask(self.mask_dpsi) #for reg matrix, set dpix to 1


    def get_diff_2nd_operator_dpsi(self):
        self.Hy_dpsi_2nd, self.Hx_dpsi_2nd = diff_2nd_operator_from_mask(self.mask_dpsi, self.dpix_dpsi)


    def get_diff_2nd_reg_operator_dpsi(self):
        self.Hy_dpsi_2nd_reg, self.Hx_dpsi_2nd_reg = diff_2nd_reg_nopad_operator_from_mask_koopman(self.mask_dpsi) #for reg matrix, set dpix to 1, to be consistent to suyu06 eq.A.3
        # self.Hy_dpsi_2nd_reg, self.Hx_dpsi_2nd_reg = diff_2nd_reg_nopad_operator_from_mask(self.mask_dpsi) #for reg matrix, set dpix to 1, to be consistent to suyu06 eq.A.3
        # self.Hy_dpsi_2nd_reg, self.Hx_dpsi_2nd_reg = diff_2nd_reg_operator_from_mask(self.mask_dpsi) #for reg matrix, set dpix to 1, to be consistent to suyu06 eq.A.3


    def get_hamiltonian_operator_data(self):
        self.Hyy_data, self.Hxx_data = diff_2nd_operator_from_mask(self.mask_data, self.dpix_data)
        self.hamiltonian_data = self.Hxx_data + self.Hyy_data


    def get_hamiltonian_operator_dpsi(self):
        self.Hyy_dpsi, self.Hxx_dpsi = diff_2nd_operator_from_mask(self.mask_dpsi, self.dpix_dpsi)
        self.hamiltonian_dpsi = self.Hxx_dpsi + self.Hyy_dpsi

        
if __name__ == '__main__':
    pass
    
