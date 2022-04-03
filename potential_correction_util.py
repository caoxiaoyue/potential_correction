import autolens as al
import numpy as np
from scipy.interpolate import griddata #, CloughTocher2DInterpolator
import grid_util

def source_gradient_from(source_points, source_values, eval_points, cross_size=1e-3):
    """
    Evalulate the source gradient at a series points (`eval_points`) 
    from the source light distribution function.

    The source light distribution function is specified by a series discrete values (`source_values`) 
    defined on irregular grids (`source_points`). Intensity at arbitrary location on source-plane is got
    from interpolation (use scipy.griddata module).

    To evaluate the source gradient, we define a square cross with size specified by `cross_size`
    around the given locations, the x/y direvitive values are got from the simple finite differential

    Input:
    source_points: the locations which we define the input source intensity function. Shape: [N_source_points, 2]
    source_values: source intensity values at a series points. Shape: [N_source_points,]
    eval_points: the locations on which we evaluate the gradient values. Shape: [N_eval_points, 2]
    cross_size: the cross_size to numerically calculate the gradient. In arcsec unit

    Output:
    source_gradient: shape [N_eval_points, 2], source_gradient[5, 0] is the y-direction differential of 6th point.
    """
    grad_points = np.zeros((len(eval_points)*4, 2)) #shape: [N_eval_points*4, 2]; save the intensity values at endpoints of cross.

    for count in range(len(eval_points)):
        #left endpoint
        grad_points[4*count,0] = eval_points[count,0] #y
        grad_points[4*count,1] = eval_points[count,1] - cross_size #x
        #right endpoint
        grad_points[4*count+1,0] = eval_points[count,0] #y
        grad_points[4*count+1,1] = eval_points[count,1] + cross_size #x
        #bottom endpoint
        grad_points[4*count+2,0] = eval_points[count,0] - cross_size #y
        grad_points[4*count+2,1] = eval_points[count,1] #x
        #top endpoint
        grad_points[4*count+3,0] = eval_points[count,0] + cross_size #y
        grad_points[4*count+3,1] = eval_points[count,1] #x

    # grad_points = grad_points.reshape(len(eval_points), 4, 2)
    #Note, eval_points[:,1]--x, eval_points[:,0]--y
    #the points in scipy.griddata are defined in (x,y) order, however autolens use (y,x) order. so we have source_points[:,::-1] here
    eval_values = griddata(source_points[:,::-1], source_values, (grad_points[:,1], grad_points[:,0]), method='linear', fill_value=0.0) #shape: [N_eval_points*4,]
    eval_values = eval_values.reshape(len(eval_points), 4)

    x_diff = (eval_values[:, 1] - eval_values[:, 0])/(2.0*cross_size) #shape: [N_eval_points,]
    y_diff = (eval_values[:, 3] - eval_values[:, 2])/(2.0*cross_size)

    return np.vstack((y_diff, x_diff)).T


def source_gradient_matrix_from(source_gradient):
    """
    Generate the source gradient matrix from the source gradient array (got from the function `source_gradient_from`)

    Input:
    source_gradient: an [N_unmasked_dpsi_points, 2] array. The x/y derivative values at the ray-traced dpsi-grid on the source-plane

    Output:
    source_gradient_matrix: an [N_unmasked_dpsi_points, 2*N_unmasked_dpsi_points] arrary. See equation-9 in our team-document.
    """
    N_unmasked_dpsi_points = len(source_gradient)
    source_gradient_matrix = np.zeros((N_unmasked_dpsi_points, 2*N_unmasked_dpsi_points))

    for count in range(N_unmasked_dpsi_points):
        source_gradient_matrix[count, 2*count:2*count+2] = source_gradient[count,::-1] #See equation-9 in our team-document

    return source_gradient_matrix


def dpsi_gradient_operator_from(Hx_dpsi, Hy_dpsi):
    """
    Accept the x/y differential operator Hx_dpsi and Hy_dps; both shapes are [n_unmased_dpsi_points, n_unmased_dpsi_points]
    Construct the dpsi_gradient_operator with shape [2*n_unmased_dpsi_points, n_unmased_dpsi_points]. 
    see eq-8 in our potential correction document
    """
    n_unmased_dpsi_points = len(Hx_dpsi)
    dpsi_gradient_operator = np.zeros((2*n_unmased_dpsi_points, n_unmased_dpsi_points))

    for count in range(n_unmased_dpsi_points):
        dpsi_gradient_operator[count*2, :] = Hx_dpsi[count, :]
        dpsi_gradient_operator[count*2+1, :] = Hy_dpsi[count, :]

    return dpsi_gradient_operator


def order_4th_reg_matrix(Hx_4th, Hy_4th):
    return np.matmul(Hx_4th.T, Hx_4th) + np.matmul(Hy_4th.T, Hy_4th)


def solve_psi_rescale_factor(fix_psi_values, fix_points, psi_new):
    """
    solve the rescaling factor of psi
    This is necessary to ensure potential correction converge

    Input:
        fix_psi_values (array): the intial lensing potential that we want to keep fixed during potential correction
        fix_points (array): the location of psi_init. shape: (3,2) [[y1,x1], [y2,x2], [y3,x3]]
        psi_new (array): the new lensing potential value given by potential correction algorithm, which will be recaled.
    both array shape are (3,)

    Output:
    return the rescale factor a1,a2 ($\vec{a}$) and c;
    see eq.22 and chapter-2.3 in our document
    """
    A_matrix = np.hstack([fix_points, np.ones(3).reshape(3,1)])
    b_vector = fix_psi_values - psi_new
    a_y, a_x, c = np.linalg.solve(A_matrix, b_vector)
    return a_y, a_x, c


def rescale_psi_map(fix_psi_values, fix_points, psi_new, psi_map, xgrid, ygrid):
    """
    rescale the psi_new potential according to psi_init
    see eq.22 and chapter-2.3 in our document

    Input:
        fix_psi_values (array): the intial lensing potential values that we want to keep fixed during potential correction; shape (3,)
        fix_points (array): the location (cooridinates) of psi_init. shape: (3,2); [(y1,x1), (y2,x2), (y3,x3)]
        psi_new (array): the new lensing potential value given by ptential correction algorithm; shape (3,)
        psi_map: the lensing potential map we want to rescale. A 2d array
        xgrid, ygrid: the x/y coordinates of psi_map, same shape as psi_map

    Output::
        array: the rescaled lensing potential map
    """
    a_y, a_x, c = solve_psi_rescale_factor(fix_psi_values, fix_points, psi_new)
    return a_x*xgrid + a_y*ygrid + c + psi_map


if __name__ == '__main__':
    #An dpsi_gradient_operator test
    grid_data = al.Grid2D.uniform(shape_native=(10,10), pixel_scales=0.1, sub_size=1)
    xgrid_data = grid_data.native[:,:,1]
    ygrid_data = grid_data.native[:,:,0]
    rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
    mask = rgrid>0.25
    grid_obj = grid_util.SparseDpsiGrid(mask, 0.1, shape_2d_dpsi=(5,5))
    grid_obj.show_grid()

    dpsi_gradient_matrix = dpsi_gradient_operator_from(grid_obj.Hx_dpsi, grid_obj.Hy_dpsi)
    # np.savetxt('test/data/dpsi_gradient_matrix.txt', dpsi_gradient_matrix, fmt='%.2f')