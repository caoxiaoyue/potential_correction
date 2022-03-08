import autolens as al
import numpy as np

#4th order regularization matrix-------------
def Hx_4th(grid_shape):
    n1, n2 = grid_shape
    npix = n1*n2
    H_matrix = np.zeros((npix, npix))
    for ii in range(npix):
        H_matrix[ii, ii] = 1.0
        id1, id2 = np.unravel_index(ii, grid_shape)
        if id2+4 <n2:
            neighbor_2_index = id1 * n2 + id2 + 1
            H_matrix[ii, neighbor_2_index] = -4.0
            neighbor_2_index = id1 * n2 + id2 + 2
            H_matrix[ii, neighbor_2_index] = 6.0
            neighbor_2_index = id1 * n2 + id2 + 3
            H_matrix[ii, neighbor_2_index] = -4.0
            neighbor_2_index = id1 * n2 + id2 + 4
            H_matrix[ii, neighbor_2_index] = 1.0
    return H_matrix

def Hy_4th(grid_shape):
    n1, n2 = grid_shape
    npix = n1*n2
    H_matrix = np.zeros((npix, npix))
    for ii in range(npix):
        H_matrix[ii, ii] = 1.0
        id1, id2 = np.unravel_index(ii, grid_shape)
        if id1+4 <n1:      
            neighbor_1_index = (id1 + 1) * n2  + id2
            H_matrix[ii, neighbor_1_index] = -4.0
            neighbor_1_index = (id1 + 2) * n2  + id2
            H_matrix[ii, neighbor_1_index] = 6.0
            neighbor_1_index = (id1 + 3) * n2  + id2
            H_matrix[ii, neighbor_1_index] = -4.0
            neighbor_1_index = (id1 + 4) * n2  + id2
            H_matrix[ii, neighbor_1_index] = 1.0
    return H_matrix

def order_4th_reg_matrix(grid_shape):
    Hx = Hx_4th(grid_shape)
    Hy = Hy_4th(grid_shape)
    return np.matmul(Hx.T, Hx) + np.matmul(Hy.T, Hy)