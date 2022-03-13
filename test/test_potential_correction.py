import autolens as al
import numpy as np
import grid_util
import potential_correction_util as pcu
import os
import pickle

current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

def test_source_gradient():
    grid_data = al.Grid2D.uniform(shape_native=(200,200), pixel_scales=0.05, sub_size=1)
    xgrid_data = np.array(grid_data.slim[:,1])
    ygrid_data = np.array(grid_data.slim[:,0])

    dpis_points_source_plane = np.vstack([ygrid_data, xgrid_data]).T

    def src_func(x, y):
        return 2*x**2 + 3*y**2 + 2

    eval_points = np.array([(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)])
    source_values = src_func(dpis_points_source_plane[:,1], dpis_points_source_plane[:,0])
    source_gradient = pcu.source_gradient_from(dpis_points_source_plane, source_values, eval_points, cross_size=1e-3)
    
    source_gradient_true = np.array([(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (3.0, 2.0)])
    assert np.isclose(source_gradient, source_gradient_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_source_gradient_matrix():
    source_gradient = np.array([(0.1, 0.2), (-0.1, 2.0), (3.0, 1.5), (3.0, 2.0)])
    source_gradient_matrix = pcu.source_gradient_matrix_from(source_gradient)

    #see eq-9 in our potential correction document
    source_gradient_matrix_true = np.array([
        [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
    ])

    assert np.isclose(source_gradient_matrix, source_gradient_matrix_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()
