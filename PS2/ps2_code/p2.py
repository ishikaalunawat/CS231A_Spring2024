import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # TODO: Implement this method!
    U, S, V = np.linalg.svd(F.T)
    # divide for homogeneous coordinate
    epipole = V[-1]/V[-1, -1]
    return epipole

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    # TODO: Implement this method!
    # get T
    h, w = im.shape[:2]
    T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
    epipole_t = T @ e
    # normalize the epipole
    epipole_t /= epipole_t[2]

    # get R
    alpha = 1 if epipole_t[0]>=0 else -1
    denom = np.linalg.norm(epipole_t[:2])

    R = np.array([[alpha * epipole_t[0]/denom, alpha*epipole_t[1]/denom, 0], 
                  [-alpha* epipole_t[1]/denom, alpha*epipole_t[0]/denom, 0], 
                  [0, 0, 1]])
    epipole_t_r = R @ epipole_t

    # get H
    G = np.array([[1, 0, 0],[0, 1, 0],[-1/epipole_t_r[0], 0, 1]])
    H = np.linalg.inv(T) @ G @ R @ T
    return H

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    def skew(x):
        return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
    
    H2 = compute_H(e2, im2)
    M = skew(e2) @ F + e2[:, None] @ np.array([1., 1., 1.])[None, :]
    
    # get p^ and p'^
    p_cap = (H2 @ M @ points1.T).T
    p_dash_cap = (H2 @ points2.T).T

    # get p^_n and p'^_n
    p_cap = p_cap / p_cap[:, 2][:, None]
    p_dash_cap /= p_dash_cap[:, 2][:, None]

    a = np.linalg.lstsq(p_cap, p_dash_cap[:, 0], rcond=None)[0]
    HA = np.array([[a[0], a[1], a[2]], 
                   [0, 1, 0], 
                   [0, 0, 1]])
    H1 = HA @ H2 @ M
    return H1, H2



if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
