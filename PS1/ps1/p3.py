# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    def line_coefficients(px, py):
        # for normalized coefficients Ax + By + C = 0
        a = px[1] - py[1]
        b =  py[0] - px[0]
        c = px[0] * py[1] - px[1] * py[0]
        coeffs = np.array([a, b, c])
        return coeffs / np.max(coeffs)
    
    # get coefficients of lines and find intersection
    param_1 = line_coefficients(points[0], points[1])
    param_2 = line_coefficients(points[2], points[3])
    intersection = np.cross(param_1, param_2)

    return intersection[:2] / intersection[2]
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Makes sure to make it so the bottom right element of K is 1 at the end.
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    # function to compute coefficients for vi^T * W * vj = 0
    def compute_coefficients(p1, p2):
        return np.array([
            p1[0] * p2[0] + p1[1] * p2[1],
            p1[0] + p2[0], 
            p1[1] + p2[1], 
            1])

    # form matrix A
    A = np.stack([
        compute_coefficients(vanishing_points[0], vanishing_points[1]),
        compute_coefficients(vanishing_points[0], vanishing_points[2]),
        compute_coefficients(vanishing_points[1], vanishing_points[2])
    ])

    # svd to get w
    _, _, V_T = np.linalg.svd(A)
    w = V_T[-1, :] 

    # form symmetric matrix W
    W = np.array([
        [w[0], 0, w[1]],
        [0, w[0], w[2]], 
        [w[1], w[2], w[3]]
        ])

    # do Cholesky decomposition to find K_T_inv
    K_T_inv = np.linalg.cholesky(W)
    K_inv = K_T_inv.T
    K = np.linalg.inv(K_inv)

    return K/K[2,2]
    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    def line_coefficients(px, py):
        # for normalized coefficients Ax + By + C = 0
        a = px[1] - py[1]
        b =  py[0] - px[0]
        c = px[0] * py[1] - px[1] * py[0]
        coeffs = np.array([a, b, c])
        return coeffs / np.max(coeffs)

    # get line equations for vanishing points
    line1 = line_coefficients(vanishing_pair1[0], vanishing_pair1[1])
    line2 = line_coefficients(vanishing_pair2[0], vanishing_pair2[1])

    # get normals to planes
    n1 = K.T @ line1
    n2 = K.T @ line2

    # angle between planes
    angle = np.arccos(np.dot(n1, n2) / (n1.T @ n1 * n2.T @ n2)) * 180 / np.pi
    return angle
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    def compute_direction_vector(p, K):
        # compute the normalized direction vector corresponding to a vanishing point.
        p_h = np.append(p, 1)  # Convert to homogeneous coordinates
        d = np.linalg.inv(K).dot(p_h)
        return d / np.linalg.norm(d)  # Normalize the direction vector

    # get direction vectors for vanishing points
    d1 = np.array([compute_direction_vector(p, K) for p in vanishing_points1]).T
    d2 = np.array([compute_direction_vector(p, K) for p in vanishing_points2]).T

    # get R using the pseudoinverse of d1 if singular
    R = d2 @ np.linalg.pinv(d1)
    return R
    # END YOUR CODE HERE

'''
TEST_P3
Test function. Do not modify.
'''
def test_p3():
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[1080, 598],[1840, 478],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[4, 878],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))


if __name__ == '__main__':
    test_p3()
