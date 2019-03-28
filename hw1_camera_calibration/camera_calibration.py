import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

DATA_SIZE = 10

# number of corner per axis
CORNER_X = 7
CORNER_Y = 7

# Initial declaration
# ------------------------------------------------------------------------------------------------------

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)
# create a (CORNER_X*CORNER_Y) x 3 2D matrix to represent corners' world coordinate value
objp = np.zeros((CORNER_X*CORNER_Y,3), np.float32)
objp[:,:2] = np.mgrid[0:CORNER_X, 0:CORNER_Y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
# [note] some of the images have been rotated
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# used to store the index of images, which have different orientation(vertical <--> horizontal) from others
rotated_index = []

# mapping between "rotated imgpoints/homography_matrices" and "original unrotated unrotated_imgpoints/unrotated_homography_matrices"
# element format: [rotated_homography matrix index, original_homography matrix index]
rotation_map = []
unrotated_imgpoints = []
unrotated_objpoints = []

# Find the 2D corner positions of each image
# ------------------------------------------------------------------------------------------------------

# Make a list of calibration images
# glob is a path library which accepts unix-like path pattern as parameter
images = glob.glob('data/*.jpg')

count = 0

# image set's orientation (horizontal or vertical)
img_horizontal = None

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):

    if count >= DATA_SIZE:
        break

    # Find the chessboard corners (from image's top(y = 0) to down, right to left(x = 0))
    print('finding chessboard corners of', fname)

    img = cv2.imread(fname)

    # check image orientation
    need_rotation = False
    if idx == 0:
        # width - height
        if img.shape[1] - img.shape[0] >= 0: 
            img_horizontal = True
        else:
            img_horizontal = False
    else:
        # check if current image's orientation is different from image set's
        if (img_horizontal and (img.shape[1] - img.shape[0]) < 0) or (not img_horizontal and (img.shape[1] - img.shape[0]) >= 0):
            print('different orientation found')
            need_rotation = True

    # change image's color space to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if need_rotation:

        # rotate image which has different orientation, so that we have all homography matrix in the same orientation,
        # then we are able to calculate the correct intrinsic matrix (camera matrix)
        rotated_gray = np.rot90(gray)

        # returned corners is a (CORNER_X*CORNER_Y) x 1 x 2 matrix, each corner is representd by a 2D matrix with single row
        ret, rotated_corners = cv2.findChessboardCorners(rotated_gray, (CORNER_X, CORNER_Y))

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(rotated_corners)

        ''' Find corners of original image

        we also have to find corners and calculate homography matrix for the original(unrotated) image,
        [WHY?] because we need the original homographys (instead of the rotated image's homography) to calculate the extrinsic matrices,
        which contain the correct translation and rotation
        '''

        rotated_index.append(idx)
        rotation_map.append([idx, len(rotated_index) - 1])

        ret, unrotated_corners = cv2.findChessboardCorners(gray, (CORNER_X, CORNER_Y))

        if ret == True:
            unrotated_objpoints.append(objp)
            unrotated_imgpoints.append(unrotated_corners)
            
    else:

        ret, corners = cv2.findChessboardCorners(gray, (CORNER_X,CORNER_Y))

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    count += 1

# Calculate the homography matrix of (with/without rotation)images
# ------------------------------------------------------------------------------------------------------

# reconstruct corners' image-points and world-points matrices
# for an image, image-points matrix: 49(CORNER_X*CORNER_Y) x 2, world-points matrix: 49(CORNER_X*CORNER_Y) x 3
new_imgpoints = np.array(imgpoints)
new_imgpoints = new_imgpoints.reshape((new_imgpoints.shape[0], (CORNER_X*CORNER_Y), 2))
new_objpoints = np.array(objpoints)

homography_matrices = []

# iterate through each image's obj points & image points
for img_corners_per_image, obj_corners_per_image in zip(new_imgpoints, new_objpoints):

    h_coeffi = np.zeros((CORNER_X * CORNER_Y * 2, 9))

    i = 0
    for corner_img, corner_obj in zip(img_corners_per_image, obj_corners_per_image):
        # corner_img -> [x, y]
        # corner_obj -> [U, V, 0]
        h_coeffi[2 * i, :] = [corner_obj[0], corner_obj[1], 1, 0, 0, 0, (-1)*corner_obj[0]*corner_img[0], (-1)*corner_obj[1]*corner_img[0], (-1)*corner_img[0]]
        h_coeffi[2 * i + 1, :] = [0, 0, 0, corner_obj[0], corner_obj[1], 1, (-1)*corner_obj[0]*corner_img[1], (-1)*corner_obj[1]*corner_img[1], (-1)*corner_img[1]]     
        i += 1

    u, s, vh = np.linalg.svd(h_coeffi, full_matrices=False)

    homography = vh.T[:, -1]

    # make every homography matrix's vectors are in the same direction
    if homography[-1] < 0:
        homography = homography * (-1)

    homography_matrices.append(homography.reshape((3,3)))

# Calculate the homography matrix of original unrotated images which have different orientation
# (algorithm is the same as the algorithm of finding rotated images' homography)
# ------------------------------------------------------------------------------------------------------

new_unrotated_imgpoints = np.array(unrotated_imgpoints)
new_unrotated_imgpoints = new_unrotated_imgpoints.reshape((new_unrotated_imgpoints.shape[0], (CORNER_X*CORNER_Y), 2))
new_unrotated_objpoints = np.array(unrotated_objpoints)

unrotated_homography_matrices = []

for img_corners_per_image, obj_corners_per_image in zip(new_unrotated_imgpoints, new_unrotated_objpoints):

    h_coeffi = np.zeros((CORNER_X * CORNER_Y * 2, 9))

    i = 0
    for corner_img, corner_obj in zip(img_corners_per_image, obj_corners_per_image):
        h_coeffi[2 * i, :] = [corner_obj[0], corner_obj[1], 1, 0, 0, 0, (-1)*corner_obj[0]*corner_img[0], (-1)*corner_obj[1]*corner_img[0], (-1)*corner_img[0]]
        h_coeffi[2 * i + 1, :] = [0, 0, 0, corner_obj[0], corner_obj[1], 1, (-1)*corner_obj[0]*corner_img[1], (-1)*corner_obj[1]*corner_img[1], (-1)*corner_img[1]]     
        i += 1

    u, s, vh = np.linalg.svd(h_coeffi, full_matrices=False)

    homography = vh.T[:, -1]

    if homography[-1] < 0:
        homography = homography * (-1)

    unrotated_homography_matrices.append(homography.reshape((3,3)))

# solve intrinsic matrix from multiple homography matrices (rotated images' homography matrices)
# ------------------------------------------------------------------------------------------------------------------

v = np.zeros((2 * len(homography_matrices), 6))
for i, h in enumerate(homography_matrices):
    v[2*i, :] = [
        h[0,1] * h[0,0],
        h[0,1] * h[1,0] + h[1,1] * h[0,0],
        h[0,1] * h[2,0] + h[2,1] * h[0,0],
        h[1,1] * h[1,0],
        h[1,1] * h[2,0] + h[2,1] * h[1,0],
        h[2,1] * h[2,0]
        ]
    v[2*i+1, :] = [
        h[0,0]**2 - h[0,1]**2,
        2 * (h[0,0] * h[1,0] - h[0,1] * h[1,1]),
        2 * (h[0,0] * h[2,0] - h[0,1] * h[2,1]),
        h[1,0]**2 - h[1,1]**2,
        2 * (h[1,0] * h[2,0] - h[1,1] * h[2,1]),
        h[2,0]**2 - h[2,1]**2
    ]

u, s, vh = np.linalg.svd(v, full_matrices=False)

b = vh.T[:, -1]

# negative diagonal
if (b[0] < 0 or b[3] < 0 or b[5] < 0):
    b = b * (-1)

B = np.array([
    [b[0], b[1], b[2]],
    [b[1], b[3], b[4]],
    [b[2], b[4], b[5]]
    ])

# B = K^(-T) * K^(-1), where K is the intrinsic matrix
# do Cholesky decomposition to solve K

l = np.linalg.cholesky( B )

intrinsic = np.linalg.inv(l.T)

# divide intrinsic by its scale ( [x, y, z] = p * H * [U, V, W], where p is scale, H is homography matrix)
intrinsic = intrinsic / intrinsic[2,2]

# Solving extrinsic matrices: rotation & translation
# ------------------------------------------------------------------------------------------------------------------------------------

'''
When calculate extrinsic matrix of different-orientation image,
we need to use original unrotated image's homography and the "right" intrinsic matrix C,
which C is the intrinsic but:
    1. (f/Sx), (f/Sy) swapped
    2. Ox, Oy swapped
(f is focal length, Sx is "sensor width/pixels number in width", Ox is offset in X direction)
'''

extrinsic_matrices = np.zeros((len(homography_matrices), 6))

for i, h in enumerate(homography_matrices):

    rotated = False
    for m in rotation_map:
        if i == m[0]:
            print("solving", i, "-th extrinsics, replace homography with unrotated one")
            h = unrotated_homography_matrices[m[1]]
            rotated = True
            break

    if rotated:
        # swapping elements related to x/y direction
        unrotated_intrinsic = np.copy(intrinsic)
        swap_tmp = unrotated_intrinsic[0,0]
        unrotated_intrinsic[0,0] = unrotated_intrinsic[1,1]
        unrotated_intrinsic[1,1] = swap_tmp
        swap_tmp = unrotated_intrinsic[0,2]
        unrotated_intrinsic[0,2] = unrotated_intrinsic[1,2]
        unrotated_intrinsic[1,2] = swap_tmp
        intrinsic_inverse = np.linalg.inv(unrotated_intrinsic)
    else:
        intrinsic_inverse = np.linalg.inv(intrinsic)

    lambda_value = 1 / np.linalg.norm(np.matmul(intrinsic_inverse, h[:, 0]))

    r1 = np.matmul(lambda_value * intrinsic_inverse, h[:, 0])
    r2 = np.matmul(lambda_value * intrinsic_inverse, h[:, 1])
    r3 = np.cross(r1, r2)

    t = np.matmul(lambda_value * intrinsic_inverse, h[:, 2])
    
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:,0] = r1
    rotation_matrix[:,1] = r2
    rotation_matrix[:,2] = r3

    r_rodrigues, _ = cv2.Rodrigues(rotation_matrix)

    extrinsic_matrices[i,:] = [r_rodrigues[0], r_rodrigues[1], r_rodrigues[2], t[0], t[1], t[2]]

# Draw camera position on 3D plot
# ---------------------------------------------------------------------------

mtx = intrinsic
extrinsics = extrinsic_matrices

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
    scale_focal, extrinsics, board_width,
    board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
