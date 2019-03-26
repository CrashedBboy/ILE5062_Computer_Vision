import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

data_size = 3

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7

# create a (corner_x*corner_y) x 3 2D matrix to represent corners' world coordinate value
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
# glob is a path library which accepts unix-like path pattern as parameter
images = glob.glob('data/*.jpg')

data_index = 0

img_horizontal = None

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):

    if data_index >= data_size:
        break

    img = cv2.imread(fname)

    if idx == 0:
        # width - height
        if img.shape[1] - img.shape[0] >= 0: 
            img_horizontal = True
        else:
            img_horizontal = False

    # change image's color space to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners (from image's top(y = 0) to down, right to left(x = 0))
    print('find the chessboard corners of',fname)

    # returned corners is a (corner_x*corner_y) x 1 x 2 matrix, each corner is representd by a 2D matrix with single row
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y))
    # print('ret:', ret, ', corners:\n', str(corners))

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        data_index += 1

        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        # plt.imshow(img)

'''
#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
# rvec [r1, r2, r3] is a vector related to rodrigues vector [theta, axisX, axisY, axisZ]
# where theta = sqrt(r1^2 + r2^2 + r3^2), [axisX, axisY, axisZ] =  rvec/theta = [r1/theta, r2/theta, r3/theta]

Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
'''

# Write your code here
# ----------------------------------------------------------------------------------

# reconstruct corners' image-points and world-points matrices
# for an image, image-points matrix: 49(corner_x*corner_y) x 2, world-points matrix: 49(corner_x*corner_y) x 3

new_imgpoints = np.array(imgpoints)
new_imgpoints = new_imgpoints.reshape((new_imgpoints.shape[0], (corner_x*corner_y), 2))

new_objpoints = np.array(objpoints)

homography_matrices = []

# iterate through each image's obj points & image points
for img_corners_per_image, obj_corners_per_image in zip(new_imgpoints, new_objpoints):

    h_coeffi = np.zeros((corner_x * corner_y * 2, 9))

    # pick 5 corners to solve homography matrix
    i = 0
    for corner_img, corner_obj in zip(img_corners_per_image, obj_corners_per_image):
        # corner_img -> [x, y]
        # corner_obj -> [U, V, 0]

        h_coeffi[2 * i, :] = [corner_obj[0], corner_obj[1], 1, 0, 0, 0, (-1)*corner_obj[0]*corner_img[0], (-1)*corner_obj[1]*corner_img[0], (-1)*corner_img[0]]
        h_coeffi[2 * i + 1, :] = [0, 0, 0, corner_obj[0], corner_obj[1], 1, (-1)*corner_obj[0]*corner_img[1], (-1)*corner_obj[1]*corner_img[1], (-1)*corner_img[1]]     
        i += 1

    u, s, vh = np.linalg.svd(h_coeffi, full_matrices=False)

    homography = vh.T[:, -1]
    homography_matrices.append(homography.reshape((3,3)))

print("homographys: \n", str(homography_matrices))

# solve intrinsic matrix from multiple homography matrix

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
print("b: \n", str(b))

# negative diagonal
if (b[0] < 0 or b[3] < 0 or b[5] < 0):
    print("b * (-1)")
    b = b * (-1)

B = np.array([
    [b[0], b[1], b[2]],
    [b[1], b[3], b[4]],
    [b[2], b[4], b[5]]
    ])

print("b: \n", str(b))
print("B: \n", str(B))

# B = K^(-T) * K^(-1), where K is the intrinsic matrix
# do Cholesky decomposition to solve K

l = np.linalg.cholesky( B )

print("K^(-T): \n", str(l))

intrinsic = np.linalg.inv(l.T)

print("raw intrinsic: \n", str(intrinsic))

# divide intrinsic by its scale ( [x, y, z] = p * H * [U, V, W], where p is scale, H is homography matrix)
intrinsic = intrinsic / intrinsic[2,2]

print("intrinsic: \n", str(intrinsic))

exit()

# ----------------------------------------------------------------------------------

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
