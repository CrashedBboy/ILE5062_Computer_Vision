# Camera Calibration

## Intro

Camera calibration is a process which calculates camera's `intrinsic matrix` and `extrinsic matrix` from multiple images. Intrinsic matrix is composed by camera's `focal length` and `offset` for X & Y direction(sensor dimension). Extrinsic matrix is composed by `translation` and `rotation` between the object and the camera(sensor).  

<img src="https://github.com/CrashedBboy/ILE5062_Computer_Vision/raw/master/hw1_camera_calibration/reference/result.jpg" width="80%">  

If we capture `multiple pictures` for a scene by using the `same camera setting` from `various viewing angles and positions`. All the relationships between real world object and 2D image have the `same intrinsic matrix`, and have `different extrinsic matrices`.

## How to

1. Define each features coordinate in `real world space(3D)`, then find these features' coordinate in `image plane(2D)`.
2. find `Homography matrix` for each image
3. find `intrinsics` and `extrinsics` from multiple (at least 6) homography matrices.