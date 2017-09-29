import cv2
import matplotlib.pyplot as plt
import numpy as np


def calibrate_camera(calibration_images, nx, ny):
    """Get the camera calibration parameters
    
    **Note:** this code is based on https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
    
    Parameters
    ----------
    calibration_images : list
        a list of images OR image filepaths
    nx : int
        the number of interior corners in the x direction
    ny : int
        the number of interior corners in the y direction
        
    Returns
    -------
    mtx : numpy.ndarray
        3x3 floating-point camera matrix
    dist : numpy.ndarray
        vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    for i, img in enumerate(calibration_images):
        # if necessary, load the image
        if isinstance(img, str):
            img = cv2.imread(img)
            
        # get the shape of the images
        if i == 0:
            img_size = img.shape[1::-1]
        
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    # get the camera calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    return mtx, dist


def show_images(img1, title1, img2=None, title2=None, fontsize=30):
    """Display 1 or 2 images
    
    Parameters
    ----------
    img1 : numpy.ndarray
        image 1 (BGR format)
    title1 : str
        title 1
    img2 : numpy.ndarray, None
        image 2 (BGR format)
    title2 : str, None
        title 2
    fontsize : int
        the font size for the image titles
    
    """
    # Visualize the images
    if img2 is None:
        f, ax1 = plt.subplots(1, 1, figsize=(20,10))
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        
        # show image 2
        if len(img2.shape) == 2:
            ax2.imshow(img2, cmap='gray')
        else:
            ax2.imshow(img2[:,:,::-1])
        ax2.set_title(title2, fontsize=fontsize)
        
    # show image 1
    if len(img1.shape) == 2:
        ax1.imshow(img1, cmap='gray')
    else:
        ax1.imshow(img1[:,:,::-1])
    ax1.set_title(title1, fontsize=30)
    
    plt.show()

