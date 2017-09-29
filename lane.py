import cv2
import matplotlib.pyplot as plt
import numpy as np

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import image


# constants
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


def _draw_polygon(img, pts):
    """Draw a polygon on an image
    
    Parameters
    ----------
    img : numpy.ndarray
        the image on which we will draw lines
    pts : list
        Nx2 matrix of (x,y) points in the image that define the polygon
        
    Returns
    -------
    numpy.ndarray
        the image with a polygon drawn on it

    """
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
        
    line_image = np.copy(img)*0
    for i in range(len(pts)):
        x1, y1 = pts[i-1]
        x2, y2 = pts[i]
        cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    
    return cv2.addWeighted(img, 0.8, line_image, 1, 0)


def _window_mask(img, row0, row1, col0, col1):
    """Apply a ones mask to an image
    
    Parameters
    ----------
    img : numpy.ndarray
        the image on which the mask is being generated
    row0 : int
        the starting row of the mask
    row1 : int
        the final row of the mask
    col0 : int
        the starting row of the mask
    col1 : int
        the final row of the mask
    
    Returns
    -------
    output : numpy.ndarray
        a masked binary image
    
    """
    output = np.zeros(img.shape[:2], dtype=np.uint8)
    output[row0:row1, col0:col1] = 1
    
    return output


class Lane(object):
    def __init__(self, img=None, imgpath=None):
        """Initialize a `Lane` object
        
        Parameters
        ----------
        img : numpy.ndarray, None
            the image from which we are finding lanes
        imgpath : str, None
            the path to the `img` image

        """
        assert img is not None or imgpath is not None, "`img` or `imgpath` must be specified"
        
        self.imgpath = imgpath
        
        # the image from which we are finding lanes
        if img is not None:
            self.img = img
        else:
            self.img = cv2.imread(imgpath)
        
        # the points fitted to get the left and right lane curves
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        
        # the coefficients of the fitted polynomials
        self.left_fit_pix = None
        self.right_fit_pix = None
        self.left_fit_m = None
        self.right_fit_m = None
        
        # radii of curvature (in meters and pixels)
        self.left_rad_curv_m = None
        self.left_rad_curv_pix = None
        
        # distance from center of the lane (in meters, + values indicate to the right of center)
        self.offcenter = None
        
        # a list of the last `Lanes` objects (`self.prev[0]` is the one immediately before)
        self.prev = None
        
        # was the line detected in the last iteration?
        self.detected = False
        
        # x values of the last n fits of the line
        self.recent_xfitted = []
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        
        #x values for detected line pixels
        self.allx = None
        
        #y values for detected line pixels
        self.ally = None
    
    # =============================================== #
    #                                                 #
    #                 Image transforms                #
    #                                                 #
    # =============================================== #
        
    def get_undistorted(self, mtx, dist, src=None, outfile=None):
        """Use `cv2.warpPerspective()` to warp the image to a top-down view
        
        Parameters
        ----------        
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        src : numpy.ndarray, None
            Nx2 matrix of (x,y) points in the original (undistorted) image
        outfile : str, None
            path where the undistorted image is to be saved
        
        Returns
        -------
        undistorted : numpy.ndarray
            the undistorted image (possibly with lines drawn on it)
        
        """
        # undistort the image
        undistorted = cv2.undistort(self.img, mtx, dist, None, mtx)
        
        # draw lines on the original image
        if src is not None:
            undistorted = _draw_polygon(undistorted, src)
        
        # save the undistorted image
        if outfile is not None:
            cv2.imwrite(outfile, undistorted)
            
        return undistorted
    
    def get_perspective(self, mtx, dist, M, dst=None, outfile=None):
        """Use `cv2.warpPerspective()` to warp the image to a top-down view
        
        Parameters
        ----------        
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix
        dst : numpy.ndarray, None
            Nx2 matrix of (x,y) points in the perspective image
        outfile : str, None
            path where the perspective image is to be saved
        
        Returns
        -------
        perspective : numpy.ndarray
            the top-down perspective image (possibly with lines drawn on it)
            
        """
        # undistort the image
        undistorted = self.get_undistorted(mtx, dist)
        
        # use cv2.warpPerspective() to warp the image to a top-down view
        img_size = self.img.shape[1::-1]
        perspective = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)
        
        # draw lines on the perspective image
        if dst is not None:
            perspective = _draw_polygon(perspective, dst)
        
        # save the perspective image
        if outfile is not None:
            cv2.imwrite(outfile, perspective)
        
        return perspective

    def get_binary(self, mtx, dist, outfile=None):
        """Generate a binary image of the lines in an image
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        outfile : str, None
            path where the binary image is to be saved
        
        Returns
        -------
        binary : numpy.ndarray
            a binary image
            
        """        
        # undistort the image
        undistorted = self.get_undistorted(mtx, dist)
        
        # get channels
        hls = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        # L channel -- threshold the gradient in the x direction
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
        sobel_x = np.absolute(sobel_x)
        sobel_x = np.uint8(255*sobel_x/np.max(sobel_x))
        
        # create a binary image
        binary = np.zeros(self.img.shape[:2], dtype=np.uint8)
        binary[(10 <= sobel_x) | (170 <= s_channel)] = 1
        
        # save the binary image
        if outfile is not None:
            cv2.imwrite(outfile, np.dstack((binary, binary, binary)).astype(np.uint8) * 255)
        
        return binary
        
    def get_binary_perspective(self, mtx, dist, M, outfile=None):
        """Get the top-down perspective of the binary image
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix
        outfile : str, None
            path where the binary perspective image is to be saved
        
        Returns
        -------
        binary_perspective : numpy.ndarray
            the top-down view of the binary image (possibly with lines drawn on it)
            
        """ 
        # get the binary image and get the top-down perspective of it
        binary = self.get_binary(mtx, dist)
        
        # get the top-down perspective of the binary image
        img_size = binary.shape[1::-1]
        binary_perspective = cv2.warpPerspective(binary, M, img_size, flags=cv2.INTER_LINEAR)
        
        # save the binary perspective image
        if outfile is not None:
            cv2.imwrite(outfile, np.dstack((binary_perspective, binary_perspective, binary_perspective)).astype(np.uint8) * 255)
        
        return binary_perspective
    
    # =============================================== #
    #                                                 #
    #                  Line fitting                   #
    #                                                 #
    # =============================================== #
    
    def fit_lines(self, mtx, dist, M, margin_naive, margin_prior, window_width, window_height, minsum, d=2):
        """Find left and right (x,y) points on the binary perspective image and fit a curve to them
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix        
        margin_naive : int
            the maximum difference between centroids from one level of the image to the next when not using prior information
        margin_prior : int
            the maximum difference between centroids from one level of the image to the next when using prior information
        window_width : int
            the width of the convolution used for computing windowed sums
        window_height : int
            we are dividing the image into sections of this height and finding their centroids
        minsum : int, float
            if the sum over the computed centroid window is < `minsum`, disregard it
        d : int
            the degree of the fitted polynomial
        
        """
        assert window_width % 2 == 0, '`window_width` must be even'
        
        binary_perspective = self.get_binary_perspective(mtx, dist, M)
        
        # use the last `Lanes` object
        if self.prev is not None:
            # the most recent `Lanes` object
            l0 = self.prev[0]
                   
            # get the indices of nonzero pixels in `binary_perspective`
            nonzero = binary_perspective.nonzero()
            nonzero_x = np.array(nonzero[1])
            nonzero_y = np.array(nonzero[0])
            
            # boolean list which is True when the corresponding `nonzero_x` is within +/- `margin_prior` of the previous left line
            left_lane_inds = ((nonzero_x > (np.polyval(l0.left_fit_pix, nonzero_y) - margin_prior)) &
                              (nonzero_x < (np.polyval(l0.left_fit_pix, nonzero_y) + margin_prior)))

            # boolean list which is True when the corresponding `nonzero_x` is within +/- `margin_prior` of the previous right line
            right_lane_inds = ((nonzero_x > (np.polyval(l0.right_fit_pix, nonzero_y) - margin_prior)) &
                               (nonzero_x < (np.polyval(l0.right_fit_pix, nonzero_y) + margin_prior)))
                               
            # set the left and right (x,y) points
            self.left_x = nonzero_x[left_lane_inds]
            self.left_y = nonzero_y[left_lane_inds]
            self.right_x = nonzero_x[right_lane_inds]
            self.right_y = nonzero_y[right_lane_inds]
        
        # don't use any prior information
        else:            
            window_centroids = self.find_window_centroids(mtx, dist, M, window_width, window_height, margin_naive, minsum)

        # fit the left lane line
        if len(self.left_y) > 0:
            self.left_fit_pix = np.polyfit(self.left_y, self.left_x, d)
            self.left_fit_m = np.polyfit(self.left_y * ym_per_pix, self.left_x * xm_per_pix, d)
        elif self.prev is not None:
            self.left_fit_pix = self.prev[0].left_fit_pix
            self.left_fit_m = self.prev[0].left_fit_m
            
        # fit the right lane line
        if len(self.right_y) > 0:
            self.right_fit_pix = np.polyfit(self.right_y, self.right_x, d)
            self.right_fit_m = np.polyfit(self.right_y * ym_per_pix, self.right_x * xm_per_pix, d)
        elif self.prev is not None:
            self.right_fit_pix = self.prev[0].right_fit_pix
            self.right_fit_m = self.prev[0].right_fit_m
        
        # fit the left and right lines (pixels)
        #self.left_fit_pix = np.polyfit(self.left_y, self.left_x, d)
        #self.right_fit_pix = np.polyfit(self.right_y, self.right_x, d)
        
        # fit the left and right lines (meters)
        #self.left_fit_m = np.polyfit(self.left_y * ym_per_pix, self.left_x * xm_per_pix, d)
        #self.right_fit_m = np.polyfit(self.right_y * ym_per_pix, self.right_x * xm_per_pix, d)
        
        # smooth by averaging over the previous `Lane` objects
        self.smooth()
        
        # compute the radii of curvature and offset
        self.get_rad_curv()
        self.get_offset()
        
        self.fix_lines(margin_naive, mtx, dist, M)
    
    def find_window_centroids(self, mtx, dist, M, window_width, window_height, margin_naive, minsum):
        """Find the window centroids, with or without prior information
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix 
        window_width : int
            the width of the convolution used for computing windowed sums
        window_height : int
            we are dividing the image into sections of this height and finding their centroids
        margin_naive : int
            the maximum difference between centroids from one level of the image to the next when not using prior information
        minsum : int, float
            if the sum over the computed centroid window is < `minsum`, disregard it
            
        Returns
        -------
        window_centroids : list
            a list of the centroids on each level: ``(left, right, level)``
        
        """
        assert window_width % 2 == 0, '`window_width` must be even'
        
        binary_perspective = self.get_binary_perspective(mtx, dist, M)
                   
        # get the indices of nonzero pixels in `binary_perspective`
        nonzero = binary_perspective.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])
        
        # constants
        rows, cols = binary_perspective.shape[:2]
        
        # maximum difference between window centroids from one level to the next
        offset = window_width//2
        
        # convolution window --> use higher weights at the center
        window = np.concatenate((np.linspace(0.5, 1.0, window_width/2), np.linspace(1.0, 0.5, window_width/2)))

        # sum the bottom 25% of the rows
        l_sum = np.sum(binary_perspective[int(3*rows/4):,:int(cols/2)], axis=0)
        r_sum = np.sum(binary_perspective[int(3*rows/4):,int(cols/2):], axis=0)
        
        # find the left and right windows with the largest sum and get their centers
        l_center = np.argmax(np.convolve(window,l_sum)) - offset
        r_center = np.argmax(np.convolve(window,r_sum)) - offset + int(cols/2)
        
        window_centroids = [(l_center, r_center, rows-window_height)]

        # Go through each layer looking for max pixel locations    
        for level in range(1, int(rows/window_height)):
            # convolve the window into the vertical slice of the image
            row0 = int(rows - (level+1)*window_height)
            row1 = int(rows - level*window_height)
            image_layer = np.sum(binary_perspective[row0:row1,:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            # find the best left centroid that is within +/- `margin_naive` pixels of the previous centroid
            l_min_index = int(max(l_center + offset - margin_naive, 0))
            l_max_index = int(min(l_center + offset + margin_naive, cols))

            # find the best left centroid that is within +/- `margin_naive` pixels of the previous centroid
            r_min_index = int(max(r_center + offset - margin_naive, 0))
            r_max_index = int(min(r_center + offset + margin_naive, cols))
            
            if minsum < max(conv_signal[l_min_index:l_max_index]) and minsum < max(conv_signal[r_min_index:r_max_index]):
                l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
                r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset 
                
                window_centroids.append((l_center, r_center, row0))
            
        # boolean lists which will be True when the corresponding `nonzero_x` is within +/- `margin_naive` of the previous left/right line
        left_lane_inds = np.array([False] * len(nonzero_x))
        right_lane_inds = np.array([False] * len(nonzero_y))
        
        # find the points (x,y) points inside the windows
        for l_center, r_center, row0 in window_centroids:
            row1 = row0 + window_height
            
            # find the indices of the left and right lane pixels that are inside the window
            left_inside = ((l_center - offset < nonzero_x) & (nonzero_x < l_center + offset) & (row0 <= nonzero_y) & (nonzero_y < row1))
            left_lane_inds[left_inside] = True
            
            right_inside = ((r_center - offset < nonzero_x) & (nonzero_x < r_center + offset) & (row0 <= nonzero_y) & (nonzero_y < row1))
            right_lane_inds[right_inside] = True

        # extract the left points
        self.left_x = nonzero_x[left_lane_inds]
        self.left_y = nonzero_y[left_lane_inds]
        
        # extract the right points
        self.right_x = nonzero_x[right_lane_inds]
        self.right_y = nonzero_y[right_lane_inds]
                
        return window_centroids
        
    def get_rad_curv(self):
        """Get the left and right radii of curvature (in meters and pixels)
        
        """
        # radii of curvature
        rows = self.img.shape[0]
        y0 = rows-1
        
        # radii of curvature (pixels)
        self.left_rad_curv_pix = ((1 + (np.polyval(np.polyder(self.left_fit_pix, 1), y0)**2)**1.5) / np.absolute(np.polyval(np.polyder(self.left_fit_pix, 2), y0)))
        self.right_rad_curv_pix = ((1 + (np.polyval(np.polyder(self.right_fit_pix, 1), y0)**2)**1.5) / np.absolute(np.polyval(np.polyder(self.right_fit_pix, 2), y0)))
        
        # radii of curvature (meters)
        self.left_rad_curv_m = ((1 + (np.polyval(np.polyder(self.left_fit_m, 1), y0*ym_per_pix)**2)**1.5) / np.absolute(np.polyval(np.polyder(self.left_fit_m, 2), y0*ym_per_pix)))
        self.right_rad_curv_m = ((1 + (np.polyval(np.polyder(self.right_fit_m, 1), y0*ym_per_pix)**2)**1.5) / np.absolute(np.polyval(np.polyder(self.right_fit_m, 2), y0*ym_per_pix)))
    
    def get_offset(self):
        """Get the offset of the vehicle from the center of the lane
        
        """
        rows, cols = self.img.shape[:2]
        y0 = rows-1
        
        left = np.polyval(self.left_fit_m, y0*ym_per_pix)
        right = np.polyval(self.right_fit_m, y0*ym_per_pix)
        center = cols / 2 * xm_per_pix
        self.offcenter = center - (left + right)/2
    
    # =============================================== #
    #                                                 #
    #                     Plotting                    #
    #                                                 #
    # =============================================== #
    
    def plot_lines(self, mtx, dist, M, Minv, show_text=True, show_plot=False, outfile=None):
        """Plot the fitted lane lines
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix        
        Minv : numpy.ndarray
            3x3 inverse transformation matrix 
        show_text : bool
            ``if show_text:`` display the radius of curvature and distance from center on the frame       
        show_plot : bool
            ``if show_plot:`` display the resulting image instead of returning it
        outfile : str, None
            path where the plotted lines image is to be saved
        
        """
        undistorted = self.get_undistorted(mtx, dist)
        binary_perspective = self.get_binary_perspective(mtx, dist, M)
        rows, cols = binary_perspective.shape[:2]
        
        # Create an image to draw the lines on
        warp_zero = np.zeros(undistorted.shape[:2], dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # generate the plot points
        plot_y = np.linspace(0, rows-1, rows)
        left_fit_x = np.polyval(self.left_fit_pix, plot_y)
        right_fit_x = np.polyval(self.right_fit_pix, plot_y)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, self.img.shape[1::-1])
             
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        
        # put text on the image
        if show_text:
            fontscale = 1.8
            thickness1 = 10
            thickness2 = 6
            color1 = (0, 0, 0)
            color2 = (255, 255, 255)
                
            cv2.putText(result, 'Left radius of curvature:', (20,60), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color1, thickness1, cv2.LINE_AA)
            cv2.putText(result, 'Left radius of curvature:', (20,60), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color2, thickness2, cv2.LINE_AA)
            cv2.putText(result, '{0:>10.3f} m'.format(self.left_rad_curv_m), (800,60), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color1, thickness1, cv2.LINE_AA)
            cv2.putText(result, '{0:>10.3f} m'.format(self.left_rad_curv_m), (800,60), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color2, thickness2, cv2.LINE_AA)
            
            cv2.putText(result, 'Right radius of curvature:', (20,130), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color1, thickness1, cv2.LINE_AA)
            cv2.putText(result, 'Right radius of curvature:', (20,130), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color2, thickness2, cv2.LINE_AA)
            cv2.putText(result, '{0:>10.3f} m'.format(self.right_rad_curv_m), (800,130), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color1, thickness1, cv2.LINE_AA)
            cv2.putText(result, '{0:>10.3f} m'.format(self.right_rad_curv_m), (800,130), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color2, thickness2, cv2.LINE_AA)
            
            cv2.putText(result, 'Distance from lane center:', (20,200), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color1, thickness1, cv2.LINE_AA)
            cv2.putText(result, 'Distance from lane center:', (20,200), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color2, thickness2, cv2.LINE_AA)
            cv2.putText(result, '{0:>+10.3f} m'.format(self.offcenter), (800,200), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color1, thickness1, cv2.LINE_AA)
            cv2.putText(result, '{0:>+10.3f} m'.format(self.offcenter), (800,200), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color2, thickness2, cv2.LINE_AA)
        
        # save the plotted lines image
        if outfile is not None:
            cv2.imwrite(outfile, result)
        
        if show_plot:   
            plt.imshow(result[:,:,::-1])
            plt.show()
        else:
            return result

    def plot_lines_perspective(self, mtx, dist, M, margin, outfile=None):
        """Plot the fitted lane lines on the binary perspective images
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix        
        margin : int
            points within +/- `margin` of the line will be highlighted
        outfile : str, None
            path where the plotted lines perspective image is to be saved
        
        """
        binary_perspective = self.get_binary_perspective(mtx, dist, M)
        rows, cols = binary_perspective.shape[:2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_perspective, binary_perspective, binary_perspective))*255
        window_img = np.zeros(out_img.shape, dtype=np.uint8)
        
        # Color in left and right line pixels
        out_img[self.left_y, self.left_x] = [0, 0, 255]
        out_img[self.right_y, self.right_x] = [255, 0, 0]
        
        # generate the plot points
        plot_y = np.linspace(0, rows-1, rows)
        left_fit_x = np.polyval(self.left_fit_pix, plot_y)
        right_fit_x = np.polyval(self.right_fit_pix, plot_y)

        # Generate a polygon to illustrate the search window area
        # and recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])        
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        result[np.int_(plot_y), np.int_(left_fit_x), :] = [0, 255, 255]
        result[np.int_(plot_y), np.int_(right_fit_x), :] = [0, 255, 255]
        
        # save the plotted lines perspective image
        if outfile is not None:
            cv2.imwrite(outfile, result)
        
        # show the plot
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(self.get_undistorted(mtx, dist)[:,:,::-1])
        ax1.set_title('Undistorted Original Image', fontsize=30)
        ax2.imshow(result)
        ax2.set_title('Fitted Lanes', fontsize=30)
        ax2.set_xlim(0, cols)
        ax2.set_ylim(rows, 0)
        plt.show()
    
    def plot_window_centroids(self, mtx, dist, M, window_centroids, window_width, window_height, outfile=None):
        """Plot the window centroids
        
        Parameters
        ----------
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix
        window_centroids : list
            a list of the ``(left, right)`` centroids on each level
        warped : numpy.ndarray
            the warped image
        window_width : int
            the width of the convolution used for computing windowed sums
        window_height : int
            the height of the windows for which we have found the centroids
        outfile : str, None
            path where the window centroids image is to be saved
        
        """
        undistorted = self.get_undistorted(mtx, dist)
        binary_perspective = self.get_binary_perspective(mtx, dist, M)
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_perspective, binary_perspective, binary_perspective))*255
        window_img = np.zeros(out_img.shape, dtype=np.uint8)
        
        # Color in left and right line pixels
        out_img[self.left_y, self.left_x] = [255, 0, 0]
        out_img[self.right_y, self.right_x] = [0, 0, 255]
            
        if len(window_centroids) > 0:
            offset = window_width//2
            
            for l_center, r_center, row0 in window_centroids:
                l_mask = _window_mask(binary_perspective, row0, row0 + window_height, l_center - offset, l_center + offset)
                r_mask = _window_mask(binary_perspective, row0, row0 + window_height, r_center - offset, r_center + offset)
                
                window_img[(l_mask == 1) | (r_mask == 1)] = [0, 255, 0]
            
            # overlay the windows on the highlighted binary perspective image
            output = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # if no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((binary_perspective, binary_perspective, binary_perspective))*255, dtype=np.uint8)
        
        # save the window centroids image
        if outfile is not None:
            cv2.imwrite(outfile, output)

        image.show_images(undistorted, 'Undistorted Original Image', output, 'Sliding Windows')
    
    # =============================================== #
    #                                                 #
    #                 Post-Processing                 #
    #                                                 #
    # =============================================== #
    
    def rmse(self, margin, mtx, dist, M):
        """Calculate the root mean squared error for the left and right lines to find which one is better
        
        Parameters
        ----------
        margin : int
            the RMSE is calculated over pixels within `margin` columns of the fitted line
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix
            
        Returns
        -------
        rmse_left : float
            the RMSE of the left fitted line
        rmse_right : float
            the RMSE of the right fitted line
        
        """
        binary_perspective = self.get_binary_perspective(mtx, dist, M)
                   
        # get the indices of nonzero pixels in `binary_perspective`
        nonzero = binary_perspective.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])
        
        # boolean list which is True when the corresponding `nonzero_x` is within +/- `margin` of the fitted left line
        left_lane_inds = ((nonzero_x > (np.polyval(self.left_fit_pix, nonzero_y) - margin)) &
                          (nonzero_x < (np.polyval(self.left_fit_pix, nonzero_y) + margin)))

        # boolean list which is True when the corresponding `nonzero_x` is within +/- `margin` of the fitted right line
        right_lane_inds = ((nonzero_x > (np.polyval(self.right_fit_pix, nonzero_y) - margin)) &
                           (nonzero_x < (np.polyval(self.right_fit_pix, nonzero_y) + margin)))
                           
        # set the left and right (x,y) points
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]
        
        left_calc = np.polyval(self.left_fit_pix, left_y)
        right_calc = np.polyval(self.right_fit_pix, right_y)
        
        rmse_left = np.linalg.norm(left_x - left_calc, 2) / np.sqrt(len(left_calc))
        rmse_right = np.linalg.norm(right_x - right_calc, 2) / np.sqrt(len(right_calc))
        
        return rmse_left, rmse_right
        
    def fix_lines(self, margin, mtx, dist, M):
        """Make sure that the fitted lines radii of curvature are consistent
        
        Parameters
        ----------
        margin : int
            the RMSE is calculated over pixels within `margin` columns of the fitted line
        mtx : numpy.ndarray
            3x3 floating-point camera matrix
        dist : numpy.ndarray
            vector of distortion coefficients: ``(k_1, k_2, p_1, p_2, k_3)``
        M : numpy.ndarray
            3x3 transformation matrix
        
        """
        # parameters in pixels
        y0_p = self.img.shape[0]-1
        x0_left_p = np.polyval(self.left_fit_pix, y0_p)
        x0_right_p = np.polyval(self.right_fit_pix, y0_p)
        dx_p = x0_right_p - x0_left_p
        
        # parameters in meters        
        y0_m = y0_p * ym_per_pix
        x0_left_m = np.polyval(self.left_fit_m, y0_m)
        x0_right_m = np.polyval(self.right_fit_m, y0_m)
        dx_m = x0_right_m - x0_left_m
        
        rmse_left, rmse_right = self.rmse(margin, mtx, dist, M)
        
        if rmse_left < rmse_right:
            # the first derivative
            dx_dy_p = np.polyval(np.polyder(self.left_fit_pix, 1), y0_p)
            dx_dy_m = np.polyval(np.polyder(self.left_fit_m, 1), y0_m)
            
            # correct the radius of curvature and the `a` coefficient
            if np.polyval(np.polyder(self.left_fit_pix, 2), y0_p) > 0:
                right_rad_curv_p = self.left_rad_curv_pix - dx_p
                right_rad_curv_m = self.left_rad_curv_m - dx_m
                
                a_p = (1 + dx_dy_p**2)**1.5 / 2 / right_rad_curv_p
                a_m = (1 + dx_dy_m**2)**1.5 / 2 / right_rad_curv_m
                
            else:
                right_rad_curv_p = self.left_rad_curv_pix + dx_p
                right_rad_curv_m = self.left_rad_curv_m + dx_m
                
                a_p = -(1 + dx_dy_p**2)**1.5 / 2 / right_rad_curv_p
                a_m = -(1 + dx_dy_m**2)**1.5 / 2 / right_rad_curv_m
                
            b_p = dx_dy_p - 2 * a_p * y0_p
            b_m = dx_dy_m - 2 * a_m * y0_m
            
            c_p = x0_right_p - a_p * y0_p**2 - b_p * y0_p
            c_m = x0_right_m - a_m * y0_m**2 - b_m * y0_m
            
            self.right_fit_pix = np.array([a_p, b_p, c_p])
            self.right_fit_m = np.array([a_m, b_m, c_m])
            #self.right_fit_m = np.array([a * xm_per_pix / ym_per_pix**2, b * xm_per_pix / ym_per_pix, c * xm_per_pix])
            
        else:
            # the first derivative
            dx_dy_p = np.polyval(np.polyder(self.right_fit_pix, 1), y0_p)
            dx_dy_m = np.polyval(np.polyder(self.right_fit_m, 1), y0_m)
            
            # correct the radius of curvature and the `a` coefficient
            if np.polyval(np.polyder(self.right_fit_pix, 2), y0_p) > 0:
                left_rad_curv_p = self.right_rad_curv_pix + dx_p
                left_rad_curv_m = self.right_rad_curv_m + dx_m
                
                a_p = (1 + dx_dy_p**2)**1.5 / 2 / left_rad_curv_p
                a_m = (1 + dx_dy_m**2)**1.5 / 2 / left_rad_curv_m
                
            else:
                left_rad_curv_p = self.right_rad_curv_pix - dx_p
                left_rad_curv_m = self.right_rad_curv_m - dx_m
                
                a_p = -(1 + dx_dy_p**2)**1.5 / 2 / left_rad_curv_p
                a_m = -(1 + dx_dy_m**2)**1.5 / 2 / left_rad_curv_m
                
            b_p = dx_dy_p - 2 * a_p * y0_p
            b_m = dx_dy_m - 2 * a_m * y0_m
            
            c_p = x0_left_p - a_p * y0_p**2 - b_p * y0_p
            c_m = x0_left_m - a_m * y0_m**2 - b_m * y0_m
            
            self.left_fit_pix = np.array([a_p, b_p, c_p])
            self.left_fit_m = np.array([a_m, b_m, c_m])
            #self.left_fit_m = np.array([a * xm_per_pix / ym_per_pix**2, b * xm_per_pix / ym_per_pix, c * xm_per_pix])
            
        self.get_rad_curv()
        self.get_offset()
    
    def smooth(self):
        """Smooth the polynomial coefficients
        
        """
        if self.prev is not None:
            self.left_fit_pix = np.mean([l.left_fit_pix for l in self.prev] + [self.left_fit_pix], axis=0)
            self.right_fit_pix = np.mean([l.right_fit_pix for l in self.prev] + [self.right_fit_pix], axis=0)
            self.left_fit_m = np.mean([l.left_fit_m for l in self.prev] + [self.left_fit_m], axis=0)
            self.right_fit_m = np.mean([l.right_fit_m for l in self.prev] + [self.right_fit_m], axis=0)
