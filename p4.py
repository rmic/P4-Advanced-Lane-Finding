# Udacity Project 4 : Advanced Lane Finding
#
# Author : Raphael Michel (raph.mic@gmail.com)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

# 1. Computer distortion matrix

nx = 9
ny = 6



def findObjPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    return ret, corners

def distortion(im):
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpoints = []

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)

        # Find the chessboard corners
        ret, corners = findObjPoints(img)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (im.shape[1], im.shape[0]), None, None)
    return mtx, dist

# 2. Un-distort images

def undistort(img, mtx, dist):
    # Use cv2.calibrateCamera() and cv2.undistort()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def sobelxy(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    return sobelx, sobely

def threshold(img, thresh=(0,255)):
    out = np.zeros_like(img)
    out[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return out

# 3. Compute gradients
def sobel(img, kernel, orient='x', thresh=(0,255)):
    sobelx, sobely = sobelxy(img, kernel)
    if orient == 'x':
        abs_sobel = np.absolute(sobelx)
    if orient == 'y':
        abs_sobel = np.absolute(sobely)

    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    return threshold(scaled, thresh)

def magnitude(img, kernel, thresh=(0,255)):
    sobelx, sobely = sobelxy(img, kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    return threshold(gradmag, thresh)

def direction(img, kernel, thresh=(0,np.pi/2)):
    sobelx, sobely = sobelxy(img, kernel)
    # Take the absolute value of the gradient direction,
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return threshold(absgraddir, thresh)


# 4. Compute color transformations

def hls_select(img, chan=2, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:,:,chan]
    return threshold(channel, thresh)

# 5. Combine
def combine(img1, img2):
    out = np.zeros_like(img1)
    out[(img1 == 1) | (img2 == 1)] = 1
    return out


# 6. Transform perspective (bird's eye view)

def warp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

# 7. Identify lane pixels


# 8. Fit polynomial


# 9. Estimate radius of curvature


# 10. Position of the vehicle wrt. center of the lane



## Utils

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def apply_lane_mask(img, l_cut=.48, r_cut=.52, b_cut=.9, t_cut=.62, br_cut=.87, bl_cut=.25):
    imshape = img.shape
    # Define the detection horizon

    vertices = np.array([[(imshape[1]*bl_cut, imshape[0]*b_cut),
                      (imshape[1] * l_cut, imshape[0] * t_cut),
                      (imshape[1] * r_cut, imshape[0] * t_cut),
                      (imshape[1]*br_cut, imshape[0]*b_cut)]], dtype=np.int32)

    return region_of_interest(img, vertices)

## Pipeline



src= np.float32([[241,684],[594,450],[686,450],[1061,684]])
dst= np.float32([[290,720],[290,0],[990,0],[990,720]])


def fit_lines(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    lane_width_px = rightx_base - leftx_base
    lane_center = leftx_base + (lane_width_px / 2)
    # Choose the number of sliding windows
    nwindows = 19
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 80
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))



    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    y_eval = np.max(ploty)
    ym_per_pix = 40 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700 # lane_width_px  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    mean_curverad = (left_curverad + right_curverad) / 2
    offset_center = (640 - lane_center) * xm_per_pix
    #print("{0:.2f} m".format(mean_curverad))
    return left_curverad, right_curverad, left_fitx, right_fitx, ploty, offset_center



def process_image(img):
    undistorted = undistort(img, mtx, dist)


    # Make binary image
    sobx = sobel(undistorted, 5, 'x', thresh=(20, 90))
    hls = hls_select(undistorted, 2, (180, 255))

    binary = combine(sobx, hls)
    binary = apply_lane_mask(binary, l_cut=.45, r_cut=.55, b_cut=.95, t_cut=.6, br_cut=.87, bl_cut=.10)


    # Warp the binary image
    warped_binary = warp(binary, src, dst)

    # Fit lines
    left_curverad, right_curverad, left_fitx, right_fitx, ploty, offset_center = fit_lines(warped_binary)
    # Now our radius of curvature is in meters
    mean_curverad = (left_curverad + right_curverad) / 2


    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, dst, src)
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    curveText = "Curve radius : {0:.2f} m".format(right_curverad)
    centerText = "Distance from center of the lane :{0:.2f} m ".format(offset_center)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, curveText, (10, 650), font, .75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, centerText, (10, 700), font, .75, (255, 255, 255), 2, cv2.LINE_AA)

    return result


img = cv2.imread('test_images/test5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mtx, dist = distortion(img)


clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile("output.mp4", audio=False,fps=25,codec='mpeg4')


print("It works.")
