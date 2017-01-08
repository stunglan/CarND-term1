#%%
# P1 task in python file

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import cv2
import os


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def imagepipeline(image):
    image_gray = grayscale(image)

    # blur
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    image_blur = gaussian_blur(image_gray,kernel_size)

    #canny edge detection
    # Define our parameters for Canny and run it (1,10)
    low_threshold = 50
    high_threshold = 150

    image_canny = canny(image_blur,low_threshold,high_threshold)
    
    # find area of interest
    left_bottom = [50, 539]
    left_top = [460,320]
    right_top = [530,320] 
    right_bottom = [890, 539]
    vertices = np.array([[(left_bottom[0],left_bottom[1]),(left_top[0], left_top[1] ), (right_top[0], right_top[1] ), (right_bottom[0],right_bottom[1])]], dtype=np.int32)
    image_triangle = region_of_interest(image_canny,vertices)
    

    # find lines

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20    #30 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 7 #minimum number of pixels making up a line
    max_line_gap = 1     # maximum gap in pixels between connectable line segments

    image_hough = hough_lines(image_triangle,rho,theta,threshold,min_line_length,max_line_gap)

    image_result = weighted_img(image_hough,image)
    return image_result

def draw_lines1(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def lowery(line):
    print("line:",line)
    return -line[0][1]

def sortlines(lines):
    resultlines = sorted(lines,key=lowery)
    return resultlines
        
        

def plotlines(lines):
    from matplotlib.path import Path
    f,ax = plt.subplots()
    codes = [Path.MOVETO,Path.LINETO]
    for line in lines:
        for x0,y0,x1,y1 in line:
            path = Path([(float(x0),float(-y0)),(float(x1),float(-y1))],codes)
            ax.add_patch(patches.PathPatch(path,color='red',lw=0.5))
    ax.set_xlim(960)
    ax.set_ylim(-540)
    ax.invert_xaxis()
    return

def plotsingleline(ax,x0,y0,x1,y1,color):
    from matplotlib.path import Path
    codes = [Path.MOVETO,Path.LINETO]
    path = Path([(float(x0),float(-y0)),(float(x1),float(-y1))],codes)
    ax.add_patch(patches.PathPatch(path,color=color,lw=0.5))
    return

def removeparalelllines(img,lines):
    blue = [255, 0, 0]
    red = [0, 0, 255]
    green = [0,255, 0]


    lines = sortlines(lines)
    leftlines = []
    rightlines = []
    noofleftlines = 0
    firstline = []
    for line in lines:


        for x0,y0,x1,y1 in line:
            slope = ((y1-y0)/(x1-x0))
#            print('slope: ',slope)
            if slope < 0: # left
                # select the rigthmost of the two first lines
                if noofleftlines == 0:
                    leftlines.append(line)
                elif noofleftlines == 1:
                    if leftlines[0][0][0] < x0: # this line is farther right
                        leftlines[0] = line
                    cv2.line(img,(leftlines[0][0][0],leftlines[0][0][1]),(leftlines[0][0][2],leftlines[0][0][3]), blue, 1)
                    print('FIRST left line ({},{}) ({},{}) slope= {}'.format(leftlines[0][0][0],leftlines[0][0][1],leftlines[0][0][2],leftlines[0][0][3],slope)) 
                else:
                    # after that select the next line hat are more to the rigth than the 
                    if  x0 > leftlines[-1][0][0]: # at least more to the right
                        # check if paralell, replace 
                        if y0 > leftlines[-1][0][3]:
                            leftlines[-1] = line
                            cv2.line(img,(x0,y0),(x1,y1),green,1)
                            print('REPLACE left line ({},{}) ({},{}) slope= {}'.format(x0,y0,x1,y1,slope))
                        # if an extension add
                        elif (y0 < leftlines[-1][0][3]) and (x0 > leftlines[-1][0][2]): # add if next y is further up and x more to the right
                            leftlines.append(line)
                            cv2.line(img,(x0,y0),(x1,y1),blue,1)
                            print('KEEP left line ({},{}) ({},{}) slope= {}'.format(x0,y0,x1,y1,slope)) 
                        else:
                            cv2.line(img,(x0,y0),(x1,y1),red,1)
                            print('THROW left line ({},{}) ({},{}) slope= {}'.format(x0,y0,x1,y1,slope)) 
                noofleftlines += 1
                # after that select the next line that are more to the rigth than the 
            else: # right
                rightlines.append(line)
                #print('right line ({},{}) ({},{}) slope= {}'.format(x0,y0,x1,y1,slope))  

    result_lines = rightlines + leftlines
    plt.imshow(img)
    plt.show()


    return result_lines
    



def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    curated_lines = []
    
    lines = removeparalelllines(img,lines)



    # find average left slope
    # if circa insadie find lowert leftmost, and top rigth
    left_slope = []
    right_slope = []
    for line in lines:
#       print('shape of line: ', line.shape)
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
#            print('slope: ',slope)
            if slope < 0:
                left_slope.append(slope)                
            else:
                right_slope.append(slope)
        



    left_slope = np.average(left_slope)
    right_slope = np.average(right_slope)
#    print('left lanes', left_slope)
#    print('right lanes', right_slope)



    xr0 = 0
    yr0 = 0
    xr1 = 10000
    yr1 = 10000
    
    xl0 = 0
    yl0 = 10000
    xl1 = 10000
    yl1 = 0

    for line in lines:

        for x0,y0,x1,y1 in line:
            slope = ((y1-y0)/(x1-x0))
            if slope > 0: # right
                if np.abs(slope - right_slope) < 0.1:
                    xr0 = max(xr0,x0)
                    yr0 = max(yr0,y0)
                    yr1 = min(yr1,y1)
                    xr1 = min(xr1,x1)
            else:   #left
                if np.abs(slope - left_slope) < 0.1:
                    xl0 = max(xl0,x0)
                    yl0 = min(yl0,y1)
                    xl1 = min(xl1,x1)
                    yl1 = max(yl1,y1)
                    #print('left line ({},{}) ({},{})'.format(x0,y0,x1,y1))    


    #extrapolate all the way down y = 539
    
    if (yl0-yl1) != 0:
        ylstart = 539   
        xlstart = math.floor(((ylstart-yl0)*(xl0-xl1))/(yl0-yl1)) + xl0
        ylend = 331
        xlend = math.floor(((ylend-yl1)*(xl1-xl0))/(yl1-yl0)) + xl1
        slope = (ylend-ylstart)/(xlend-xlstart)
        if abs(slope-left_slope) > 0.5:
            pass
            #print('Differnet slope on the left: ', [xlstart,ylstart,xlend,ylend],[slope,left_slope,slope-left_slope])
        else:
            if (110 > xlstart) or (xlstart > 250):
                pass
                #print('left X out of position: (110-250)', [xlstart,ylstart,xlend,ylend],[slope,left_slope,slope-left_slope])
            else:
                curated_lines.append([[xlstart,ylstart,xlend,ylend]])
    

    if (yr0-yr1) != 0:
        yrstart = 331
        xrstart = math.floor(((yrstart-yr0)*(xr0-xr1))/(yr0-yr1)) + xr0
        yrend = 539
        xrend = math.floor(((yrend-yr1)*(xr1-xr0))/(yr1-yr0)) + xr1
        slope = (yrend-yrstart)/(xrend-xrstart)
        if abs(slope-right_slope) > 0.5:
            pass #noop
            #print('Differnet slope on the right: ', [xrstart,yrstart,xrend,yrend],[slope,right_slope,slope-right_slope])
        else:
            if (750 > xrend) or (xrend > 900):
                pass #noop
                ##print('rigth X out of position: (750-900) ', [xrstart,yrstart,xrend,yrend],[slope,right_slope,slope-right_slope])
            else:
                curated_lines.append([[xrstart,yrstart,xrend,yrend]])
            


    
    curated_lines = np.array(curated_lines,dtype=int)
#    print('shape of lines: ', curated_lines.shape)
#    print(curated_lines)
    
    for line in curated_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, 10)

  

  #reading in an image
print('GO AGAIN ---------')
image = mpimg.imread('test_images/solidYellowLeft.jpg')
image_result = imagepipeline(image)
plt.interactive(False)

plt.figure(1)
plt.imshow(image_result,cmap='gray')
plt.figure(2)
plt.imshow(image_result)
plt.show()

