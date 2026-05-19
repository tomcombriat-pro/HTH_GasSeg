import numpy as np
from scipy import interpolate
from scipy.ndimage import map_coordinates
from skimage.segmentation import flood_fill
from scipy.ndimage import label
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops
from morgana.ImageTools.morphology import (
    anchorpoints, 
    spline, 
    midline, 
    meshgrid
)

# Code modified from Morgana: 
# https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/morphology/computemorphology.py
# https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/straightmorphology/computestraightmorphology.py
# Code Reused under MIT License
# MIT License
# Copyright (c) [2021] [MOrgAna]
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def get_midline(mas,margin = 3):
    mask = mas.copy()
    mask = np.array(mask,dtype = bool)
    strele = disk(margin)
    dilated_mask = binary_dilation(mask,strele)
    dilated_mask = label(dilated_mask)[0]
    props = regionprops(np.ones_like(dilated_mask))
    slice_prop = props[0]["slice"] ## just to get a false slice for computing the anchors
    ma= dilated_mask[slice_prop]
    original_mask_cropped = mask[slice_prop]
    anch = anchorpoints.compute_anchor_points(dilated_mask,slice_prop,1)
    N_points, tck = spline.compute_spline_coeff(ma,np.zeros_like(ma),anch)

    diagonal = int(np.sqrt(ma.shape[0]**2+ma.shape[1]**2)/2)
    mid, tangent, width = midline.compute_midline_and_tangent(anch,N_points,tck,diagonal)
    return mid[margin:-margin]


def get_midline_length(mid):
    length = 0
    for i in range(len(mid)-1):
        length += np.sqrt((mid[i][0]-mid[i+1][0])**2 + (mid[i][1]-mid[i+1][1])**2)
    return length




# END CODE MODIFIED FROM MORGANA

import matplotlib.pyplot as plt 

def divide_Nlabels(mas, N):
    successful = False
    N_it = 0

    while (True):
        successful = True
        
        mid = get_midline(mas)
        tck,u = interpolate.splprep([mid[:,1], mid[:,0]],k=2)
        xnew = np.linspace(0,1,N+1) ## edges of the division
        xnew_cen = np.linspace(0.05,0.95,N) # centers of teh divisions

        edges = np.array(interpolate.splev(xnew,tck)).T
        centers = np.array(interpolate.splev(xnew_cen,tck)).T


        #print(edges)

        ## Drawing the division lines
        lines = np.zeros_like(mas,dtype=int)
        for i in range(1,N):
            start_point = edges[i]
            angle = np.arctan2(edges[i-1][1]-edges[i+1][1],edges[i-1][0]-edges[i+1][0])

            pp = 0
            while True: # fill the separating line on one side
                i1 = int(start_point[1]+pp*np.cos(angle))
                i2 = int(start_point[0]-pp*np.sin(angle))
                if (mas[i1][i2]):
                    lines[i1][i2]=i+1
                    pp+=1
                else:
                    break
            pp = 1
            while True:
                i1 = int(start_point[1]-pp*np.cos(angle))
                i2 = int(start_point[0]+pp*np.sin(angle))
                if (mas[i1][i2]):
                    lines[i1][i2]=i+1
                    pp+=1
                else:
                    break
        mask_line = np.array(mas,dtype=int)
        mask_line += lines

        for i in range(N):
            if not (mas[int(centers[i][1]),int(centers[i][0])]):
                successful = False
                print("/!\ Error, retrying")
                break
            flood_fill(mask_line,(int(centers[i][1]),int(centers[i][0])),i+2,in_place=True,connectivity = 1)
        mask_line[mask_line==1] = 0 # removing small chunk not enough connected
        mask_line = mask_line-1
        mask_line[mask_line==-1] = 0

        #### Sorting by biggest first
        A1 = 0
        A2 = 0

        N_av = int(N/2)
        for i in range(N_av):
            A1+=np.sum(mask_line==i+1)
            A2+=np.sum(mask_line==N-i)
        if (A2>A1):
            mask_line_ordered = np.zeros_like(mask_line)
            for i in range(N):
                mask_line_ordered[mask_line==i+1] = N-i
        else:
            mask_line_ordered = mask_line

        N_it +=1

        if (successful or N_it>10):
            break



    if successful:    
        return mask_line_ordered,get_midline_length(mid),mid
    else:
        print("/!\ ABORTED")
        return np.zeros_like(mask_line_ordered), 0,0
