import numpy as np
from scipy import interpolate
from scipy.ndimage import map_coordinates
from skimage.segmentation import flood_fill
from scipy.ndimage import label
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops, label

import matplotlib.pyplot as plt 

def divide_Nlabels_ell(mas,N):
    lab = label(mas)
    props = regionprops(lab)[0]
    centroid = props.centroid
    #print(centroid)
    #plt.figure()
    #plt.imshow(mas)
    
    #plt.plot([centroid[1],centroid[1]+props.axis_major_length/2*np.sin(props.orientation)],[centroid[0],centroid[0]+props.axis_major_length/2*np.cos(props.orientation)])
    #plt.show()
    #mid = np.array([[centroid[1]-props.axis_major_length/2*np.sin(props.orientation),centroid[1]+props.axis_major_length/2*np.sin(props.orientation)], [centroid[0]-props.axis_major_length/2*np.cos(props.orientation),centroid[0]+props.axis_major_length/2*np.cos(props.orientation)]])
    mid = np.array([[centroid[0]-props.axis_major_length/2*np.cos(props.orientation),centroid[0]+props.axis_major_length/2*np.cos(props.orientation)],[centroid[1]-props.axis_major_length/2*np.sin(props.orientation),centroid[1]+props.axis_major_length/2*np.sin(props.orientation)]]).T

    tck,u = interpolate.splprep([mid[:,1], mid[:,0]],k=1)
    xnew = np.linspace(0,1,N+1) ## edges of the division
    xnew_cen = np.linspace(0.05,0.95,N) # centers of teh divisions
    
    edges = np.array(interpolate.splev(xnew,tck)).T
    centers = np.array(interpolate.splev(xnew_cen,tck)).T

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





  
    return mask_line_ordered,props.axis_major_length,mid,props.eccentricity


