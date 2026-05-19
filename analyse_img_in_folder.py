#!/usr/bin/python3


import matplotlib.pyplot as plt 
import numpy as np  # numerics
import math as math  # math
import os as os
import pickle
import sys
from tqdm import tqdm
import signal
from src.Segmentation import *
from src.midline import *
from src.ImUtils import *
from src.ellipsis import *



"""
Copyright: Combriat Thomas, 2026
HTH, University of Oslo

This script performs the segmentation of the gastruloid, divides it into 10 segments and computes the area and fluorescence intensities for each segment.

Dependencies of this script use some code adapted from Morgana and do not claim any authorship on these (see src/midline.py)
https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/morphology/computemorphology.py
https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/straightmorphology/computestraightmorphology.py

Usage:
./analyse_img_in_folder.py folder -args
with:
 - folder being where the data is stored. It should contain at least one subfolder "BF" containing the brightfield images. "Red" and "Green" subfolders can also be present if the intensities of fluorescence in these channels should be computed.
 - -args can be use to select the segmentation method and the model to use. The segmentation method can be chosen by passing "-useSAM" or "-useMIX". If none is given, a pure machine learning model is used. "-useSAM" uses only the Segment Anything Model to segment the gastruloid whereas "-useMIX" uses a combination of ML and SAM, using ML to guide SAM towards interesting objects.
 - -model XXX can be used to select another ML model than the default one. XXX should be the filename of the model contained in the "trained_models" folder, and generated with the "train_Segmenter.py" script.

Output:
This script creates two folders, alongside the data:
 - results, which will contain a single text file with the numerical values of the results.
 - masks, which will contain visuals of the generated masks with simple graphs for sanity checks.
"""



useSAM = False
segmentation_method = "ML"
model_name = "cB_Natalia_intensity.pic"
if (len(sys.argv)>2): #we have extra arguments
    for i in range(2,len(sys.argv)):
        if (sys.argv[i] == "-useSAM"):
            useSAM = True
            segmentation_method = "SAM"
            print("/!\\ Using SAM")
        if (sys.argv[i] == "-useMIX"):
            segmentation_method = "MIX"
            print("/!\\ Using MIX SAM/ML")
        if (sys.argv[i] == "-model"):
            model_name = sys.argv[i+1]
            print("/!\\ Using the model:" + model_name)
        
            



## Loading the segmenter
#segmenter_file = open("trained_models"+os.sep+"cB_incucyte_blur.pic","rb") # Note: not the same than for the widefield

if (segmentation_method == "ML"):
    segmenter_file = open("trained_models"+os.sep+model_name,"rb") # Note: not the same than for the widefield
    Segmenter = pickle.load(segmenter_file)
    segmenter_file.close()
elif (segmentation_method == "SAM"):
    Segmenter = get_sam_segmenter()

elif (segmentation_method == "MIX"):
    segmenter_file = open("trained_models"+os.sep+model_name,"rb") # Note: not the same than for the widefield
    Segmenter_ML = pickle.load(segmenter_file)
    segmenter_file.close()
    Segmenter_SAM = get_sam_segmenter()


curved_midline = True ## if true we do not assume that the gastruloid can be fitted by an ellipse.



timout = 150 #in sec


def handler(signum, frame):
    raise Exception("Segmentation timout")
signal.signal(signal.SIGALRM, handler)



## Data parsing
#folder = os.sep+"home"+os.sep+"tom"+os.sep+"data"+os.sep+"Igor"+os.sep+"IncuSeg"+os.sep+"Plate6"+os.sep+"4DPA"
folder = sys.argv[1]
if (folder[-1] != os.sep):
    folder += os.sep


folder_BF = folder + "BF" + os.sep
ls = os.listdir(folder_BF)


ls_filtered = []
for i in ls:
    if (".tif" in i.lower()):
        ls_filtered.append(i)
        
ls = ls_filtered
ls.sort()
try:
    os.mkdir(folder+os.sep+"results")
except:
    pass

try:
    os.mkdir(folder+os.sep+"masks")
except:
    pass

fig = plt.figure(figsize=(20,10))

## Creating the output file
N_seg = 10

if not (curved_midline):
    output = open(folder+os.sep+"results"+os.sep+"results_ell.txt","w")
    output.write("#File Area MidLineLength Eccentricity")
else:
    output = open(folder+os.sep+"results"+os.sep+"results_midline.txt","w")
    output.write("#File Area MidLineLength ")
for j in range(N_seg): 
    output.write(" Seg"+str(j)+"_Area MeanRed MeanGreen")
output.write("\n")
### Looping on all the images of the folder
N=0
for i in ls:
    print(N,len(ls),end="")
    print(" Treating:", i)
    img = io.imread(folder_BF+i)
    
    ### Normalizing to 8bits in case we have EVOS data
    if (img.dtype==np.uint16):
        img = np.array(img,dtype=float)
        img /= np.amax(img)/255.
        img = np.array(img,dtype=np.uint8)
        img = img[:,:,0]
    try:
        img_red = io.imread(folder+"Red"+os.sep+i)
        if (img_red.ndim ==3):
            img_red = img_red[:,:,0]
    except:
        print("/!\ Red image not found for:",i)
        img_red = np.zeros_like(img)
    try:
        img_green = io.imread(folder+"Green"+os.sep+i)
        if (img_green.ndim==3):
            img_green = img_green[:,:,0]
    except:
        print("/!\ Green image not found for:",i)
        img_green = np.zeros_like(img)

    signal.alarm(timout)
    try:
        if (segmentation_method=="SAM"):
            mask = segment_biggest_sam(img,Segmenter,fill=True,inv=True)
        elif (segmentation_method=="ML"):
            mask = segment_biggest(img,Segmenter)
        elif (segmentation_method=="MIX"):
            mask = segment_biggest_sam_ML(img, Segmenter_SAM, Segmenter_ML)
    except:
        print("/!\ SEGMENTATION TIMOUT")
        mask = np.zeros_like(img)
    signal.alarm(0)
    green_seg = np.zeros(N_seg)
    red_seg = np.zeros(N_seg)
    sizes_seg = np.zeros(N_seg)
    ## Checking if the mask is valid
    if (np.sum(mask)>10000 and np.sum(mask)<(len(mask)*len(mask[0]))/2):
        signal.alarm(timout*4)
        try:
            if (curved_midline):
                div,length,mid = divide_Nlabels(mask,N_seg)
            else:
                div,length,mid,ecc = divide_Nlabels_ell(mask,N_seg)
        except:
            print("/!\ MIDLINE TIMOUT")
            length=0
        signal.alarm(0)
        if (length > 0):
            if not (curved_midline):
                output.write("%s %i %f %f" % (i, np.sum(mask), length,ecc))
            else:
                output.write("%s %i %f" % (i, np.sum(mask), length))
            for n in range(N_seg):
                output.write(" %i" % (np.sum(div==n+1)))
                sizes_seg[n] = np.sum(div==n+1)
                if (img_red !=0).all():
                    output.write(" %f" % (np.mean(img_red[div==n+1])))
                    red_seg[n] = np.mean(img_red[div==n+1])
                    
                else:
                    output.write(" %f" % (0))
                if (img_green !=0).all():
                    output.write(" %f" % (np.mean(img_green[div==n+1])))
                    green_seg[n] = np.mean(img_green[div==n+1])
                else:
                    output.write(" %f" % (0))
                    
            output.write("\n")
            
            
            plt.subplot(221)
            plt.imshow(img,cmap="gist_gray")
            plt.contour(mask)
            plt.subplot(222)
            plt.imshow(div)
            plt.scatter(mid[:,1],mid[:,0])

            plt.subplot(289)
            if (img_green !=0).all():
                
                #plt.imshow(np.array(img_green,dtype=float)/np.amax(img_green)*mask,cmap = "inferno")
                img_green_renor = np.array(img_green,dtype=float)
                img_green_renor -= np.amin(img_green_renor[mask])
                img_green_renor /= np.amax(img_green_renor[mask])
                plt.imshow(img_green_renor*mask,cmap="inferno")
                plt.title("Green masked")
                plt.subplot(2,8,10)
                img_green_renor[img_green_renor>1] = 1
                img_green_renor[img_green_renor<0] = 0
                plt.imshow(img_green_renor,cmap="inferno")
                plt.title("Green")
            plt.subplot(2,8,11)
            if (img_red !=0).all():
                #plt.imshow(np.array(img_red,dtype=float)/np.amax(img_red)*mask,cmap = "inferno")
                img_red_renor = np.array(img_red,dtype=float)
                img_red_renor -= np.amin(img_red_renor[mask])
                img_red_renor /= np.amax(img_red_renor[mask])
                plt.imshow(img_red_renor*mask,cmap="inferno")
                plt.title("Red masked")
                plt.subplot(2,8,12)
                img_red_renor[img_red_renor>1] = 1
                img_red_renor[img_red_renor<0] = 0
                plt.imshow(img_red_renor,cmap="inferno")
                plt.title("Red")
            
            plt.subplot(224)
            plt.plot(sizes_seg/np.amax(sizes_seg),label="Segment sizes, tot="+str(np.sum(mask)),c="k")
            plt.plot((red_seg-np.amin(red_seg))/np.amax(red_seg-np.amin(red_seg)),label = "red",c="red")
            plt.plot((green_seg-np.amin(green_seg))/np.amax(green_seg-np.amin(green_seg)),label = "green",c="green")
            plt.legend()
            plt.tight_layout()
            plt.savefig(folder+os.sep+"masks"+os.sep+i+".png")
            plt.clf()
        
        else:
            print("No mid-line!")
            plt.subplot(221)
            plt.imshow(img)
            plt.subplot(222)
            #plt.imshow(mask)
            if (segmentation_method=="ML"):
                prob = Segmenter.get_proba(img)
                plt.imshow(prob)
                plt.colorbar()
            plt.subplot(223)
            plt.imshow(mask)
            plt.subplot(224)
            #plt.imshow(div)
            #plt.scatter(mid[:,1],mid[:,0])
            plt.savefig(folder+os.sep+"masks"+os.sep+i+"_invalid.png")
            plt.clf()
    else:
        print("No valid mask!")
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(mask)
        if (segmentation_method=="ML"):
            plt.subplot(224)
            prob = Segmenter.get_proba(img)
            plt.imshow(prob)
            plt.colorbar()
        plt.savefig(folder+os.sep+"masks"+os.sep+i+"_invalid.png")
        plt.clf()
        
    N+=1


output.close()
