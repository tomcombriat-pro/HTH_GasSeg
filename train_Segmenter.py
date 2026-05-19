#!/usr/bin/python3

"""
Copyright: Combriat Thomas, 2026
HTH, University of Oslo

This script trains a classifier on pre-labelled data for segmentation.
"""



import matplotlib.pyplot as plt 
import numpy as np  # numerics
import math as math  # math
import os as os
import random

from skimage.segmentation import morphological_chan_vese
from skimage.segmentation import chan_vese
from skimage.segmentation import inverse_gaussian_gradient
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.filters import threshold_otsu
from skimage.filters import laplace
from skimage.feature import canny
import skimage.future as ft
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from skimage.feature import multiscale_basic_features
import catboost as cb
import skimage.io as io
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import pickle
import sys

from src.Segmentation import *


useGPU = False
N_it = 1000
folder = sys.argv[1]
out_name = sys.argv[2]
if (len(sys.argv)>3):
    for i in range(3, len(sys.argv)):
        if (sys.argv[i] == "-GPU"):
            useGPU = True
            print("/*\ GPU will be used")
        if (sys.argv[i] == "-N_it"):
            N_it = int(sys.argv[i+1])
            print("/*\ N_it set to: ", N_it)
ls = os.listdir(folder)




ls_no_labels = []
for i in ls:
    if not ("labels" in i):
        ls_no_labels.append(i)
random.shuffle(ls_no_labels)


if (useGPU):
    mySegmenter =  Segmenter(clf=cb.CatBoostClassifier(task_type="GPU",devices="0",iterations=N_it))
else:
    mySegmenter = Segmenter(clf=cb.CatBoostClassifier(iterations=N_it))


mySegmenter.cf = cf0


N_training = int(len(ls_no_labels)*.8)
N_test = 1
N_validate = len(ls_no_labels)-N_training-N_test



training_imgs = []
training_features = []
training_labels = []

for i in range(N_training):
    training_imgs.append(io.imread(folder+os.sep+ls_no_labels[i]))
    training_labels.append(io.imread(folder+os.sep+ls_no_labels[i][0:-4]+"_labels.tif"))

for i in training_imgs:
    training_features.append(mySegmenter.cf(i))
training_features = np.concatenate(training_features)
training_labels = np.concatenate(training_labels)


validating_imgs = []
validating_features = []
validating_labels = []

for i in range(N_training,N_training+N_validate):
    validating_imgs.append(io.imread(folder+os.sep+ls_no_labels[i]))
    validating_labels.append(io.imread(folder+os.sep+ls_no_labels[i][0:-4]+"_labels.tif"))

for i in validating_imgs:
    validating_features.append(mySegmenter.cf(i))
validating_features = np.concatenate(validating_features)
validating_label = validating_labels[0]
validating_labels = np.concatenate(validating_labels)
  
testing_img = io.imread(folder+os.sep+ls_no_labels[-1])
testing_labels = io.imread(folder+os.sep+ls_no_labels[-1][0:-4]+"_labels.tif").flatten()


mySegmenter.fit(training_features,training_labels.flatten(),eval_set = (validating_features,validating_labels.flatten()))


### Finding the optimal threshold

preds = []
for i in validating_imgs:
    preds.append(mySegmenter.clf.predict_proba(mySegmenter.cf(i))[:,1])
pred = np.concatenate(preds)



prec, rec, th_rp = precision_recall_curve((validating_labels==1).flatten(),pred)
fscore = (2 * prec * rec) / (prec + rec)
        
# locate the index of the largest f score
fscore = np.nan_to_num(fscore)
print("Optimal threshold:", th_rp[np.argmax(fscore)])
mySegmenter.threshold = th_rp[np.argmax(fscore)]

pred = mySegmenter.clf.predict_proba(mySegmenter.cf(testing_img))

fpr,tpr,th = roc_curve(testing_labels==1,pred[:,1])
print("AUC:", auc(fpr,tpr))
#plt.figure()

#plt.plot(fpr,tpr)



### Outputting the trained model

output_model= open("trained_models"+os.sep+out_name+".pic","wb")
pickle.dump(mySegmenter,output_model)
output_model.close()
