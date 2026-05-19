# HTH_GasSeg

Thomas Combriat, HTH, UiO, 2026

## Morphological evaluation of gastruloids.

GasSeg is a small program to compute some shape estimators of gastruloids from brightfields images. It is also capable of computing the fluorescence spatial repartition.

### Procedure

The gastruloid is first segmented, usually using a combination of machine learning and [SAM](https://segment-anything.metademolab.com/). The algorithm expects only one gastruloid per image, and will only segment the biggest one if several objects are present. Images for which the segmented objects touches any edges of the image are considered invalid.

The midline of the object is then computed using an algorithm from [MOrgAna](https://github.com/LabTrivedi/MOrgAna), and the object is divided into a number of segment (10 by default) alongside the midline.

The size of each of these segments is computed alongside the fluorescence intensities if some fluorescence images are present.



### File list

Informations on how to run the different scripts are given at the top of the files. A brief description

 - `analyse_img_in_folder.py`: main script. Computes the morphological features and fluorescence intensities of the images.
 - `train_Segmenter.py`: can be used to train a new ML segmenter on different images.
 - `src/Segmenter.py`: the class file for the Segmentation objects.
 - `src/midline.py`: utility file for extracting the midline of a segmented object.
 - `src/ellipsis.py`: utility file for extracting the ellipsis parameter and long axis of a segmented object.
 - `trained_models/`: folder containing some ML trained models that can be used by `analyse_img_in_folder.py`.
 
 
### Output examples

![](img/ex_output.png)
