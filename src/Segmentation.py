import catboost as cb
from skimage.feature import multiscale_basic_features
from skimage.measure import label
import skimage.io as io
from skimage.filters import threshold_otsu
from functools import partial
import numpy as np
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def cf0(mat):
    return multiscale_basic_features(mat,intensity = False).reshape((len(mat)*len(mat[0]),-1))

def cf1(mat):
    return multiscale_basic_features(mat,intensity = False).reshape((len(mat)*len(mat[0]),-1))



class Segmenter:

    def __init__(self, clf=cb.CatBoostClassifier(),cf = cf0):
        self.clf = clf
        self.cf = cf
        self.threshold = 0.5

    def fit(self,training_features, training_labels, **kwargs):
        self.clf.fit(training_features,training_labels,**kwargs)
        

    def segment(self, img):
        pred = self.clf.predict_proba(self.cf(img)) > self.threshold
        if (img.ndim == 3):
            shape = (len(img),len(img[0]))
        else:
            shape = img.shape
        return pred[:,1].reshape(shape)

    def get_proba(self,img):
        pred = self.clf.predict_proba(self.cf(img))
        if (img.ndim == 3):
            shape = (len(img),len(img[0]))
        else:
            shape = img.shape
        return pred[:,1].reshape(shape)
        

    

def segment_fill(img, segmenter):
    pred = segmenter.segment(img)
    pred = binary_fill_holes(pred)
    return pred


        
def segment_biggest(img,segmenter):
    pred = segment_fill(img,segmenter)
    labels,num = label(pred,return_num = True)
    biggest_size = 0
    biggest_label = 0

    for i in range(1,num+1):
        current_size = np.sum(labels==i)
        if (current_size > biggest_size):
            biggest_size = current_size
            biggest_label = i
    if (biggest_label != 0):
        ret = labels == biggest_label
        # Checking if that touches the edge
        if (np.sum(ret[:,0])==0 and np.sum(ret[:,-1])==0 and np.sum(ret[0,:])==0 and np.sum(ret[-1,:])==0):
            return ret
        

    return np.zeros_like(img)


def segment_biggest_otsu(img):
    #pred = segment_fill(img,segmenter)
    pred = img < threshold_otsu(img)
    pred = binary_fill_holes(pred)
    labels,num = label(pred,return_num = True)
    biggest_size = 0
    biggest_label = 0

    for i in range(1,num+1):
        current_size = np.sum(labels==i)
        if (current_size > biggest_size):
            biggest_size = current_size
            biggest_label = i
    if (biggest_label != 0):
        ret = labels == biggest_label
        # Checking if that touches the edge
        if (np.sum(ret[:,0])==0 and np.sum(ret[:,-1])==0 and np.sum(ret[0,:])==0 and np.sum(ret[-1,:])==0):
            return ret
        

    return np.zeros_like(img)
    
    
        




# Downloading the SAM model
def download_sam_checkpoint(model_type="vit_h"):
    """Download SAM checkpoint if not present"""
    import os
    import urllib.request
    
    checkpoints = {
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "filename": "sam_vit_b_01ec64.pth"
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "filename": "sam_vit_l_0b3195.pth"
        },
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "filename": "sam_vit_h_4b8939.pth"
        }
    }
    
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[model_type]["filename"])
    
    if not os.path.exists(checkpoint_path):
        print(f"Downloading {model_type} checkpoint...")
        urllib.request.urlretrieve(checkpoints[model_type]["url"], checkpoint_path)
        print(f"Downloaded to {checkpoint_path}")
    
    return checkpoint_path

# Imports SAM
def get_sam_segmenter(model_type="vit_h"):
    """Initialize and return SAM segmenter with MPS support"""

    # Check for available devices in order of preference
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    
    # Auto-download checkpoint if needed
    checkpoint_path = download_sam_checkpoint(model_type)
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Force float32 for MPS compatibility
    if device.type == "mps":
        sam = sam.float()  # Ensure model uses float32
    
    # Create mask generator with float32 points
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,
        pred_iou_thresh=0.88 , #default 0.88
        stability_score_thresh=0.9 , #default 0.92
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        box_nms_thresh=0.7,
        #min_mask_region_area=10000,
    ) 
    
    # Monkey patch to fix MPS float64 issue only for mps problems.
    if device.type == "mps":
        # Store original methods
        original_process_batch = mask_generator._process_batch
        original_torch_as_tensor = torch.as_tensor
        
        def safe_as_tensor(data, dtype=None, device=None):
            """Convert to tensor, forcing float32 on MPS devices"""
            if device is not None and hasattr(device, 'type') and device.type == 'mps':
                if isinstance(data, np.ndarray) and data.dtype == np.float64:
                    data = data.astype(np.float32)
                if dtype == torch.float64:
                    dtype = torch.float32
            return original_torch_as_tensor(data, dtype=dtype, device=device)
        
        def patched_process_batch(points, im_size, crop_box, orig_size):
            # Convert all inputs to float32
            points = points.astype(np.float32)
            
            # Temporarily replace torch.as_tensor to handle all tensor creation
            torch.as_tensor = safe_as_tensor
            try:
                result = original_process_batch(points, im_size, crop_box, orig_size)
            finally:
                # Restore original torch.as_tensor
                torch.as_tensor = original_torch_as_tensor
            
            return result
        
        mask_generator._process_batch = patched_process_batch
    
    return mask_generator


    

def segment_sam(img, segmenter, min_area=2500):
    """Segment all objects with SAM above min_area threshold"""
    # Convert to RGB if grayscale
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img
    
    # Check if image has enough contrast
    if img.std() < 5:  # Adjust threshold as needed
        print("Skipping very low contrast image")
        return np.zeros(img.shape[:2], dtype=np.uint16)
        
    # SAM (and torchvision transforms used internally) expect the image to be
    # a floating point tensor.  On Apple-Silicon (MPS) backends in particular
    # an integer tensor will raise the error that the user observed:
    #   "Input tensor should be a float tensor. Got <dtype>".
    # We therefore make sure the numpy array we pass to SAM is
    # float32 in the 7 range.  If the dynamic range is 0-255 we rescale to 0-1.
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32)

    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / img_rgb.max()
    
    # Debugging and continues segmenting when SAM handles low contrast pictures.
    try:
        masks = segmenter.generate(img_rgb)
    except IndexError as e:
        print(f"SAM internal error (likely edge case image): {e}")
        return np.zeros(img.shape[:2], dtype=np.uint16)
    except Exception as e:
        print(f"SAM failed: {e}")
        return np.zeros(img.shape[:2], dtype=np.uint16)
    
    # Create label image with correct shape
    labels = np.zeros(img.shape[:2], dtype=np.uint16)
    label_idx = 1
    
    # Sort by area to process larger masks first
    for mask in sorted(masks, key=lambda x: x['area'], reverse=True):
        seg = mask['segmentation']
        area = mask['area']
        
        if area < min_area:
            continue
            
        # Skip if touches border (within 4 pixels of edge)
        if (np.any(seg[0:4,:]) or      # top 4 rows
            np.any(seg[-4:,:]) or       # bottom 4 rows
            np.any(seg[:,0:4]) or       # left 4 columns
            np.any(seg[:,-4:])):        # right 4 columns
            continue
            
        # Add to labels where not already labeled
        labels[seg & (labels == 0)] = label_idx
        label_idx += 1
    
    return labels


def segment_biggest_sam(img, segmenter, fill=False,inv=False,min_area = 2500):
    """Segment image with sam and return the biggest object"""

    labels = segment_sam(img,segmenter,min_area)
    num = np.amax(labels)
    biggest_size = 0
    biggest_label = 0

    for i in range(1,num+1):
        current_size = np.sum(labels==i)
        if (current_size > biggest_size):
            biggest_size = current_size
            biggest_label = i
    if (biggest_label != 0):
        ret = labels == biggest_label
        # Checking if that touches the edge
        if (np.sum(ret[:,0])==0 and np.sum(ret[:,-1])==0 and np.sum(ret[0,:])==0 and np.sum(ret[-1,:])==0):
            return ret
        

    return np.zeros_like(img)
    """
    # Convert to RGB if grayscale
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img


    
    # SAM (and torchvision transforms used internally) expect the image to be
    # a floating point tensor.  On Apple-Silicon (MPS) backends in particular
    # an integer tensor will raise the error that the user observed:
    #   "Input tensor should be a float tensor. Got <dtype>".
    # We therefore make sure the numpy array we pass to SAM is
    # float32 in the 7 range.  If the dynamic range is 0-255 we rescale to 0-1.
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32)

    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / img_rgb.max()

    if (inv):
        img_rgb = 1-img_rgb


    
    try:
        masks = segmenter.generate(img_rgb)
    except Exception as e:
        print(f"SAM segmentation failed: {e}")
        return np.zeros(img.shape[:2], dtype=bool)
    
    if not masks:
        return np.zeros(img.shape[:2], dtype=bool)

    print(len(masks))
    # Find largest mask by area
    largest_mask = max(masks, key=lambda x: x['area'])
    pred = largest_mask['segmentation']


    #plt.imshow(masks)
    #plt.show()
    if fill:
        pred = binary_fill_holes(pred)
    print(pred.shape)
    return pred
    """


def segment_biggest_sam_ML(img,segmenter_sam,segmenter_ML):
    """
    Uses an homemade ML segmenter to hint SAM for correct segmentation.
    """
    mask_ML = segment_biggest(img,segmenter_ML)
    
    #Converting to RGB float for SAM
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img

    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32)

    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / img_rgb.max()
        
    predictor = segmenter_sam.predictor
    predictor.set_image(img_rgb)    
    input_point = np.array([np.flip(np.mean(np.where(mask_ML),axis=-1))])
    input_label = np.array([1])
    masks_predictor, scores,_ = predictor.predict(input_point,input_label, multimask_output=True)
    mask = masks_predictor[np.argmax(scores)]
    
    return mask
