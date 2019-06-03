import numpy as np
from PIL import Image
from skimage import transform
import os, os.path

def load_image(filename):
   return Image.open(filename)

def load_images_from_dir(path):
    test_images = []
    
    for f in os.listdir(path):
        test_images.append(load_image(os.path.join(path, f)))    
        
    return test_images

def prepare_image(image):
   image = np.array(image).astype('float32') / 255
   image = transform.resize(image, (128, 128, 3))
   image = np.expand_dims(image, axis = 0)
   
   return image