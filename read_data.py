import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from tqdm import tqdm

DATADIR = '/content/gdrive/My Drive/CNN_dataset_deneme/1000_adet_veri'

CATEGORIES = ["all", "hem"]

for category in CATEGORIES:  
    path = os.path.join(DATADIR ,category)  
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img)) 
        plt.imshow(img_array, cmap='gray')  
        plt.show() 
        break 
    break   

IMG_SIZE = 32

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
