import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from tqdm import tqdm

#DATA = '/content/gdrive/My Drive/CNN_dataset_deneme/test_data'

def plot_image(input, predictions_array):            
    
    
    predictions_array = predictions_array[input]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    
    
    kanser_riski= 100*np.max(predictions_array) 
   
    if kanser_riski>50:
       sonuc=class_names[0]
    else:
       sonuc=class_names[1]
  
  
    print("kanser riski: %", kanser_riski)
    print("sonuç:",sonuc)
    print("Görsel:")
    sonuc_path = os.path.join(DATADIR_TEST_3)  

    sonuc_img_array= cv2.imread(os.path.join(sonuc_path,os.listdir(sonuc_path)[input]))

    sonuc_new_array = cv2.resize(sonuc_img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(sonuc_new_array, cmap='gray')
    plt.show()
