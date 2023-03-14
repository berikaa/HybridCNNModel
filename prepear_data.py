import numpy as np
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf

DATADIR = '/content/gdrive/My Drive/CNN_dataset_deneme/1000_adet_veri'
DATADIR_TEST =  '/content/gdrive/My Drive/CNN_dataset_deneme/yeni_test_data'

CATEGORIES = ["all", "hem"]
IMG_SIZE = 32

def create_training_data():
    training_data = []
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
     return  training_data
  
def train_data():      
    X = []
    y = []
    for features,label in create_training_data():
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,3)

    #with cross valid
    kf = KFold(n_splits=5, random_state=42, shuffle=False)
    for train, test in kf.split(X):
        y=np.array(y)
        train_filenames, val_filenames=X[train], X[test]
        train_labels, val_labels= y[train], y[test]

    train_labels=np.array(train_labels)
    val_labels=np.array(val_labels)

    #or with split
    from sklearn.model_selection import train_test_split
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(X, y, test_size=0.15, random_state=42)
    
    return train_filenames, val_filenames,train_labels,val_labels

def create_test_data():
      test_filenames = []

        test_path = os.path.join(DATADIR_TEST)  
    
        for img in tqdm(os.listdir(test_path)):  
            try:
                test_img_array = cv2.imread(os.path.join(test_path,img))  # convert to array
                test_new_array = cv2.resize(test_img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_filenames.append([test_new_array])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
    return test_filenames

def tes_data():
  test_data = []
  for features in create_test_data():
      test_data.append(features)
  test_data = np.array(test_data).reshape(-1, IMG_SIZE, IMG_SIZE,1)
  return test_data
