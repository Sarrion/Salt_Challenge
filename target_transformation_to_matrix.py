# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:42:57 2018

@author: avicent
"""
import os
from PIL import Image
import numpy as np
import pandas as pd
#import math

# =============================================================================
#     CODE CHUNK TO TRANSFORM TRAIN IMAGES
# =============================================================================
images = np.zeros((4000, 10201))
list_images = os.listdir('data/train/images')
for i in range(0, len(list_images)):
    image = np.array(Image.open("data/train/images/" + list_images[i]))[:,:,1]
    image = image.reshape(1, 101**2)
    images[i, :] = image
images = images/255
#np.save('images.npy', images)

# =============================================================================
#     CODE CHUNK TO TRANSFORM TARGET
# =============================================================================
target_csv = pd.read_csv('data/train.csv')
target_csv = target_csv.sort_values(by = "id")
rle_mask = target_csv.rle_mask.str.split(' ')

targets = np.zeros((4000, 10201))
for i in range(1,4000):
    if(type(rle_mask[i]) != float):
        for j in range(0, len(rle_mask[i]), 2):
            initio = int(rle_mask[i][j]) - 1
            finale =  initio + int(rle_mask[i][j + 1])
            targets[i, initio:finale] = 1
# np.save('targets.npy', targets)        
            
