
# import all necesary lib
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import skimage.color
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.layers import flatten
import pickle
def normalize(img):
    img = (img- np.mean(img))/(np.std(img)+ np.finfo('float32').eps)
    return img
def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2YUV)[:,:,0]
    #return skimage.color.rgb2gray(img)

def adjust_contract(img):
    return cv2.equalizeHist(img)

def preprocess(X):
    processed_img = np.empty((X.shape[0],X.shape[1],X.shape[2]),dtype=np.float32)
    for i in range(X.shape[0]):
        processed_img[i] = normalize(adjust_contract(to_gray(X[i])))
    return processed_img
BATCH_SIZE = 128
def predict(X_data):
    num_examples = len(X_data)
    predicted_results = []
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset+BATCH_SIZE]
        possiblility = sess.run(accuracy_operation, feed_dict={x: batch_x,keep_prob:1.00})
        predicted_results.append(possiblility)
    return predicted_results

def predict_id(img):
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img,axis=3)
    pred_result = predict(img)
    return pred_result[0]

sign_df = pd.read_csv('signnames.csv')
sign_df.set_index('ClassId')

test_images = []
test_images_sign_id = []
path = './test_images/'

plt.figure(figsize=(15, 20))

for image in os.listdir(path):
    img = cv2.imread(path + image)

    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_images.append(img)
    test_images_sign_id.append(int(image.split("-", 1)[0]))

test_images = np.array(test_images)
# test_images = test_images/255

test_images = preprocess(test_images)
# print(sign_namestest_images_sign_serial_no)
for i, image in enumerate(test_images):
    grid = plt.subplot(1, 10, i + 1)
    # grid = plt.subplot(len(test_images)/4,5,i+1)
    grid.imshow(image), plt.axis('off')

plt.show()