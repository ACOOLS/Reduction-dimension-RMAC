from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential,Model, load_model
import numpy as np
import os
import csv
#import keras
import operator
import ntpath
import tensorflow as tf
import pickle
import warnings
from operator import itemgetter
from tensorflow.keras.optimizers import Adam
import argparse
from shutil import copyfile
import os.path
from os import path
from matplotlib.pyplot import imread
#from keras.backend.tensorflow_backend import set_session
from scipy import spatial
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten,Lambda
#from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions #299*299
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions #224*224
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input,decode_predictions# input shape= 299x299
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input,decode_predictions# input shape= 299x299
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions# input shape= 224x224 
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, SGD

# For automated test log result directly in the csv
import csv
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import math
import argparse
import matplotlib
import imghdr
import pickle as pkl
import datetime
import urllib.request
from cycler import cycler
from PIL import Image, ImageEnhance
import zipfile
import shutil


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Indexation

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--dossier', type=str,default='Results',
        help='path to image requete '
    )
    parser.add_argument(
        '--base', type=str,default='new_cars_cools',
        help='modele  '
    )
    parser.add_argument(
        '--gpu', type=int,default=0,
        help='gpu number '
    )
    FLAGS = parser.parse_args()
    gpu_number=FLAGS.gpu
    if gpu_number == 0 :
        classifier_list = ["Xception","MobileNet","ResNet50"]
    elif gpu_number == 1 :
        classifier_list = ["InceptionV3","InceptionResNetV2"]
    elif gpu_number == 2 :
        classifier_list = ["VGG16","VGG19","DenseNet169"]
    elif gpu_number == 3 :
        classifier_list = ["DenseNet121","DenseNet201"]

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
        except RuntimeError as e:
            print(e)
    datasets = FLAGS.base
    dossier = FLAGS.dossier
    old_dataset="../../DataSets/"+datasets

    for id_classifier in range(0,len(classifier_list)):
        classifier=classifier_list[id_classifier]
        if classifier=="InceptionResNetV2" or classifier=="SqueezeNet" or classifier=="Xception" or classifier=="InceptionV3" or classifier=="InceptionResNetV2" or classifier =="DenseNet121" or classifier =="DenseNet169" or classifier=="DenseNet201" :
            img_height = 299
            img_width = 299
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)
        elif classifier=="VGG16" or classifier=="VGG19" or classifier=="ResNet50" or classifier=="MobileNet" or classifier=="NASNetMobile" :
            img_height = 224
            img_width = 224
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)
        elif classifier=="NASNetLarge" :
            img_height = 331
            img_width = 331
            input_size=(img_height,img_height)
            input_shape=(img_width,img_height,3)  
        

        result_path = "../../normal/Cars/"+dossier+"/"+classifier
        model_final_of_classification=os.path.join(result_path, classifier+'_final.h5')
        model=load_model(model_final_of_classification)
        #print("----------")
        #print(model.layers[-4].name)
        x = model.layers[-4].output
        x = Flatten()(x)
        model2 = Model(inputs=model.inputs, outputs=x)
        model2.summary()
        #quit()
        
        # indexation 

        features = [] #Stocker les caractérstiques
        folder_features="Features_flatten"+dossier
        classifier_path=folder_features+"/"+classifier
        #features_path = classifier_path +"/features"
        if os.path.exists(classifier_path) == False:
            os.makedirs(classifier_path)
            features_path = classifier_path +"/features"
            if os.path.exists(features_path) == False:
                os.makedirs(features_path)
            pas =0
            for k in os.listdir(old_dataset) :
                data1 = os.path.join(old_dataset, k)
                print(data1)
                for j in os.listdir(data1):
                    if not j.endswith(".jpg"):
                        continue
                    print(j)
                    num_marque, num_modele, marque, modele, number  = j.split("_") 
                    number = number.split(".")[0]
                    number = int(number)
                    data = os.path.join(data1, j)
                    #print (data)
                    if not data.endswith(".jpg"):
                        continue
                    file_name = os.path.basename(data)
                    # load an image from file
                    image = load_img(data, target_size=input_size)
                    # convert the image pixels to a numpy array
                    image = img_to_array(image)
                    # reshape data for the model
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    # prepare the image for the VGG model
                    image = preprocess_input(image)
                    # predict the probability across all output classes
                    # Model 1
                    feature = model2.predict(image)
                    feature = np.array(feature[0])  
                    np.savetxt(features_path+"/"+os.path.splitext(file_name)[0]+".txt",feature)
                    features.append((data,feature,number))
                    #features.append((number,feature))
                    print (pas)
                    pas = pas+1
                features=sorted(features,key=itemgetter(0))
                with open(classifier_path+"/features.txt", "wb") as output:
                    pickle.dump(features, output)
    
